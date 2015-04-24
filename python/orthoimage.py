#!/usr/bin/env python3

import sys
import os.path
import numpy as np
from skimage import io, draw
from itertools import chain, combinations

from PIL import Image, ImageDraw

import sensor_types
from project import Project

sys.path.append("build")
try: import pymodel0
except: print("Error importing pymodel0"); sys.exit(-1)

class WorldRect(object):
    """
    World bounding box aligned with the X/Y axes
    Represented by self.low and self.high
    which are lower-left and higer-right corners
    """

    def __init__(self, lower, higher):
        self.low, self.high = lower, higher

    @classmethod
    def from_points(cls, points):
        lower  = np.array([points[:,0].min(), points[:,1].min()], dtype=np.float64)
        higher = np.array([points[:,0].max(), points[:,1].max()], dtype=np.float64)
        return cls(lower, higher)

def image_bounds_mask(pixel_points, image_shape):
    """
    Returns a numpy mask (array of Bool) of size pixel_points.shape
    where each columns is True iff the pixel lies wihtin image bounds
    """

    contained_mask = np.empty(pixel_points.shape, dtype=bool)
    contained_mask[:,:] = np.all((
            # 0.5 because nearest neighbour interpolation rounds to closest integer
            pixel_points[:,0] >= 0,
            pixel_points[:,0] < (image_shape[0] - 0.5),
            pixel_points[:,1] >= 0,
            pixel_points[:,1] < (image_shape[1] - 0.5)
            ), axis=0)[:,np.newaxis]
    return contained_mask

def get_pixel_colors(image, pixel_points):
    "Read image value at float coordinate using nearest neighbour interpolation"
    indexes = np.asarray(np.round(pixel_points), dtype=int)
    return image[indexes[:,0], indexes[:,1]]

class FlatTile(object):
    def __init__(self, rect, gsd):
        margin = gsd*5 # border margin in meters
        self.origin = np.array([rect.low[0] - margin, rect.high[1] + margin]) # world coordinate of the (0,0) pixel
        rows = np.ceil((rect.high[1] - rect.low[1] + 2*margin) / gsd) # number of rows in image
        cols = np.ceil((rect.high[0] - rect.low[0] + 2*margin) / gsd) # number of cols in image
        self.image = np.zeros((rows, cols, 3), dtype=np.uint8)
        self.gsd = gsd

    def world_to_image(self, point):
        v = (point - self.origin) / self.gsd
        j = np.round(v[0])
        i = np.round(-v[1])
        return np.array([i, j], dtype=np.long)

    def image_to_world(self, points):
        "Converts a (n,2) array from image to world coordinates"
        x = np.array(( points[:,1] * self.gsd) + self.origin[0], dtype=np.float64)
        y = np.array((-points[:,0] * self.gsd) + self.origin[1], dtype=np.float64)
        return np.column_stack((x, y))

    def draw_world_point(self, point, color):
        i, j = self.world_to_image(point)
        self.image[i, j] = color

    def draw_cam_trace(self, corners):
        im = Image.fromarray(self.image)
        draw = ImageDraw.Draw(im)
        for c1, c2 in combinations(corners, 2):
            a = self.world_to_image(c1)
            b = self.world_to_image(c2)
            draw.line([tuple(a[::-1]), tuple(b[::-1])], fill="white", width=2)
        self.image = np.array(im, dtype=np.uint8)

    def draw_observations(self, internal, external, elevation, project, observations):
        im = Image.fromarray(self.image)
        draw = ImageDraw.Draw(im)
        sensors = sensor_types.pixel_to_sensor(observations, project.model.pixel_size, project.model.rows, project.model.cols)
        world_points = pymodel0.model0_inverse_array(internal, external, sensors, elevation)
        size = 3
        for point in world_points:
            i, j = self.world_to_image(point[:2])
            draw.ellipse([j-size, i-size, j+size, i+size], fill="blue", outline="black")
        self.image = np.array(im, dtype=np.uint8)

    def draw_obs_pair(self, internal, cam_a, cam_b, elevation, project, obs_a, obs_b):
        im = Image.fromarray(self.image)
        draw = ImageDraw.Draw(im)
        sensors_a = sensor_types.pixel_to_sensor(obs_a, project.model.pixel_size, project.model.rows, project.model.cols)
        sensors_b = sensor_types.pixel_to_sensor(obs_b, project.model.pixel_size, project.model.rows, project.model.cols)
        world_points_a = pymodel0.model0_inverse_array(internal, cam_a, sensors_a, elevation)
        world_points_b = pymodel0.model0_inverse_array(internal, cam_b, sensors_b, elevation)

        size = 3
        for (point_a, point_b) in zip(world_points_a, world_points_b):
            i_a, j_a = self.world_to_image(point_a[:2])
            i_b, j_b = self.world_to_image(point_b[:2])
            draw.line([j_a, i_a, j_b, i_b], fill="green", width=3)
        self.image = np.array(im, dtype=np.uint8)

    def project_camera(self, internal, external, elevation, pixel_size, image):
        """
        Orthorectify one image onto the tile
        The procedure works with 2D arrays of shape (n, 2),
        where each represents the same 'point' in different frames.
        """

        # All pixels in the tile image
        # 'ij' indexing and reversed i
        ii, jj = np.meshgrid(np.arange(0, self.image.shape[0], dtype=np.long), np.arange(0, self.image.shape[1], dtype=np.long), indexing="ij")
        tile_pixels = np.column_stack((ii.flat, jj.flat))

        # Corresponding ground points in world coordinates
        ones = elevation*np.ones((tile_pixels.shape[0], 1), dtype=np.float64)
        world_points = np.copy(np.column_stack((self.image_to_world(tile_pixels), ones)))

        # Project ground points to get pixel coordinates
        image_pixels = pymodel0.model0_projection_array(internal, external, world_points)
        image_pixels = sensor_types.sensor_to_pixel(image_pixels, pixel_size, image.shape[0], image.shape[1])

        # Remove out of bounds pixels
        mask = image_bounds_mask(image_pixels, image.shape)
        image_pixels = image_pixels[mask].reshape((-1, 2))
        tile_pixels = tile_pixels[mask].reshape((-1, 2))

        # Retrive RGB values and store them
        pixel_colors = get_pixel_colors(image, image_pixels)
        self.image[tile_pixels[:,0], tile_pixels[:,1]] = pixel_colors


def project_corners(internal, camera, pixel_size, im_shape, elevation):
    """Project the four corners of an image onto the ground"""
    rows, cols = im_shape[0], im_shape[1]
    points_image = np.array([
        [ pixel_size*cols/2,  pixel_size*rows/2],
        [-pixel_size*cols/2,  pixel_size*rows/2],
        [ pixel_size*cols/2, -pixel_size*rows/2],
        [-pixel_size*cols/2, -pixel_size*rows/2]])
    return np.array([pymodel0.model0_inverse(internal, camera, pix, elevation) for pix in points_image])

# TODO: profile orthoimage for performance
# TODO: check why first two solutions are identical

def main():
    if len(sys.argv) < 2:
        print("Usage: ./orthoimage.py <data_root> <project_dir>")
        sys.exit(-1)

    data_root = sys.argv[1]
    project_dir = sys.argv[2]
    model_filename = os.path.join(project_dir, "project.json")
    solution_max = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    project = Project(model_filename)
    elevation = 0
    internal = np.array(project.model.internal, dtype=np.float64)
    pixel_size = project.model.pixel_size

    left = io.imread(os.path.join(data_root, project.data_set.filenames[0]))
    right = io.imread(os.path.join(data_root, project.data_set.filenames[1]))
    image_shape = left.shape

    # Down project all cameras corners to get the area

    os.makedirs(os.path.abspath(os.path.join(project_dir, "tile")), exist_ok=True)
    tile_dir = os.path.join(project_dir, "tile")
    number_of_solutions = len(project.model.solutions)
    for solution_number in range(number_of_solutions if solution_max is None else solution_max):
        print("{}/{}".format(solution_number+1, number_of_solutions))
        cam_left = np.array(project.model.solutions[solution_number].cameras[0], dtype=np.float64)
        cam_right = np.array(project.model.solutions[solution_number].cameras[1], dtype=np.float64)

        corners_left  = project_corners(internal, cam_left , pixel_size, image_shape, elevation)
        corners_right = project_corners(internal, cam_right, pixel_size, image_shape, elevation)

        world_rect = WorldRect.from_points(np.vstack([corners_left, corners_right]))

        gsd = 0.25
        tile = FlatTile(world_rect, gsd)

        tile.draw_cam_trace(corners_left)
        tile.draw_cam_trace(corners_right)
        tile.project_camera(internal, cam_left, elevation, pixel_size, left)
        tile.project_camera(internal, cam_right, elevation, pixel_size, right)
        tile.draw_observations(internal, cam_left, elevation, project, project.features.edges[0].obs_a)
        tile.draw_observations(internal, cam_right, elevation, project, project.features.edges[0].obs_b)
        tile.draw_obs_pair(internal, cam_left, cam_right, elevation, project, project.features.edges[0].obs_a, project.features.edges[0].obs_b)

        io.imsave(os.path.join(tile_dir, "tile{}.jpg".format(solution_number)), tile.image)

if __name__ == "__main__":
    main()

