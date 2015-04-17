#!/usr/bin/env python3

import sys
import os.path
import json
import numpy as np
from skimage import io, draw
from itertools import chain, combinations

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

    def draw_world_point(self, point, color):
        i, j = self.world_to_image(point)
        self.image[i, j] = color

    def draw_cam_trace(self, corners):
        # Draw a line between each corner pair
        for c1, c2 in combinations(corners, 2):
            a = self.world_to_image(c1)
            b = self.world_to_image(c2)
            rr, cc = draw.line(a[0], a[1], b[0], b[1])
            mask = np.logical_and.reduce((
                rr > 0,
                rr < self.image.shape[0],
                cc > 0,
                cc < self.image.shape[1]))
            self.image[rr[mask], cc[mask], :] = 255

    def project_one_camera():
        pass

def project_corners(internal, camera, pixel_size, im_shape, elevation):
    """Project the four corners of an image onto the ground"""
    rows, cols = im_shape[0], im_shape[1]
    points_image = np.array([
        [ pixel_size*rows/2,  pixel_size*cols/2],
        [ pixel_size*rows/2, -pixel_size*cols/2],
        [-pixel_size*rows/2,  pixel_size*cols/2],
        [-pixel_size*rows/2, -pixel_size*cols/2]])
    return np.array([pymodel0.model0_inverse(internal, camera, pix, elevation) for pix in points_image])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./orthoimage.py <data_root> <model.json>")
        sys.exit(-1)

    data_root = sys.argv[1]
    model_filename = sys.argv[2]

    project = json.load(open(model_filename))
    elevation = 0
    internal = np.array(project["model"]["internal"], dtype=np.float64)
    pixel_size = project["model"]["pixel_size"]

    left = io.imread(os.path.join(data_root, project["data"]["left"]))
    right = io.imread(os.path.join(data_root, project["data"]["right"]))
    image_shape = left.shape

    # Down project all cameras corners to get the area
    world_point = np.zeros((len(project["model"]["cameras"])*4, 2))
    cam_left = project["model"]["cameras"][0]
    cam_right = project["model"]["cameras"][1]

    corners_left  = project_corners(internal, np.array(cam_left , dtype=np.float64), pixel_size, image_shape, elevation)
    corners_right = project_corners(internal, np.array(cam_right, dtype=np.float64), pixel_size, image_shape, elevation)

    world_rect = WorldRect.from_points(np.vstack([corners_left, corners_right]))

    gsd = 0.2
    tile = FlatTile(world_rect, gsd)

    tile.draw_cam_trace(corners_left)
    tile.draw_cam_trace(corners_right)

    io.imsave("tile.jpg", tile.image)

"""
load project
project cameras corners to get the bounds
create a big image covering the entire region
for each camera project it down using direct model
option: for each keypoint down project it using inverse model and draw matches lines
"""

def tile_pixel_grid(tilesize):
    """All pixel coordinates of the tile image"""

    # 'ij' indexing and reversed i
    ii, jj = np.meshgrid(np.arange(0, tilesize, dtype=int), np.arange(0, tilesize, dtype=int), indexing="ij")
    return np.column_stack((ii.flat[::-1], jj.flat))

def xyz_grid(pixel_grid, tile_x, tile_y, zoom, tilesize):
    """
    From a list of pixel points, makes the list of XYZ ground points
    """

    gm = GlobalMercator(tileSize=tilesize)
    minx, miny, maxx, maxy = gm.TileBounds(tile_x, tile_y, zoom)

    # Meter coordinates are tile origin + pix*gsd, accounting for i axis flip
    gsd = zoom_to_gsd(zoom)
    points = np.empty_like(pixel_grid, dtype=np.float64)
    points[:,0] = minx + pixel_grid[:,1]*gsd
    points[:,1] = miny + (tilesize-pixel_grid[:,0])*gsd

    return points

class Tile(object):
    def __init__(self, tx, ty, zoom, tilesize):
        self.x, self.y = tx, ty
        self.zoom = zoom
        self.tilesize = tilesize
        self.image = np.zeros((tilesize, tilesize, 3), dtype='uint8')

    def project_one_camera(self, internal, external, elevation, zone, im, input_scale):
        """
        Orthorectify one image onto the tile
        The procedure works with 2D arrays of shape (n, 2),
        where each represents the same 'point' in different frames.
        """

        # All pixels in the tile image
        tile_pixels = tile_pixel_grid(self.tilesize)

        # Corresponding ground points in ESPE3857
        xyz_points = xyz_grid(tile_pixels, self.x, self.y, self.zoom, self.tilesize)

        # Corresponding UTM points
        pjt = ProjTransform(zone)
        utm_points = pjt.xyz_to_utm(xyz_points)

        # Project ground points
        ground_points = np.vstack((utm_points.transpose(), elevation*np.ones((1, utm_points.shape[0]))))
        image_pixels = camera_project(internal, external, ground_points).transpose()

        # Remove out of bounds pixels
        mask = image_bounds_mask(image_pixels, im.shape, input_scale)
        image_pixels = image_pixels[mask].reshape((-1, 2))
        tile_pixels = tile_pixels[mask].reshape((-1, 2))

        # Retrive RGB values and store them
        pixel_colors = get_pixel_colors(im, image_pixels, input_scale)
        self.image[tile_pixels[:,0], tile_pixels[:,1]] = pixel_colors

def get_pixel_colors(image, pixel_points, input_scale):
    """
    Read image value at float coordinate using nearest neighbour interpolation
    """
    indexes = np.asarray(np.round(pixel_points / float(input_scale)), dtype=int)
    return image[indexes[:,0], indexes[:,1]]

def image_bounds_mask(pixel_points, image_shape, input_scale):
    """
    Returns a numpy mask (array of Bool) of size pixel_points.shape
    where each columns is True iff the pixel lies wihtin image bounds
    """

    contained_mask = np.empty(pixel_points.shape, dtype=bool)
    contained_mask[:,:] = np.all((
            # 0.5 because nearest neighbour interpolation rounds to closest integer
            pixel_points[:,0] >= 0,
            pixel_points[:,0] / input_scale < (image_shape[0] - 0.5),
            pixel_points[:,1] >= 0,
            pixel_points[:,1] / input_scale < (image_shape[1] - 0.5)
            ), axis=0)[:,np.newaxis]
    return contained_mask
