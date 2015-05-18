#!/usr/bin/env python3

"Creates a DTM in xyzrgb format from a project.json"

import sys
import os.path
import numpy as np
from skimage import io

import sensor_types
from project import Project, pixel_size
from orthoimage import get_pixel_colors, image_bounds_mask

sys.path.append("build")
try: import pymodel0
except: print("Error importing pymodel0"); sys.exit(-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: ./dtm.py <data_root> <project_dir>")
        sys.exit(-1)

    # Parse arguments
    data_root = sys.argv[1]
    project_dir = sys.argv[2]
    model_filename = os.path.join(project_dir, "project.json")

    # Load project and images
    project = Project(model_filename)
    image_left = io.imread(os.path.join(data_root, project.data_set.filenames[0]))
    # image_right = io.imread(os.path.join(data_root, project.data_set.filenames[1]))
    data_set = project.data_set

    # For each model
    for (model_number, model) in enumerate(project.models):
        solution_number = -1
        # TODO final interface shouldn't have sol_number parameter
        # there's only 1 final
        # what about orthoimage?
        cam_left = model.fexternal(solution_number)[0]
        # cam_right = model.fexternal(solution_number)[1]
        points = model.fterrain(solution_number)

        # For each point in the DTM, project to camera and extract color
        sens_left = pymodel0.model0_projection_array(model.finternal(solution_number), cam_left, points)
        pix_left = sensor_types.sensor_to_pixel(sens_left, pixel_size(model.finternal(solution_number)), data_set.rows, data_set.cols)

        # Filter out of image bounds points
        size_before_filter = pix_left.shape[0]
        mask = image_bounds_mask(pix_left, (data_set.rows, data_set.cols))
        pix_left = pix_left[mask[:,0], :]
        points = points[mask[:,0], :]
        filtered_points = size_before_filter - pix_left.shape[0]

        dtm_dir = os.path.abspath(os.path.join(project_dir, "dtm{}-{}".format(model_number, type(model).__name__)))

        if filtered_points > 0:
            print("Info: {} out of bounds pixels were filtered from {}".format(filtered_points, dtm_dir))

        # Take left image for now
        # More properly should take a (gamma corrected?) blending of all visible images
        # or the brightest, or something
        colors = get_pixel_colors(image_left, pix_left)

        os.makedirs(dtm_dir, exist_ok=True)
        np.savetxt(os.path.join(dtm_dir, "dtm.xyz"), np.hstack((points, colors)), "%0.0f")

if __name__ == "__main__":
    main()
