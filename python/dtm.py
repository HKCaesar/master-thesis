#!/usr/bin/env python3

"Creates a DTM in xyzrgb format from a project.json"

import sys
import os.path
import numpy as np
from skimage import io

import sensor_types
from project import Project
from orthoimage import get_pixel_colors

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
    image_right = io.imread(os.path.join(data_root, project.data_set.filenames[1]))

    # For each point in the DTM, project to camera and extract color
    elevation = 0
    sol = project.model.solutions[0]

    cam_left = sol.cameras[0]
    cam_right = sol.cameras[1]

    n = sol.terrain.shape[0]
    points = np.hstack((sol.terrain, elevation*np.ones((n, 1))))

    sens_left = pymodel0.model0_projection_array(project.model.internal, cam_left, points)
    pix_left = sensor_types.sensor_to_pixel(sens_left, project.model.pixel_size, project.model.rows, project.model.cols)

    # Take left image for now
    # More proerly should take a (gamma corrected?) blending of all visible images
    # or the brightest, or something
    colors = get_pixel_colors(image_left, pix_left)
    
    dtm_dir = os.path.abspath(os.path.join(project_dir, "flatdtm"))
    os.makedirs(dtm_dir, exist_ok=True)
    np.savetxt(os.path.join(dtm_dir, "dtm.xyz"), np.hstack((points, colors)), "%0.0f")

if __name__ == "__main__":
    main()

