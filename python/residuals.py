#!/usr/bin/env python3

import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt

import sensor_types
from project import Project

sys.path.append("build")
try: import pymodel0
except: print("Error importing pymodel0"); sys.exit(-1)

def compute_residuals(project, solution_number):
    terrain = np.hstack((project.model.solutions[solution_number].terrain, np.zeros((len(project.model.solutions[0].terrain), 1))))

    # in model0 all points are visible in all cams
    image_pixels = pymodel0.model0_projection_array(project.model.internal, project.model.solutions[solution_number].cameras[0], terrain)
    # subtract to features
    residuals = sensor_types.sensor_to_pixel(image_pixels, project.model.pixel_size, project.model.rows, project.model.cols) - project.features.observations[:,[0,1]]
    return residuals

def main():
    model_filename = sys.argv[1]
    project = Project(model_filename)

    residuals = compute_residuals(project, 0)
    plt.plot(residuals[:,0], residuals[:,1], '+')

    residuals = compute_residuals(project, -1)
    plt.plot(residuals[:,0], residuals[:,1], 'x')

    plt.show()

if __name__ == "__main__":
    main()

