#!/usr/bin/env python3

import sys
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("build")
try: import pymodel0
except: print("Error importing pymodel0"); sys.exit(-1)

from orthoimage import mm_to_pixels

def compute_residuals(solution, internal, pixel_size, shape, observations):
    cams = np.array(solution["cameras"], dtype=np.float64)
    terrain = np.hstack((np.array(solution["terrain"]), np.zeros((len(solution["terrain"]), 1))))

    # in model0 all points are visible in all cams
    image_pixels = pymodel0.model0_projection_array(internal, cams[0], terrain)
    # subtract to features
    residuals = mm_to_pixels(image_pixels, pixel_size, shape) - observations[:,[0,1]]
    return residuals

def main():
    model_filename = sys.argv[1]
    project = json.load(open(model_filename))

    residuals = compute_residuals(project["model"]["ptr_wrapper"]["data"]["solutions"][-1],
        np.array(project["model"]["ptr_wrapper"]["data"]["internal"]),
        project["model"]["ptr_wrapper"]["data"]["pixel_size"],
        (0,0),
        np.array(project["features"]["ptr_wrapper"]["data"]["observations"]))

    plt.plot(residuals[:,0], residuals[:,1], '+')
    plt.show()

if __name__ == "__main__":
    main()

