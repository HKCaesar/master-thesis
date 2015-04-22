#!/usr/bin/env python3

import sys
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt

from orthoimage import mm_to_pixels

sys.path.append("build")
try: import pymodel0
except: print("Error importing pymodel0"); sys.exit(-1)

class Solution(object):
    def __init__(self, data):
        self.cameras = np.array(data["cameras"], dtype=np.float64)
        self.terrain = np.array(data["terrain"], dtype=np.float64)

class Model0(object):
    def __init__(self, data):
        self.internal = np.array(data["internal"], dtype=np.float64)
        self.pixel_size = np.array(data["pixel_size"], dtype=np.float64)
        self.solutions = [Solution(sol) for sol in data["solutions"]]

class ImageFeatures(object):
    def __init__(self, data):
        self.observations = np.array(data["observations"], dtype=np.float64)

class DataSet(object):
    def __init__(self, data):
        self.filenames = data["ptr_wrapper"]["data"]["filenames"]

class Project(object):
    def __init__(self, filename):
        p = json.load(open(filename))
        self.model = Model0(p["model"]["ptr_wrapper"]["data"])
        self.features = ImageFeatures(p["features"]["ptr_wrapper"]["data"])
        # self.data_set = DataSet(p["data_set"]["ptr_wrapper"]["data"])

def compute_residuals(project, solution_number):
    terrain = np.hstack((project.model.solutions[0].terrain, np.zeros((len(project.model.solutions[0].terrain), 1))))

    # in model0 all points are visible in all cams
    image_pixels = pymodel0.model0_projection_array(project.model.internal, project.model.solutions[solution_number].cameras[0], terrain)
    # subtract to features
    residuals = mm_to_pixels(image_pixels, project.model.pixel_size, (0,0)) - project.features.observations[:,[0,1]]
    return residuals

def main():
    model_filename = sys.argv[1]
    project = Project(model_filename)

    residuals = compute_residuals(project, -1)

    plt.plot(residuals[:,0], residuals[:,1], '+')
    plt.show()

if __name__ == "__main__":
    main()

