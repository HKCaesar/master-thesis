import json
import numpy as np

class Solution(object):
    def __init__(self, data):
        self.cameras = np.array(data["cameras"], dtype=np.float64)
        self.terrain = np.array(data["terrain"], dtype=np.float64)

class Model0(object):
    def __init__(self, data):
        self.internal = np.array(data["internal"], dtype=np.float64)
        self.pixel_size = np.array(data["pixel_size"], dtype=np.float64)
        self.solutions = [Solution(sol) for sol in data["solutions"]]
        self.rows = np.array(data["rows"], dtype=np.float64)
        self.cols = np.array(data["cols"], dtype=np.float64)

class ImageFeatures(object):
    def __init__(self, data):
        self.observations = np.array(data["observations"], dtype=np.float64)

class DataSet(object):
    def __init__(self, data):
        self.filenames = data["filenames"]

class Project(object):
    def __init__(self, filename):
        p = json.load(open(filename))
        self.model = Model0(p["model"]["ptr_wrapper"]["data"])
        self.features = ImageFeatures(p["features"]["ptr_wrapper"]["data"])
        self.data_set = DataSet(p["data_set"]["ptr_wrapper"]["data"])

