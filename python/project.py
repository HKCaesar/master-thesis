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

class ObsPair(object):
    def __init__(self, data):
        self.cam_a = data["cam_a"]
        self.cam_b = data["cam_b"]
        self.obs_a = np.array(data["obs_a"], dtype=np.float64)
        self.obs_b = np.array(data["obs_b"], dtype=np.float64)

class ImageGraph(object):
    def __init__(self, data):
        self.number_of_matches = data["number_of_matches"]
        self.compute_scale = data["compute_scale"]
        self.computed = data["computed"]
        self.edges = [ObsPair(d) for d in data["edges"]]

class DataSet(object):
    def __init__(self, data):
        self.filenames = data["filenames"]
        self.rows = np.array(data["rows"], dtype=np.float64)
        self.cols = np.array(data["cols"], dtype=np.float64)

class Project(object):
    def __init__(self, filename):
        p = json.load(open(filename))
        self.model = Model0(p["model"]["ptr_wrapper"]["data"])
        self.features = ImageGraph(p["features"]["ptr_wrapper"]["data"])
        self.data_set = DataSet(p["data_set"]["ptr_wrapper"]["data"])

