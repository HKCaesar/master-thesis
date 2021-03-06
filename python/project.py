import json
import numpy as np

def pixel_size(internal):
    return internal[3]

class DataSet(object):
    def __init__(self, data, ptrmap=None):
        self.filenames = data["filenames"]
        self.rows = np.array(data["rows"], dtype=np.float64)
        self.cols = np.array(data["cols"], dtype=np.float64)

class ObsPair(object):
    def __init__(self, data, ptrmap=None):
        self.cam_a = data["cam_a"]
        self.cam_b = data["cam_b"]
        self.obs_a = np.array(data["obs_a"], dtype=np.float64)
        self.obs_b = np.array(data["obs_b"], dtype=np.float64)

class ImageGraph(object):
    def __init__(self, data, ptrmap=None):
        self.number_of_matches = data["number_of_matches"]
        self.compute_scale = data["compute_scale"]
        self.computed = data["computed"]
        self.edges = [ObsPair(d) for d in data["edges"]]

class Model0Solution(object):
    def __init__(self, data, ptrmap=None):
        self.cameras = np.array(data["cameras"], dtype=np.float64)
        self.terrain = np.array(data["terrain"], dtype=np.float64)

class Model0(object):
    def __init__(self, data, ptrmap=None):
        if ptrmap is not None:
            self.features = ptrmap.load(ImageGraph, data["base"]["features"])
        self.internal = np.array(data["internal"], dtype=np.float64)
        self.solutions = [Model0Solution(sol) for sol in data["solutions"]]

    def fexternal(self, solution_number):
        return self.solutions[solution_number].cameras

    def finternal(self, solution_number):
        return self.internal

    def fterrain(self, solution_number):
        terrain = self.solutions[solution_number].terrain
        n = terrain.shape[0]
        return np.hstack((terrain, np.zeros((n, 1))))

class ModelTerrainSolution(object):
    def __init__(self, data, ptrmap=None):
        self.terrain = np.array(data["terrain"], dtype=np.float64)

class ModelTerrain(object):
    def __init__(self, data, ptrmap=None):
        if ptrmap is not None:
            self.features = ptrmap.load(ImageGraph, data["base"]["features"])
        self.internal = np.array(data["internal"], dtype=np.float64)
        self.cameras = np.array(data["cameras"], dtype=np.float64)
        self.solutions = [ModelTerrainSolution(sol) for sol in data["solutions"]]

    # Interfaces
    def fexternal(self, solution_number):
        return self.cameras

    def finternal(self, solution_number):
        return self.internal

    def fterrain(self, solution_number):
        return self.solutions[solution_number].terrain

polymorphic_models = {
    "Model0": Model0,
    "ModelTerrain": ModelTerrain
}

class PtrMap(object):
    intmax = 2147483648
    def __init__(self):
        self.ptrmap = {}

    def load(self, Type, obj):
        ptr_wrapper = obj["ptr_wrapper"]
        ptr_id = ptr_wrapper["id"]
        if ptr_id > self.intmax:
            obj = Type(ptr_wrapper["data"], self)
            self.ptrmap[ptr_id - self.intmax] = obj
            return obj
        else:
            return self.ptrmap[ptr_id]

    def load_polymorphic(self, obj, type_map):
        ptr_wrapper = obj["ptr_wrapper"]
        ptr_id = ptr_wrapper["id"]
        if ptr_id > self.intmax:
            Type = type_map[obj["polymorphic_name"]]
            obj = Type(ptr_wrapper["data"], self)
            self.ptrmap[ptr_id - self.intmax] = obj
            return obj
        else:
            return self.ptrmap[ptr_id]

class Bootstrap(object):
    def __init__(self, data, ptrmap=None):
        if ptrmap is not None:
            self.base_model = ptrmap.load_polymorphic(data["base_model"], polymorphic_models)
        self.number_of_samples = data["number_of_samples"]
        self.size_of_samples = data["size_of_samples"]
        self.internals = np.array(data["internals"], dtype=np.float64)
        self.externals = np.array(data["externals"], dtype=np.float64)

    def extract_cameras(self):
        "Generator for cameras externals"
        number_of_samples, number_of_cameras, six = self.externals.shape
        assert number_of_samples == self.number_of_samples
        assert six == 6
        for cam_number in range(number_of_cameras):
            yield self.externals[:, cam_number, :]

class Project(object):
    def __init__(self, filename):
        ptrmap = PtrMap()
        p = json.load(open(filename))
        self.data_set = ptrmap.load(DataSet, p["data_set"])
        self.features = [ptrmap.load(ImageGraph, ig) for ig in p["features_list"]]
        self.models = [ptrmap.load_polymorphic(m, polymorphic_models) for m in p["models"]]
        self.bootstraps = [ptrmap.load(Bootstrap, boot) for boot in p["bootstraps"]]
