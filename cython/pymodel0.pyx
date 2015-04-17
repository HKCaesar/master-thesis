import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "../src/camera_models.h":
    bool model0_projection_double(const double* internal, const double* external, const double* point, double* residuals)

cdef assert_size(np.ndarray[double, ndim=1] arr, long expected_size):
    assert arr.size == expected_size, "Expected array of size {}, got {}".format(expected_size, arr.size)

def model0_projection(np.ndarray[double, ndim=1] internal, np.ndarray[double, ndim=1] external, np.ndarray[double, ndim=1] point):
    assert_size(internal, 3)
    assert_size(external, 6)
    assert_size(point, 2)
    cdef double residuals[2]
    model0_projection_double(&internal[0], &external[0], &point[0], residuals)
    return np.array([residuals[0], residuals[1]])

