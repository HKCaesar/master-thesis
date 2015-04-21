import numpy as np
cimport numpy as np
from libcpp cimport bool
import cython

cdef extern from "../src/camera_models.h":
    bool model0_projection_double(const double* internal, const double* external, const double* point, double* residuals)
    void model0_image_to_world_double(const double* const internal, const double* const external, const double* pix, const double* elevation, double* dx, double* dy)

cdef assert_size(np.ndarray[double, ndim=1] arr, long expected_size):
    assert arr.size == expected_size, "Expected array of size {}, got {}".format(expected_size, arr.size)

@cython.boundscheck(False)
@cython.wraparound(False)
def model0_projection(np.ndarray[double, ndim=1] internal, np.ndarray[double, ndim=1] external, np.ndarray[double, ndim=1] point):
    assert_size(internal, 3)
    assert_size(external, 6)
    assert_size(point, 2)
    cdef double residuals[2]
    model0_projection_double(&internal[0], &external[0], &point[0], residuals)
    return np.array([residuals[0], residuals[1]])

@cython.boundscheck(False)
@cython.wraparound(False)
def model0_projection_array(np.ndarray[double, ndim=1] internal, np.ndarray[double, ndim=1] external, np.ndarray[double, ndim=2] point):
    assert_size(internal, 3)
    assert_size(external, 6)

    cdef np.ndarray[double, ndim=2] result = np.zeros((point.shape[0], 2), dtype=np.float64)
    for i in range(point.shape[0]):
        model0_projection_double(&internal[0], &external[0], &point[i, 0], &result[i, 0])
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def model0_inverse(np.ndarray[double, ndim=1] internal, np.ndarray[double, ndim=1] external, np.ndarray[double, ndim=1] pix, double elevation):
    assert_size(internal, 3)
    assert_size(external, 6)
    assert_size(pix, 2)

    cdef double dx
    cdef double dy
    model0_image_to_world_double(&internal[0], &external[0], &pix[0], &elevation, &dx, &dy)
    return np.array([dx, dy], dtype=np.float64)

