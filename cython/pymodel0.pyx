import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "../src/camera_models.h":
    bool model0_projection_double(const double* internal, const double* external, const double* point, double* residuals)

# One dimensional array
#ctypedef np.ndarray[np.float64_t, ndim=1] array

def model0_projection(np.ndarray[double, ndim=1] internal, np.ndarray[double, ndim=1] external, np.ndarray[double, ndim=1] point):
    cdef double residuals[2]
    model0_projection_double(&internal[0], &external[0], &point[0], residuals)
    #"model0_projection<double>"(&internal[0], &external[0], &point[0], residuals)
    return np.array([residuals[0], residuals[1]])

