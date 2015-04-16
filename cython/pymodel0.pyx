import numpy as np
cimport numpy as np

cdef extern from "../src/camera_models.h":
    void vector_print(double* vec)

def hello(np.ndarray[np.float64_t, ndim=1] npvec):
    cdef double* vec = [npvec[0], npvec[1], npvec[2]]
    vector_print(vec)

