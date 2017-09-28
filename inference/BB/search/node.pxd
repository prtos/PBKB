import numpy
cimport numpy


ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int64_t INT64_t


cdef class Node:
    cdef public str y
    cdef public FLOAT64_t real_min_bound, real_max_bound, bound, bound_minimize
    cdef FLOAT64_t get_bound(self)
    cdef void invert_bound(self)