cimport numpy
from cpython cimport bool

ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int64_t INT64_t
ctypedef numpy.int8_t INT8_t


cdef struct SearchStats:
    INT64_t n_iterations, solution_n_iterations
    FLOAT64_t total_time, solution_time, start_nodes_time
    INT8_t is_approximate


cdef class SearchStatsBuilder:
    cdef SearchStats search_stats
    cdef FLOAT64_t start_time
    cdef FLOAT64_t max_time, max_n_iterations
    cdef void start(self)
    cdef void end_start_nodes(self)
    cdef void add_iteration(self)
    cdef void update_solution(self)
    cdef void end(self)
    cdef bool is_time_or_n_iterations_expired(self)
    cpdef dict build(self)


#todo put this crap somewhere else or maybe I should just do C++ testing
cdef class SearchStatsBuilderTest:
    cdef SearchStatsBuilder search_stats_builder