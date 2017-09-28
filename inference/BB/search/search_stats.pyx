cimport numpy
from cpython cimport bool
from timeit import default_timer
cimport search_stats


cdef class SearchStatsBuilder:
    def __init__(self, max_n_iterations, max_time):
        self.max_n_iterations = max_n_iterations
        self.max_time = max_time
        self.start()

    cdef void start(self):
        self.search_stats = SearchStats(0, 0, 0, 0, 0, False)
        self.start_time = default_timer()

    cdef void end_start_nodes(self):
        self.search_stats.start_nodes_time = default_timer() - self.start_time

    cdef void add_iteration(self):
        self.search_stats.n_iterations += 1

    cdef void update_solution(self):
        self.search_stats.solution_time = default_timer() - self.start_time
        self.search_stats.solution_n_iterations = self.search_stats.n_iterations

    cdef void end(self):
        self.search_stats.is_approximate = self.is_time_or_n_iterations_expired()
        self.search_stats.total_time = default_timer() - self.start_time

    cdef bool is_time_or_n_iterations_expired(self):
        cdef FLOAT64_t elapsed_time = default_timer() - self.start_time
        cdef bool is_expired = False
        if elapsed_time >= self.max_time or self.search_stats.n_iterations >= self.max_n_iterations:
            is_expired = True
        return is_expired

    cpdef dict build(self):
        # I wasn't able to automatically convert struct to dict in Python 2 (but it worked in Python 3)
        cdef dict stats_dict = dict()
        stats_dict['n_iterations'] = self.search_stats.n_iterations
        stats_dict['solution_n_iterations'] = self.search_stats.solution_n_iterations
        stats_dict['total_time'] = self.search_stats.total_time
        stats_dict['solution_time'] = self.search_stats.solution_time
        stats_dict['start_nodes_time'] = self.search_stats.start_nodes_time
        stats_dict['is_approximate'] = self.search_stats.is_approximate
        return stats_dict


cdef class SearchStatsBuilderTest:
    def __init__(self, max_n_iterations, max_time):
        self.search_stats_builder = SearchStatsBuilder(max_n_iterations, max_time)

    def start(self):
        self.search_stats_builder.start()

    def end_start_nodes(self):
        self.search_stats_builder.end_start_nodes()

    def add_iteration(self):
        self.search_stats_builder.add_iteration()

    def update_solution(self):
        self.search_stats_builder.update_solution()

    def end(self):
        self.search_stats_builder.end()

    def is_time_or_n_iterations_expired(self):
        return self.search_stats_builder.is_time_or_n_iterations_expired()

    def build(self):
        return self.search_stats_builder.build()

