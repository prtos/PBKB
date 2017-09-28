cimport numpy
from cpython cimport bool

from node cimport Node
from bound_calculator cimport BoundCalculator, Bound, FLOAT64_t


cdef class NodeCreator:
    cdef:
        BoundCalculator min_bound_calculator
        BoundCalculator max_bound_calculator
        list n_grams
        bool is_normalized

    cdef Node create_node(self, str  y, Node parent_node, int final_length)

    cdef list get_start_nodes(self, int final_length)

    cdef list _get_start_nodes_not_normalized(self, FLOAT64_t[::1] max_values, FLOAT64_t[::1] max_bounds,
                                              FLOAT64_t[::1] min_values, FLOAT64_t[::1] min_bounds, int final_length)

    cdef list _get_start_nodes_normalized(self, FLOAT64_t[::1] max_values, FLOAT64_t[::1] max_bounds,
                                          FLOAT64_t[::1] min_values, FLOAT64_t[::1] min_bounds, int final_length)