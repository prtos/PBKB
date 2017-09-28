from libc.math cimport sqrt
cimport node_creator


cdef class NodeCreator:
    def __init__(self, min_bound_calculator, max_bound_calculator, n_grams, is_normalized):
        self.min_bound_calculator = min_bound_calculator
        self.max_bound_calculator = max_bound_calculator
        self.n_grams = n_grams
        self.is_normalized = is_normalized

    # For unit tests only
    def create_node_python(self, str  y, Node parent_node, int final_length):
        return self.create_node(y, parent_node, final_length)

    cdef Node create_node(self, str  y, Node parent_node, int final_length):
        cdef Bound max_bound = self.max_bound_calculator.compute_bound(y, parent_node.real_max_bound, final_length)
        cdef Bound min_bound = self.min_bound_calculator.compute_bound(y, parent_node.real_min_bound, final_length)
        cdef FLOAT64_t bound_value
        if self.is_normalized:
            bound_value = max_bound.bound_value / sqrt(min_bound.bound_value)
        else:
            bound_value = min_bound.bound_value - 2 * max_bound.bound_value
        return Node(y, bound_value, min_bound.real_value, max_bound.real_value, self.is_normalized)

    # For unit tests only
    def get_start_nodes_python(self, int final_length):
        return self.get_start_nodes(final_length)

    cdef list get_start_nodes(self, int final_length):
        cdef list start_nodes
        cdef FLOAT64_t[::1] max_values = self.max_bound_calculator.get_start_node_real_values(final_length)
        cdef FLOAT64_t[::1] max_bounds = self.max_bound_calculator.get_start_node_bounds(final_length)
        cdef FLOAT64_t[::1] min_values = self.min_bound_calculator.get_start_node_real_values(final_length)
        cdef FLOAT64_t[::1] min_bounds = self.min_bound_calculator.get_start_node_bounds(final_length)
        if self.is_normalized:
            start_nodes = self._get_start_nodes_normalized(max_values, max_bounds, min_values, min_bounds, final_length)
        else:
            start_nodes = self._get_start_nodes_not_normalized(max_values, max_bounds, min_values, min_bounds,
                                                               final_length)
        return start_nodes

    cdef list _get_start_nodes_normalized(self, FLOAT64_t[::1] max_values, FLOAT64_t[::1] max_bounds,
                                              FLOAT64_t[::1] min_values, FLOAT64_t[::1] min_bounds, int final_length):
        cdef list start_nodes = []
        for i in range(len(self.n_grams)):
            start_nodes.append(Node(str(self.n_grams[i]), max_bounds[i] / sqrt(min_bounds[i]), min_values[i],
                                    max_values[i], self.is_normalized))
        return start_nodes

    cdef list _get_start_nodes_not_normalized(self, FLOAT64_t[::1] max_values, FLOAT64_t[::1] max_bounds,
                                              FLOAT64_t[::1] min_values, FLOAT64_t[::1] min_bounds, int final_length):
        cdef list start_nodes = []
        for i in range(len(self.n_grams)):
            start_nodes.append(Node(str(self.n_grams[i]), min_bounds[i] - 2 * max_bounds[i], min_values[i],
                                    max_values[i], self.is_normalized))
        return start_nodes
