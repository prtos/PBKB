
cimport node


cdef class Node:
    def __init__(self, y, bound, real_min_bound, real_max_bound, is_maximize):
        self.y = y
        self.bound = bound
        self.real_min_bound = real_min_bound
        self.real_max_bound = real_max_bound
        self.bound_minimize = -self.bound if is_maximize else self.bound

    cdef FLOAT64_t get_bound(self):
        return self.bound

    def __str__(self):
        node_string = "y: {}, bound: {}, real_min_bound: {}, real_max_bound: {}"
        return node_string.format(self.y, self.bound, self.real_min_bound, self.real_max_bound)

    def __richcmp__(self, Node other_node, int op):
        if op == 0:
            if self.bound_minimize == other_node.bound_minimize:
                return len(self.y) > len(other_node.y)
            else:
                return self.bound_minimize < other_node.bound_minimize
        if op == 2:
            return self.y == other_node.y and self.bound == other_node.bound \
                   and self.real_min_bound == other_node.real_min_bound \
                   and self.real_max_bound == other_node.real_max_bound \
                   and self.bound_minimize == other_node.bound_minimize

    cdef void invert_bound(self):
        self.bound_minimize *= -1