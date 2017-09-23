cimport numpy

ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int64_t INT64_t


cdef struct Bound:
    FLOAT64_t bound_value
    FLOAT64_t real_value


cdef class BoundCalculator:
    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length)

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length)

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length)


cdef class MaxBoundCalculator(BoundCalculator):
    cdef:
        FLOAT64_t[:,::1] graph, graph_weights, end_weights
        int max_n, min_length
        dict n_gram_to_index


cdef class OCRMinBoundCalculator(BoundCalculator):
    cdef:
        FLOAT64_t[::1] position_weights, start_node_real_values, y_y_bounds
        int min_n, max_n, max_y_length, alphabet_length
        list n_grams

    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y)
    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_y_length)
    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_length)
    cdef FLOAT64_t _get_n_gram_y_y_prime_bound(self, str y, int y_length, int n, dict n_gram_to_index)
    #cdef list _get_positions(self, str y, int n, dict n_gram_to_index, int unknown_part_length)
    cdef FLOAT64_t[::1] _get_start_node_y_y_prime_bounds(self, final_length)


cdef class PeptideMinBoundCalculator(BoundCalculator):
    cdef:
        int n
        int alphabet_length
        dict letter_to_index
        FLOAT64_t[:,::1] similarity_matrix
        FLOAT64_t[:,::1] position_matrix
        FLOAT64_t[::1] start_node_bound_values
        FLOAT64_t[::1] start_node_real_values
        FLOAT64_t[::1] y_y_bounds

    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_length)

    cdef FLOAT64_t[::1] precompute_start_node_bounds(self, int final_length, list n_grams)

    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y)

    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_start_index)

    cdef FLOAT64_t compute_n_gram_y_y_prime_bound(self, int n_gram_length, int n_gram_index, str y, int y_start_index,
                                                  numpy.ndarray[FLOAT64_t, ndim=2] similarity_matrix)

    cdef numpy.ndarray[FLOAT64_t, ndim=1] transform_letter_scores_in_n_gram_scores(self,
                                                                                   numpy.ndarray[FLOAT64_t, ndim=1]
                                                                                   letter_scores, int n_gram_length,
                                                                                   int index_in_n_gram)