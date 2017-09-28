"""Factory initializing bounds and node creator for the branch and bound search."""

__author__ = 'amelie'

import numpy as np
from PBKB.inference.BB.search.bound_calculator import OCRMinBoundCalculator, MaxBoundCalculator, PeptideMinBoundCalculator
from PBKB.inference.BB.search.node_creator import NodeCreator

from PBKB.inference.BB.utils import get_n_gram_to_index, get_n_grams, get_index_to_n_gram


class BoundParameters:
    def __init__(self, graph, graph_weights, end_weights):
        self.graph = graph
        self.graph_weights = graph_weights
        self.end_weights = end_weights


class BoundFactory:
    def __init__(self, alphabet, min_n, max_n, is_normalized, kernel, max_y_length):
        self.min_n = min_n
        self.max_n = max_n
        self.is_normalized = is_normalized
        self.alphabet = alphabet
        self.kernel = kernel
        self.max_y_length = max_y_length
        self._min_bound_precomputed = None
        self._setup_n_grams(kernel)
        self._setup_min_bound(max_y_length)

    def _setup_n_grams(self, kernel):
        self.n_grams = list(get_n_grams(self.alphabet, self.max_n))
        self.n_gram_to_index = get_n_gram_to_index(self.alphabet, self.max_n)
        self.index_to_n_gram = get_index_to_n_gram(self.alphabet, self.max_n)
        self.n_gram_real_values = kernel.element_wise_kernel(np.array(self.n_grams))

    def _setup_min_bound(self, max_y_length):
        self._build_min_bound(max_y_length)

    def build_node_creator(self, bound_parameters, min_y_length, max_y_length=None):
        """Create the bounds and the node creator for the branch and bound search of the n-gram kernel

        Parameters
        ----------
        bound_parameters : BoundParameters

        Returns
        -------
        node_creator : NodeCreator
            Node creator for the branch and bound search instantiated with the n-gram bounds
        """
        max_y_length = min_y_length if max_y_length is None else max_y_length
        min_bound = self._build_min_bound(max_y_length)
        max_bound = self._build_max_bound(bound_parameters, min_y_length)
        node_creator = NodeCreator(min_bound, max_bound, self.n_grams, bool(self.is_normalized))
        return node_creator

    def _build_min_bound(self, max_y_length):
        if max_y_length > self.max_y_length or self._min_bound_precomputed is None:
            self.max_y_length = max_y_length
            position_weights = self.kernel.compute_position_weights(0, max_y_length)
            self._min_bound_precomputed = OCRMinBoundCalculator(self.min_n, self.max_n, position_weights, self.n_grams,
                                                                max_y_length, len(self.alphabet),
                                                                self.n_gram_real_values)
        return self._min_bound_precomputed

    def _build_max_bound(self, bound_parameters, min_y_length):
        if bound_parameters.graph_weights.ndim == 1:
            bound_parameters.graph_weights = bound_parameters.graph_weights.reshape(1, -1)
        if bound_parameters.end_weights.ndim == 1:
            bound_parameters.end_weights = bound_parameters.end_weights.reshape(1, -1)
        max_bound = MaxBoundCalculator(self.max_n, bound_parameters.graph, bound_parameters.graph_weights,
                                       bound_parameters.end_weights, min_y_length, self.n_gram_to_index)
        return max_bound


def get_gs_similarity_node_creator(alphabet, n, graph, graph_weights, y_length, gs_kernel):
    """Create the bounds and the node creator for the branch and bound search of the generic string kernel.

    Takes in account the position and the n-gram penalties when comparing strings (sigma_p and sigma_c in the
    gs kernel).

    Parameters
    ----------
    alphabet : list
        List of letters.
    n : int
        N-gram length.
    graph : array, shape = [n_partitions, len(alphabet)**n]
        Array representation of the graph. graph[i, j] represents the maximum value of a string of length i + n ending
        with the jth n-gram.
    graph_weights : array, shape = [n_partitions, len(alphabet)**n]
        Weight of each n-gram.
    y_length : int
        Length of the string to predict.
    gs_kernel : GenericStringKernel
        Generic String Kernel with position and n-gram penalties.

    Returns
    -------
    node_creator : NodeCreator
        Node creator for the branch and bound search instantiated with the generic string bounds
    """
    n_gram_to_index = get_n_gram_to_index(alphabet, n)
    letter_to_index = get_n_gram_to_index(alphabet, 1)
    n_grams = get_n_grams(alphabet, n)
    min_bound = PeptideMinBoundCalculator(n, len(alphabet), n_grams, letter_to_index, y_length, gs_kernel)
    end_weights = np.zeros(graph_weights.shape)
    max_bound = MaxBoundCalculator(n, graph, graph_weights, end_weights, y_length, n_gram_to_index)
    node_creator = NodeCreator(min_bound, max_bound, n_grams, True)
    return node_creator