__author__ = 'amelie'

from sklearn.base import BaseEstimator
import numpy

from PBKB.inference.BB.search.graph_builder import GraphBuilder
from PBKB.inference.BB.search.branch_and_bound import BranchAndBound
from PBKB.inference.BB.search.bound_factory import get_gs_similarity_node_creator
from PBKB.inference.BB.features.gs_similarity_feature_space import GenericStringSimilarityFeatureSpace
from PBKB.inference.BB.search.stats_builder import LearnerStatsBuilder


class Preimage(BaseEstimator):
    def __init__(self, alphabet, n, gs_kernel, max_time=30, max_n_iterations=numpy.inf, seed=42):
        self.n = int(n)
        self.alphabet = [str(letter) for letter in alphabet]
        self.gs_kernel = gs_kernel
        self.max_time = max_time
        self.max_n_iterations = max_n_iterations
        self.seed = seed
        self._is_normalized = True
        self._graph_builder = None
        self._node_creator_ = None
        self._y_length_ = None
        self._stats_builder = None

    def fit(self, X, learned_weights, y_length):
        self._graph_builder = GraphBuilder(self.alphabet, 1, self.n)
        self._stats_builder = LearnerStatsBuilder()
        self._stats_builder.start_fit(X.shape[0])
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, self.n, X, self._is_normalized,
                                                            self.gs_kernel)
        gs_weights = feature_space.compute_weights(learned_weights, y_length)
        graph = self._graph_builder.build_graph(gs_weights, y_length)
        self._node_creator_ = get_gs_similarity_node_creator(self.alphabet, self.n, graph, gs_weights, y_length,
                                                             self.gs_kernel)
        self._y_length_ = y_length
        self._branch_and_bound_ = BranchAndBound(None, self._stats_builder, self.alphabet,
                                                 self.seed, self.max_time, self.max_n_iterations)
        self._stats_builder.end_fit()

    def predict(self, n_predictions):
        self._stats_builder.start_predict(1)
        strings, bounds = self._branch_and_bound_.search_multiple_solutions(self._node_creator_, self._y_length_, n_predictions)
        self._stats_builder.end_predict()
        return strings, bounds

    def get_stats(self):
        return self._stats_builder.build()