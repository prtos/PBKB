__author__ = 'amelie'

from numpy.random import RandomState

from PBKB.inference.BB.inference.search_stats import SearchStatsBuilder
from PBKB.inference.BB.inference._branch_and_bound import branch_and_bound, branch_and_bound_no_length
from PBKB.inference.BB.inference._branch_and_bound import branch_and_bound_multiple_solutions


class BranchAndBound:
    def __init__(self, bound_factory, stats_builder, alphabet, seed, max_time, max_n_iterations):
        self.bound_factory = bound_factory
        self.alphabet = alphabet
        self.max_time = max_time
        self.random_state = RandomState(seed)
        self.stats_builder = stats_builder
        self.search_stats_builder = SearchStatsBuilder(max_n_iterations, max_time)

    def search(self, bound_parameters, y_length):
        seed = self.random_state.randint(0, 2 ** 32)
        node_creator = self.bound_factory.build_node_creator(bound_parameters, y_length)
        y, bound = branch_and_bound(node_creator, y_length, self.alphabet, self.search_stats_builder, seed)
        self.stats_builder.add_search_stats(self.search_stats_builder.build())
        return y, bound

    def search_in_length_range(self, bound_parameters, min_y_length, max_y_length):
        seed = self.random_state.randint(0, 2 ** 32)
        node_creator = self.bound_factory.build_node_creator(bound_parameters, min_y_length, max_y_length)
        y, bound = branch_and_bound_no_length(node_creator, min_y_length, max_y_length, self.alphabet,
                                              self.search_stats_builder, seed)
        self.stats_builder.add_search_stats(self.search_stats_builder.build())
        return y, bound

    def search_multiple_solutions(self, node_creator, y_length, n_predictions):
        seed = self.random_state.randint(0, 2 ** 32)
        solutions, bounds = branch_and_bound_multiple_solutions(node_creator, y_length, n_predictions, self.alphabet,
                                                                self.search_stats_builder, seed)
        self.stats_builder.add_search_stats(self.search_stats_builder.build())
        return solutions, bounds