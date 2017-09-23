__author__ = 'amelie'

from timeit import default_timer


# todo change this stupid name that looks too much like the others. Maybe this code should go in searchStatsBuilder instead
class SearchStatsMerger:
    def __init__(self):
        self.n_examples = 0
        self.n_iterations = 0
        self.solution_n_iterations = 0
        self.total_time = 0
        self.solution_time = 0
        self.start_nodes_time = 0
        self.n_approximates = 0

    def add_stats(self, n_iterations, solution_n_iterations, total_time, solution_time, start_nodes_time,
                  is_approximate):
        self.n_examples += 1
        self.n_iterations += n_iterations
        self.solution_n_iterations += solution_n_iterations
        self.total_time += total_time
        self.solution_time += solution_time
        self.start_nodes_time += start_nodes_time
        self.n_approximates += is_approximate

    def build(self):
        search_stats = {}
        if self.n_examples > 0:
            search_stats['search_mean_time'] = float(self.total_time) / self.n_examples
            search_stats['search_mean_solution_time'] = float(self.solution_time) / self.n_examples
            search_stats['search_mean_start_nodes_time'] = float(self.start_nodes_time) / self.n_examples
            search_stats['search_mean_n_iterations'] = float(self.n_iterations) / self.n_examples
            search_stats['search_mean_solution_n_iterations'] = float(self.solution_n_iterations) / self.n_examples
            search_stats['search_percentage_of_approximates'] = float(self.n_approximates) / self.n_examples
            search_stats['search_n_examples'] = self.n_examples
        return search_stats


class LearnerStatsBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reset_train()
        self.reset_predict()

    def reset_train(self):
        self._start_time = 0
        self._n_examples_train = 0
        self._train_time = 0

    def reset_predict(self):
        self._start_time = 0
        self._n_examples_test = 0
        self._test_time = 0
        self._search_stats_merger = SearchStatsMerger()

    def start_fit(self, n_examples_train):
        self.reset()
        self._start_time = default_timer()
        self._n_examples_train = n_examples_train

    def end_fit(self):
        self._train_time = default_timer() - self._start_time

    def start_predict(self, n_examples_test):
        self.reset_predict()
        self._start_time = default_timer()
        self._n_examples_test = n_examples_test

    def end_predict(self):
        self._test_time = default_timer() - self._start_time

    def add_search_stats(self, search_stats):
        self._search_stats_merger.add_stats(**search_stats)

    def build(self):
        stats = {}
        stats['train_time'] = self._train_time
        stats['test_time'] = self._test_time
        stats['mean_test_time'] = float(self._test_time) / self._n_examples_test
        stats['n_examples_train'] = self._n_examples_train
        stats['n_examples_test'] = self._n_examples_test
        stats.update(self._search_stats_merger.build())
        return stats