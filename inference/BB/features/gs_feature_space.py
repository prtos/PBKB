__author__ = 'amelie'

import numpy

from PBKB.inference.BB.features.string_feature_space import build_feature_space_for_each_n_gram_length
from PBKB.inference.BB.exceptions import InvalidYLengthError, InvalidMinLengthError

# Todo merge similar code from gs, weighted degree and n_gram feature spaces
class GenericStringFeatureSpace:
    """Output feature space for the Generic String kernel with position weights

    Creates a sparse matrix representation of the n-grams in each training string. The representation takes in account
    the positions of the n-grams in the strings, This is used to compute the weights of the graph during the inference
    phase. This doesn't take in account the similarity between the n-grams (no sigma_c).

    Attributes
    ----------
    n : int
        N-gram length.
    sigma_position : float
        Parameter of the Generic String Kernel controlling the penalty incurred when two n-grams are not sharing the
        same position.
    max_n_gram_count : int
        The number of n-grams in the training string of highest length.
    feature_space : sparse matrix, shape = [n_samples, max_n_gram_count * len(alphabet)**n]
        Sparse matrix representation of the n-grams in each training string, where n_samples is the number of training
        samples.
    """

    def __init__(self, alphabet, Y, min_n, max_n, kernel, is_normalized):
        """Create the output feature space for the Generic String kernel

        Parameters
        ----------
        alphabet : list
            list of letters
        n : int
            n-gram length
        Y : array, [n_samples, ]
            The training strings.
        sigma_position : float
            Parameter of the Generic String Kernel controlling the penalty incurred when two n-grams are not sharing the
            same position.
        is_normalized : bool
            True if the feature space should be normalized, False otherwise.
        """
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.kernel = kernel
        self._alphabet_length = len(alphabet)
        self._max_y_length = numpy.max([len(y) for y in Y])
        self._feature_spaces = build_feature_space_for_each_n_gram_length(alphabet, Y, self.min_n, self.max_n, True)
        self._alphabet_length = len(alphabet)
        self._normalize(self._feature_spaces, Y, is_normalized)

    def _get_max_n_gram_count(self, alphabet_n_gram_count, feature_space):
        n_columns = feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def _normalize(self, feature_spaces, Y, is_normalized):
        if is_normalized:
            y_y_similarity = self.kernel.element_wise_kernel(Y)
            y_normalization = 1. / numpy.sqrt(y_y_similarity)
            for n in range(self.min_n, self.max_n + 1):
                data_normalization = y_normalization.repeat(numpy.diff(feature_spaces[n].indptr))
                feature_spaces[n].data *= data_normalization

    def _get_n_gram_count_in_each_y(self, n, Y):
        y_n_gram_counts = numpy.array([len(y) - n + 1 for y in Y])
        return y_n_gram_counts

    def compute_weights(self, y_weights, y_length):
        """Compute the inference graph weights

        Parameters
        ----------
        y_weights :  array, [n_samples]
            Weight of each training example.
        y_length : int
            Length of the string to predict.

        Returns
        -------
        gs_weights : [y_n_gram_count, len(alphabet)**n]
            Weight of each n-gram at each position.
        """
        self._check_y_length_greater_than_min_n(y_length)
        y_max_n_gram_count = max(y_length - self.max_n + 1, 1)
        weights = numpy.zeros((y_max_n_gram_count, self._alphabet_length ** self.max_n))
        end_weights = numpy.zeros(self._alphabet_length ** self.max_n)
        for n in range(self.min_n, self.max_n + 1):
            if y_length >= n:
                n_gram_weights = self._get_weight_of_each_n_gram_at_each_position(y_weights, y_length, n)
                weights += self._transform_weights_in_max_n_gram_weights(n_gram_weights, y_max_n_gram_count, n)
                end_weights += self._compute_end_weights(n_gram_weights, y_max_n_gram_count, n, y_length)
        return weights, end_weights

    def compute_weights_in_length_range(self, y_weights, min_y_length, max_y_length):
        """Compute the inference graph weights

        Parameters
        ----------
        y_weights :  array, [n_samples]
            Weight of each training example.
        y_length : int
            Length of the string to predict.

        Returns
        -------
        weighted_degree_weights : [len(alphabet)**n, y_n_gram_count * len(alphabet)**n]
            Weight of each n-gram at each position.
        """
        self._check_y_length_greater_than_min_n(min_y_length)
        self._check_min_length_greater_than_max_length(min_y_length, max_y_length)
        y_max_n_gram_count = max(max_y_length - self.max_n + 1, 1)
        length_count = max_y_length - min_y_length + 1
        weights = numpy.zeros((y_max_n_gram_count, self._alphabet_length ** self.max_n))
        end_weights = numpy.zeros((length_count, self._alphabet_length ** self.max_n))
        for n in range(self.min_n, self.max_n + 1):
            if max_y_length >= n:
                n_gram_weights = self._get_weight_of_each_n_gram_at_each_position(y_weights, max_y_length, n)
                weights += self._transform_weights_in_max_n_gram_weights(n_gram_weights, y_max_n_gram_count, n)
                self._add_end_weights_for_each_length(end_weights, min_y_length, max_y_length, n, n_gram_weights)
        return weights, end_weights

    def _get_weight_of_each_n_gram_at_each_position(self, y_weights, y_length, n):
        data_copy = numpy.copy(self._feature_spaces[n].data)
        self._feature_spaces[n].data *= self._repeat_each_y_weight_by_y_column_count(y_weights, n)
        weight_vector = numpy.array(self._feature_spaces[n].sum(axis=0))[0].reshape(self._max_y_length - n + 1, -1)
        self._feature_spaces[n].data = data_copy
        gs_weights = self._transform_in_gs_weights(y_length - n + 1, weight_vector, n)
        return gs_weights

    def _transform_weights_in_max_n_gram_weights(self, n_gram_weights, y_max_n_gram_count, n):
        n_repeat = self._alphabet_length ** (self.max_n - n)
        max_n_gram_weights = numpy.repeat(n_gram_weights[0:y_max_n_gram_count], n_repeat, axis=1)
        return max_n_gram_weights

    def _add_end_weights_for_each_length(self, end_weights, min_y_length, max_y_length, n, gs_weights):
        for i, length in enumerate(range(min_y_length, max_y_length + 1)):
            if length > n:
                length_max_n_gram_count = max(length - self.max_n + 1, 1)
                end_weights[i] += self._compute_end_weights(gs_weights, length_max_n_gram_count, n, length)

    def _compute_end_weights(self, gs_weights, y_max_n_gram_count, n, y_length):
        end_weights = numpy.zeros(self._alphabet_length ** self.max_n)
        for i in range(min(self.max_n, y_length) - n):
            n_tile = self._alphabet_length ** (i + 1)
            n_repeat = self._alphabet_length ** self.max_n / (gs_weights.shape[1] * n_tile)
            end_weights += numpy.repeat(numpy.tile(gs_weights[y_max_n_gram_count + i], n_tile), n_repeat)
        return end_weights

    def _transform_in_gs_weights(self, y_n_gram_count, weighted_degree_weights, n):
        gs_weights = numpy.empty((y_n_gram_count, self._alphabet_length ** n))
        for i in range(y_n_gram_count):
            position_weights = self.kernel.compute_position_weights(i, self._max_y_length - n + 1)
            gs_weights[i, :] = (weighted_degree_weights * position_weights.reshape(-1, 1)).sum(axis=0)
        return gs_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights, n):
        return y_weights.repeat(numpy.diff(self._feature_spaces[n].indptr))

    def _check_y_length_greater_than_min_n(self, y_length):
        if y_length < self.min_n:
            raise InvalidYLengthError(self.min_n, y_length)

    def _check_min_length_greater_than_max_length(self, min_length, max_length):
        if min_length > max_length:
            raise InvalidMinLengthError(min_length, max_length)