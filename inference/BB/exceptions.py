__author__ = 'amelie'


class InvalidNGramError(ValueError):
    def __init__(self, n, n_gram):
        self.n = n
        self.n_gram = n_gram

    def __str__(self):
        error_message = "{} is not a possible {:d}_gram for this alphabet".format(self.n_gram, self.n)
        return error_message


class InvalidNGramLengthError(ValueError):
    def __init__(self, n, min_n=0):
        self.n = n
        self.min_n = min_n

    def __str__(self):
        error_message = 'n must be greater than {:d}. Got: n={:d}'.format(self.min_n, self.n)
        return error_message


class InvalidYLengthError(ValueError):
    def __init__(self, n, y_length, is_min=True):
        self.n = n
        self.y_length = y_length
        self.n_name = 'min_n' if is_min else 'max_n'

    def __str__(self):
        error_message = 'y_length must be >= min_n. Got: y_length={:d}, min_n={:d}'.format(self.y_length, self.n)
        return error_message


class InvalidMinLengthError(ValueError):
    def __init__(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def __str__(self):
        error_message = 'min_length must be <= max_length. ' \
                        'Got: min_length={:d}, max_length={:d}'.format(self.min_length, self.max_length)
        return error_message


class NoThresholdsError(ValueError):
    def __str__(self):
        error_message = 'thresholds must be provided when y_length is None'
        return error_message


class NoYLengthsError(ValueError):
    def __str__(self):
        error_message = "y_lengths can't be None if is_using_length = True"
        return error_message


class InvalidShapeError(ValueError):
    def __init__(self, parameter_name, parameter_shape, valid_shapes):
        self.parameter_name = parameter_name
        self.parameter_shape = parameter_shape
        self.valid_shapes = [str(valid_shape) for valid_shape in valid_shapes]

    def __str__(self):
        valid_shapes_string = ' or '.join(self.valid_shapes)
        error_message = "{} wrong shape: Expected: {} Got: {}".format(self.parameter_name, valid_shapes_string,
                                                                      str(self.parameter_shape))
        return error_message