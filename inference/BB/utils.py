__author__ = 'amelie'

import numpy
from itertools import product
from PBKB.inference.BB.exceptions import InvalidNGramLengthError


def get_n_gram_to_index(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = numpy.arange(len(n_grams))
    n_gram_to_index = dict(zip(n_grams, indexes))
    return n_gram_to_index


def get_index_to_n_gram(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = numpy.arange(len(n_grams))
    index_to_n_gram = dict(zip(indexes, n_grams))
    return index_to_n_gram


def get_n_grams(alphabet, n):
    n = int(n)
    if n <= 0:
        raise InvalidNGramLengthError(n)
    n_grams = [''.join(n_gram) for n_gram in product(alphabet, repeat=n)]
    return n_grams


def transform_strings_to_integer_lists(Y, alphabet):
    letter_to_int = get_n_gram_to_index(alphabet, 1)
    n_examples = numpy.array(Y).shape[0]
    max_length = numpy.max([len(y) for y in Y])
    Y_int = numpy.zeros((n_examples, max_length), dtype=numpy.int8) - 1
    for y_index, y in enumerate(Y):
        for letter_index, letter in enumerate(y):
            Y_int[y_index, letter_index] = letter_to_int[letter]
    return Y_int