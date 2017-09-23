import numpy
from os.path import dirname, join


aa_file = dataset_dir = join(dirname(dirname(__file__)), 'ressources/aa_matrix/AA.blosum62.natural.dat')
aa_file_oboc = dataset_dir = join(dirname(dirname(__file__)), 'ressources/aa_matrix/AA.blosum62.oboc.dat')


def load_amino_acids_and_descriptors(file_name=aa_file):
    path_to_file = join(dirname(__file__), 'amino_acid_matrix', file_name)
    with open(path_to_file, 'r') as data_file:
        lines = data_file.readlines()
    splitted_lines = numpy.array([line.split() for line in lines])
    amino_acids = numpy.array(splitted_lines[:, 0], dtype=numpy.str)
    descriptors = numpy.array(splitted_lines[:, 1:], dtype=numpy.float)
    return amino_acids, descriptors
#

