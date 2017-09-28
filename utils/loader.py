import numpy
from os.path import dirname, join


def from_pwd(*args):
    return str(join(*args))


package_dir = dirname(dirname(__file__))
dataset_dir = from_pwd(package_dir, 'ressources', 'datasets')
aa_file = from_pwd(package_dir, 'ressources', 'aa_matrix', 'AA.blosum62.natural.dat')
aa_file_oboc = from_pwd(package_dir, 'ressources', 'aa_matrix', 'AA.blosum62.oboc.dat')


def load_amino_acids_and_descriptors(file_name=aa_file):
    with open(file_name, 'r') as data_file:
        lines = data_file.readlines()
    splitted_lines = numpy.array([line.split() for line in lines])
    amino_acids = numpy.array(splitted_lines[:, 0], dtype=numpy.str)
    descriptors = numpy.array(splitted_lines[:, 1:], dtype=numpy.float)
    return amino_acids, descriptors
#

if __name__ == '__main__':
    print package_dir
    print dataset_dir
    print aa_file
    print aa_file_oboc