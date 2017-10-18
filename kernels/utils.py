#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from os import path


def compute_P(max_string_length, sigma_position):
    """
    P is a matrix that contains all possible position
    uncertainty values. This function pre-compute all
    possible values since those values are independant of
    the amino acids sequence.
    """

    P = np.zeros((max_string_length, max_string_length))

    for i in xrange(max_string_length):
        for j in xrange(max_string_length):
            P[i, j] = i - j

    P = np.square(P)
    P /= -2.0 * (sigma_position ** 2.0)
    P = np.exp(P)

    return P

def compute_psi_dict(amino_acids, aa_descriptors):
    """
    This function pre-compute the square Euclidean distance
    between all amino acids descriptors and stock the distance
    in an hash table for easy and fast access during the
    GS kernel computation.

    amino_acids -- List of all amino acids in aa_descriptors

    aa_descriptors -- The i-th row of this matrix contain the
        descriptors of the i-th amino acid of amino_acids list.
    """

    # For every amino acids couple (a_1, a_2) psiDict is a hash table
    # that contain the squared Euclidean distance between the descriptors
    # of a_1 and a_2
    psiDict = {}

    # Fill the hash table psiDict
    for i in xrange(len(amino_acids)):
        for j in xrange(len(amino_acids)):
            c = aa_descriptors[i] - aa_descriptors[j]
            psiDict[amino_acids[i], amino_acids[j]] = np.dot(c, c)

    return psiDict


def load_AA_matrix(matrix_path):
    """
    Load the amino acids descriptors.
    Return the list of amino acids and a matrix where
    each row are the descriptors of an amino acid.

    matrix_path -- Path to the file containing the amino acids descriptors.
        See the amino_acid_matrix folder for the file format.
    """

    # Read the file
    f = open(path.expandvars(matrix_path), 'r')
    lines = f.readlines()
    f.close()

    amino_acids = []
    nb_descriptor = len(lines[0].split()) - 1
    aa_descriptors = np.zeros((len(lines), nb_descriptor))

    # Read descriptors
    for i in xrange(len(lines)):
        s = lines[i].split()
        aa_descriptors[i] = np.array([float(x) for x in s[1:]])
        amino_acids.append(s[0])

    # If nb_descriptor == 1, then all normalized aa_descriptors will be 1
    if nb_descriptor > 1:
        # Normalize each amino acid feature vector
        for i in xrange(len(aa_descriptors)):
            aa_descriptors[i] /= np.sqrt(np.dot(aa_descriptors[i], aa_descriptors[i]))

    return amino_acids, aa_descriptors
