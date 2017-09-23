#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
---------------------------------------------------------------------
Copyright 2015 Sebastien Giguere

This file is part of peptide_design

peptide_design is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

peptide_design is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with peptide_design.  If not, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------
'''

import numpy as np
from itertools import repeat
from cartesian import cartesian
from gs_kernel.gs_kernel_slow import load_AA_matrix, compute_psi_dict 
from gs_kernel.gs_kernel import compute_P

try:
    from preimage_fast import k_longest_path as fast_k_longest_path
    from preimage_fast import W_weight as fast_W_weight
    FAST_MODE = True
except:
    print "WARNING: You are using the slow implementation. See README for compilation of the fast Cython version."
    FAST_MODE = False

PATH_LENGTH_DTYPE = np.float64
INT_AMINO_ACID_DTYPE = np.int8
EDGES_ID_DTYPE = np.uint64


class Preimage(object):
    def __init__(self, peptide_length, training_peptides, beta, amino_acid_property_file, sigma_position, sigma_amino_acid, substring_length):
        '''
        peptide_length : int
            The length of the peptide.
            
        training_peptides : list, np.array
            The peptides amino acids sequences of the training set.
        
        beta : np.array
            Weight on training examples.
            
        amino_acid_property_file : string
            Path to the amino acid property file.
        
        sigma_position: float
            \sigma_p hyper-parameter of the GS kernel.
            
        sigma_amino_acid: float
            \sigma_c hyper-parameter of the GS kernel.
        
        substring_length: int
            Substring length hyper-parameter of the GS kernel
        '''
        
        if peptide_length < substring_length:
            raise Exception("The peptide length has to be greater or equal to the substring length.")
        
        # Override function with faster implementation
        if FAST_MODE:
            print "Using Cython fast implementation"
            self.k_longest_path = self.__k_longest_path_wrapper
            self.W_weight = self.__W_weight_wraper
        
        # Length of peptides
        self.peptide_length = peptide_length
        
        # Weight vector on each training example
        self.beta = beta
        
        # \sigma_p hyper-parameter of the GS kernel
        self.sigma_position = sigma_position
        
        # \sigma_c hyper-parameter of the GS kernel
        self.sigma_amino_acid = sigma_amino_acid
        
        # Substring length hyper-parameter of the GS kernel
        self.substring_length = substring_length
        
        # Load amino acids properties from file
        (amino_acids, aa_descriptors) = load_AA_matrix(amino_acid_property_file)
        self.aminoacid_count = len(amino_acids)
        
        # Dictionary to switch from amino acids string to integer and vice versa
        self.aa_to_int = dict(zip(amino_acids,range(self.aminoacid_count)))
        self.int_to_aa = dict(zip(range(self.aminoacid_count), amino_acids))
        
        # Length of peptides from the training set
        self.training_peptides_length = np.array([len(x) for x in training_peptides], dtype=np.int)
        
        # Maximum length of peptides
        max_length = max(peptide_length, max(self.training_peptides_length))
        
        # Peptides from the training set
        self.training_peptides = np.array([self.encode_peptides(x,max_length) for x in training_peptides], dtype=INT_AMINO_ACID_DTYPE)
        
        # Pre-compute the euclidean distance between every amino acids
        self.psi_matrix = self.psi_dict_to_matrix(compute_psi_dict(amino_acids, aa_descriptors))
        
        # Pre-compute the position penalization of the GS kernel for all position
        self.position_matrix = compute_P(max_length, sigma_position)
        
        # Edge of the graph that are blacklisted
        self.edge_blacklist = np.zeros(0,dtype=EDGES_ID_DTYPE)
    
    
    def __k_longest_path_wrapper(self, k):
        return fast_k_longest_path(self, k)
    
    def __W_weight_wraper(self, substring, substring_position):
        return fast_W_weight(substring, substring_position, self.training_peptides, self.training_peptides_length, self.psi_matrix, self.position_matrix, self.beta)
    
    def longest_path(self, spur_node_position = 0, spur_node = None):
        """
        Find the longest path in a multi-partite graph.
        This is an implementation of algorithm (1).
        
        
        spur_node_position : int
            Position of the spur node in the multi-partite graph,
            0 means the true source node of the graph.
            The spur node is used in the k-longest path algorithm
            were we find the longest path starting at a specific node
            not only from the source node of the graph.
        
        spur_node : np.array
            Identifier of the spur node. ex: np.array([11,4,19])
            for self.substring_length = 3.
            Recall that each node are identified using by a
            substring of length self.substring_length
        """
        
        # The "n" of the n-partite graph
        n = self.peptide_length - self.substring_length + 1
        
        # The spur node is just before the sink node: spur_node -> t
        # The longest path is trivial
        if spur_node_position == n:
            return spur_node[1:], self.Wt_weight(spur_node[1:], n)
        
        # Arrays for the dynamic programming
        length_to = np.zeros((2, self.aminoacid_count**self.substring_length), dtype=PATH_LENGTH_DTYPE) - PATH_LENGTH_DTYPE(np.inf)
        predecessor = np.zeros((n, self.aminoacid_count**self.substring_length), dtype=INT_AMINO_ACID_DTYPE)
        
        edge_index = 0
        
        # Edges leaving the source node
        if spur_node_position == 0:
            a_index=0
            for a in cartesian(repeat(np.arange(self.aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=self.substring_length)):
                if not self.is_blacklisted(edge_index):
                    length_to[0,a_index] = self.W_weight(a, 0)
                a_index +=1
                edge_index += 1
        # The source is a spur node
        else:
            # Keep the count 
            edge_index += self.aminoacid_count**self.substring_length
        
        # Visit each edges of the n-partite graph using a pre-defined topological order
        # Edges are always of the form: a -> s
        for i in xrange(1,n):

            # Initialize the length_to vector
            length_to[i%2] = length_to[i%2] - PATH_LENGTH_DTYPE(np.inf)

            a_index = 0     # Do not need anymore
            s_index = 0

            # Edges are not attainable from the spur node.
            # i.e. edges on the left side of the spur node
            if i < spur_node_position:
                # Keep track of the edge index
                edge_index += self.aminoacid_count**(self.substring_length + 1)

            # Edges at the spur node position
            elif i == spur_node_position:
                for s in cartesian(repeat(np.arange(self.aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=self.substring_length)):

                    # Weight on the edge a -> s
                    edge_weight = self.W_weight(s, i)

                    a_index = (s_index - s[-1]) / self.aminoacid_count

                    for a_prime in xrange(self.aminoacid_count):

                        # a is obtained partly from s
                        a = np.roll(s,1)
                        a[0] = a_prime

                        # Only edges of the form: spur_node -> s
                        if np.alltrue(a == spur_node) and not self.is_blacklisted(edge_index):
                            # Weight on the edge spur_node -> s
                            length_to[i%2, s_index] = edge_weight

                            predecessor[i, s_index] = a_prime

                        edge_index += 1
                        a_index += self.aminoacid_count**(self.substring_length-1)

                    s_index += 1

            # Edges on the right side of the spur node
            else:
                for s in cartesian(repeat(np.arange(self.aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=self.substring_length)):

                    # Weight on the edge a -> s
                    edge_weight = self.W_weight(s, i)

                    a_index = (s_index - s[-1]) / self.aminoacid_count

                    for a_prime in xrange(self.aminoacid_count):
                        if not self.is_blacklisted(edge_index):

                            # Edge a -> s
                            if length_to[i%2, s_index] < length_to[(i-1)%2, a_index] + edge_weight:
                                length_to[i%2, s_index] = length_to[(i-1)%2, a_index] + edge_weight
                                predecessor[i, s_index] = a_prime

                        edge_index += 1
                        a_index += self.aminoacid_count**(self.substring_length-1)

                    s_index += 1

        length_to_t = PATH_LENGTH_DTYPE(-np.inf)
        predecessor_of_t = None
        a_index = 0

        # Edges heading to the sink node
        for a in cartesian(repeat(np.arange(self.aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=self.substring_length)):

            # Their is no point at blacklisting an edge heading
            # to the sink node but it's possible!
            if not self.is_blacklisted(edge_index):
                edge_weight = self.Wt_weight(a[1:], n)

                if length_to_t < length_to[(n-1)%2, a_index] + edge_weight:
                    length_to_t = length_to[(n-1)%2, a_index] + edge_weight
                    predecessor_of_t = np.array(a, dtype=INT_AMINO_ACID_DTYPE)

            a_index += 1
            edge_index += 1

        # There is no path between the source and the sink
        if predecessor_of_t == None:
            return np.zeros(0, dtype=INT_AMINO_ACID_DTYPE), -np.Inf

        # Initialise the peptide sequence
        peptide = np.zeros(self.peptide_length, dtype=INT_AMINO_ACID_DTYPE)
        peptide[-self.substring_length:] = predecessor_of_t

        # Revert the path using the predecessors to find the peptide
        for i in range(self.peptide_length - self.substring_length)[::-1]:

            # Compute the prececessor index
            predecessor_index = 0
            for j in xrange(self.substring_length):
                predecessor_index += peptide[i+j+1] * self.aminoacid_count**(self.substring_length-j-1)

            # Keep track of the peptide sequence
            peptide[i] = predecessor[i+1,predecessor_index]

        # Return the encoded peptide sequence and the binding affinity (path length)
        return peptide[spur_node_position:], length_to_t


    def k_longest_path(self, k):
        """
        Return the k-longest path, i.e. the k peptides with the greater binding affinty.
        """

        if k > self.aminoacid_count**self.peptide_length:
            raise Exception("There is only %i^%i peptides of length %i"%(self.aminoacid_count, self.peptide_length, self.peptide_length))

        A = np.zeros((k, self.peptide_length), dtype=INT_AMINO_ACID_DTYPE)
        A_length = np.zeros(k) - np.inf

        # Initialize the heap to store the potential kth longest path
        B = np.zeros((self.peptide_length * k, self.peptide_length), dtype=INT_AMINO_ACID_DTYPE)
        B_length = np.zeros(self.peptide_length * k) - np.inf
        B_index = 0

        # Eugene Lawler proposed a modification to Yen's algorithm in which duplicates
        # path are not calculated as opposed to the original algorithm where they are
        # calculated and then discarded when they are found to be duplicates.
        lawler_dict = {}

        # Determine the shortest path from the source to the sink
        path, path_length = self.longest_path()
        A[0] = path
        A_length[0] = path_length

        #print "Longest path #0 :", path, "\t Length:", path_length

        for i in xrange(1, k):
            # The spur node is the graph source node
            for path in A[:i]:
                self.blacklist_edge(path[0:self.substring_length], 0)

            # Lawler's modification
            if not lawler_dict.has_key((tuple(self.edge_blacklist),)):
                spur_path, path_length = self.longest_path()
                lawler_dict[(tuple(self.edge_blacklist),)] = True

                # If there is a path
                if path_length > -np.Inf:
                    # Add the path to the heap
                    B[B_index] = spur_path
                    B_length[B_index] = path_length
                    B_index += 1

            self.clear_blacklist()

            # The spur node ranges from the first node to the next to last node in the longest path
            for j in xrange(0, self.peptide_length - self.substring_length):

                # Spur node is retrieved from the previous i-shortest path, i − 1
                spur_node = A[i-1][j:j+self.substring_length]

                # The sequence of nodes from the source to the spur node of the previous i-longest path
                root_path = A[i-1][0:j+self.substring_length]

                for path in A[:i]:
                    if np.alltrue(root_path == path[0:j+self.substring_length]):
                        # Remove the links that are part of the previous shortest paths which share the same root path
                        self.blacklist_edge(path[j:j+self.substring_length+1], j)

                # Calculate the spur path from the spur node to the sink.
                # Lawler's modification
                if lawler_dict.has_key((tuple(self.edge_blacklist), j+1, tuple(spur_node))):
                    spur_path, spur_path_length = lawler_dict[ (tuple(self.edge_blacklist), j+1, tuple(spur_node)) ]
                else:
                    spur_path, spur_path_length = self.longest_path(j+1, spur_node)
                    lawler_dict[(tuple(self.edge_blacklist), j+1, tuple(spur_node))] = (spur_path.copy(), spur_path_length)

                # Entire path is made up of the root path and spur path
                total_path = np.append(root_path[0:j+1], spur_path)
                total_path_length = self.length_to_spur_node(root_path) + spur_path_length

                # If there is a path
                if total_path_length > -np.Inf:
                    # Add the potential i-longest path to the heap
                    # Only if the path is not already in B
                    if np.sum(np.alltrue(B[:B_index] == total_path, axis=1)) == 0:
                        B[B_index] = total_path
                        B_length[B_index] = total_path_length
                        B_index += 1

                # Add back the edges that were removed from the graph
                self.clear_blacklist()

            # Sort B
            s = np.argsort(B_length)

            new_path = B[s][-i]
            new_path_length = B_length[s][-i]

            # Add the maximal cost path becomes the i-longest path
            A[i] = new_path
            A_length[i] = new_path_length

            #print "Longest path #"+str(i)+" :", new_path, "\t Length:", new_path_length

        return np.array([self.decode_peptides(x) for x in A]), np.array(A_length)


    def W_weight(self, substring, substring_position):
        """
        Weight function given at equation (8) for edges of the graph
        substring -> s from equation 8
        substring_position -> i from equation 8
        """

        weight = 0.0
        substring_length = substring.shape[0]

        # Sum over training examples
        for n in xrange(self.training_peptides.shape[0]):
            example_length = self.training_peptides_length[n]

            # The GS kernel part
            kernel = 0.0
            max_length = substring_length

            for j in range(example_length):
                if example_length-j < max_length:
                    max_length = (example_length-j)

                tmp = 1.0
                b = 0.0

                for l in range(max_length):
                    tmp *= self.psi_matrix[ substring[l], self.training_peptides[n, j+l] ]
                    b += tmp

                kernel += self.position_matrix[substring_position,j] * b

            weight += self.beta[n] * kernel

        return weight


    def Wt_weight(self, substring, position):
        """
        Weight function for edges heading to the sink node
        given at equation (9) of manuscript
        """
        weight = 0.0
        for i in xrange(substring.shape[0]):
            weight += self.W_weight(substring[i:], i+position)

        return weight


    def length_to_spur_node(self, path):
        """
        Compute the length of the path from the source node
        to the spur_node
        """

        length = 0.0
        n = len(path) - self.substring_length + 1
        for k in xrange(n):
            length += self.W_weight(path[k:k+self.substring_length], k)

        return length


    def is_blacklisted(self, edge_index):
        """
        Since the graph is never constructed this function
        allows edges to be black listed. A path cannot use
        a blacklisted edge.
        """
        if self.edge_blacklist.shape[0] == 0:
            return False
        else:
            return not np.alltrue(self.edge_blacklist != edge_index)


    def blacklist_edge(self, edge_sequence, position):
        """
        Blacklisting an edge is equivalent to blacklisting the use
        of a substring at a certain position in the peptide.

        For example, for k=3:
        - If the edge s -> abc is blacklisted
        the substring abc cannot be produced at that position 0.

        - If the edge cde -> t is blacklisted
        the substring cde cannot end the peptide

        - If the edge abc -> bcd is blacklised at some position n
        the substring starting abcd cannot be at position n.

        """
        edge_id = 0
        edge_length = edge_sequence.shape[0]

        # Only consider the "self.substring_length" right most amino acid
        for i in xrange(self.substring_length):
            edge_id += edge_sequence[edge_length-i-1] * self.aminoacid_count**(i)

        # This is an edge from the core of the graph
        if edge_length == self.substring_length + 1:
            # Account for the first amino acid
            edge_id *= self.aminoacid_count
            edge_id += edge_sequence[0]

            # Account for edges leaving the source node
            edge_id += self.aminoacid_count**self.substring_length

            if position > 0:
                # Account for edges from the core of the graph i.e. the (position -1)-partite of the graph
                edge_id += position * self.aminoacid_count**(self.substring_length+1)

        if not self.is_blacklisted(edge_id):
            self.edge_blacklist = np.append(self.edge_blacklist, np.uint64(edge_id))


    def clear_blacklist(self):
        """
        Clear the edge edge_blacklist
        """
        self.edge_blacklist = np.zeros(0,dtype=EDGES_ID_DTYPE)


    def encode_peptides(self, str_peptide, max_length = None):
        """
        Use a compact integer representation to encode amino acids sequence
        into int8 array

        str_peptide -- string of amino acids
        max_length -- Used to have array of the same length,
            unused position are padded with -1

        """

        # Initialise the array and pad with -1 values if necessary
        int_peptide = np.zeros(max(len(str_peptide), max_length), dtype=INT_AMINO_ACID_DTYPE) - 1

        # Encode each amino acid
        for i in xrange(len(str_peptide)):
            try:
                int_peptide[i] = self.aa_to_int[str_peptide[i]]
            except KeyError:
                raise Exception("Unknown amino acid: " + str_peptide[i])

        return int_peptide


    def decode_peptides(self, int_peptide):
        """
        Use a compact integer representation of amino acids
        to decode int8 array into amino acid sequence
        """

        str_peptide = ""

        for i in xrange(len(int_peptide)):
            str_peptide += self.int_to_aa[int_peptide[i]]
            
        return str_peptide
    
    
    def psi_dict_to_matrix(self, psi_dict):
        """
        It is possible to get a perfect hash using a squared matrix containing
        the squared Euclidean distance between all possible amino acids.
        
        This hasing function differ from the one currently used in the GS kernel
        implementation (ASCII value of the amino acid).
        Here we use a compact hashing function to ensure that each
        aminoacid is maped to a integer \in [0, aminoacid_count -1].
        
        
        """
        
        N = self.aminoacid_count
        psi_matrix = np.zeros((N,N))+4
        idx = np.arange(N)
        psi_matrix[idx,idx] = 0 # set diagonal to zero
        
        for key, val in psi_dict.items():
            i = self.aa_to_int[key[0]]
            j = self.aa_to_int[key[1]]
            psi_matrix[i,j] = val
            
        return np.exp( -psi_matrix/(2.0 * (self.sigma_amino_acid**2)))
    
    
