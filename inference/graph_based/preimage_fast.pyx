# cython: profile=False
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

#Cython stuff
import cython
cimport numpy as np
np.import_array()
from cpython cimport bool

ctypedef np.float64_t FLOAT64_t
ctypedef np.int8_t INT8_t
ctypedef np.int64_t INT64_t

PATH_LENGTH_DTYPE = np.float64
ctypedef np.float64_t PATH_LENGTH_DTYPE_t

INT_AMINO_ACID_DTYPE = np.int8
ctypedef np.int8_t INT_AMINO_ACID_DTYPE_t

EDGES_ID_DTYPE = np.uint64
ctypedef np.uint64_t EDGES_ID_DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool all_equal(INT_AMINO_ACID_DTYPE_t[::1] x,
                    INT_AMINO_ACID_DTYPE_t[::1] y):
    """
    Return True if x and y are all equal,
    False otherwise.
    """
                    
    cdef int i
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            return False
    
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef roll_right(INT_AMINO_ACID_DTYPE_t[::1] source,
                INT_AMINO_ACID_DTYPE_t[::1] target):
    """
    Roll source right one position.
    Same as np.roll(s,1)
    """
    
    cdef int length, i
    length = source.shape[0]
    
    target[0] = source[length-1]
    
    for i in xrange(length-1):
        target[i+1] = source[i]
    
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT64_t W_weight(   INT_AMINO_ACID_DTYPE_t[::1] substring,
                            int substring_position,
                            INT_AMINO_ACID_DTYPE_t[:,::1] training_peptides,
                            INT64_t[::1] training_peptides_length,
                            FLOAT64_t[:,::1] psi_matrix,
                            FLOAT64_t[:,::1] position_matrix,
                            FLOAT64_t[::1] beta):
    """
    Weight function given at equation (8) for edges of the graph
    substring -> s from equation 8
    substring_position -> i from equation 8
    """
    cdef int n, j, l, example_length, substring_length, max_length
    cdef FLOAT64_t tmp, b, kernel, weight
    
    weight = 0.0
    substring_length = substring.shape[0]
    
    # Sum over training examples
    for n in xrange(training_peptides.shape[0]):
        example_length = training_peptides_length[n]
        
        # The GS kernel part
        kernel = 0.0
        max_length = substring_length
        
        for j in range(example_length):
            if example_length-j < max_length:
                max_length = (example_length-j)
            
            tmp = 1.0
            b = 0.0
            
            for l in range(max_length):
                tmp *= psi_matrix[ substring[l], training_peptides[n, j+l] ]
                b += tmp
                
            kernel += position_matrix[substring_position,j] * b
        
        weight += beta[n] * kernel
    
    return weight

@cython.boundscheck(False)
@cython.wraparound(False)
cdef FLOAT64_t Wt_weight(   INT_AMINO_ACID_DTYPE_t[::1] substring,
                            int position,
                            INT_AMINO_ACID_DTYPE_t[:,::1] training_peptides,
                            INT64_t[::1] training_peptides_length,
                            FLOAT64_t[:,::1] psi_matrix,
                            FLOAT64_t[:,::1] position_matrix,
                            FLOAT64_t[::1] beta):
    """
    Weight function for edges heading to the sink node
    given at equation (9) of manuscript
    """
    cdef FLOAT64_t weight
    cdef int i
    
    weight = 0.0
    for i in xrange(substring.shape[0]):
        weight += W_weight(substring[i:], i+position, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
    
    return weight

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bool is_blacklisted(    EDGES_ID_DTYPE_t edge_index,
                                    EDGES_ID_DTYPE_t[::1] edge_blacklist,
                                    int edge_blacklist_count):

    """
    Since the graph is never constructed this function
    allows edges to be black listed. A path cannot use 
    a blacklisted edge.
    """

    cdef int i
    
    for i in xrange(edge_blacklist_count):
        if edge_blacklist[i] == edge_index:
            return True
    
    return False

@cython.boundscheck(False)
def longest_path(INT_AMINO_ACID_DTYPE_t[:,::1] training_peptides,
                 INT64_t[::1] training_peptides_length,
                 FLOAT64_t[:,::1] psi_matrix,
                 FLOAT64_t[:,::1] position_matrix,
                 FLOAT64_t[::1] beta,
                 EDGES_ID_DTYPE_t[::1] edge_blacklist,
                 int edge_blacklist_count,
                 int aminoacid_count,
                 int substring_length,
                 int peptide_length,
                 int spur_node_position = 0,
                 spur_node = None):
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
    
    # Table for dynamic programming
    cdef PATH_LENGTH_DTYPE_t[:,::1] length_to
    cdef INT_AMINO_ACID_DTYPE_t[:,::1] predecessor
    
    cdef INT_AMINO_ACID_DTYPE_t[::1] a, s, predecessor_of_t, peptide
    cdef INT_AMINO_ACID_DTYPE_t[:,::1] C
    
    cdef int i, j, n, a_index, s_index, predecessor_index
    cdef EDGES_ID_DTYPE_t edge_index
    cdef FLOAT64_t edge_weight
    cdef PATH_LENGTH_DTYPE_t length_to_t
    cdef INT_AMINO_ACID_DTYPE_t a_prime
    
    # The "n" of the n-partite graph
    n = peptide_length - substring_length + 1
    
    # The spur node is just before the sink node: spur_node -> t
    # The longest path is trivial
    if spur_node_position == n:
        return spur_node[1:], Wt_weight(spur_node[1:], n, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
    
    # Arrays for the dynamic programming
    length_to = np.zeros((2, aminoacid_count**substring_length), dtype=PATH_LENGTH_DTYPE) - PATH_LENGTH_DTYPE(np.inf)
    predecessor = np.zeros((n, aminoacid_count**substring_length), dtype=INT_AMINO_ACID_DTYPE)
    
    edge_index = 0
    
    # Edges leaving the source node
    if spur_node_position == 0:
        a_index=0
        C = cartesian(repeat(np.arange(aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=substring_length))
        for i in xrange(aminoacid_count**substring_length):
            a = C[i]
            
            if not is_blacklisted(edge_index, edge_blacklist, edge_blacklist_count):
                length_to[0,a_index] = W_weight(a, 0, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
            
            a_index +=1
            edge_index += 1
    
    # The source is a spur node
    else:
        # Keep the count 
        edge_index += aminoacid_count**substring_length
    
    # Visit each edges of the n-partite graph using a pre-defined topological order
    # Edges are always of the form: a -> s
    for i in xrange(1,n):
        
        # Initialize the length_to vector
        length_to[i%2,:] = - PATH_LENGTH_DTYPE(np.inf)
        
        a_index = 0
        s_index = 0
        
        # Edges are not attainable from the spur node.
        # i.e. edges on the left side of the spur node
        if i < spur_node_position:
            # Keep track of the edge index
            edge_index += aminoacid_count**(substring_length + 1)
        
        # Edges at the spur node position
        elif i == spur_node_position:
            a = np.zeros(substring_length, dtype=INT_AMINO_ACID_DTYPE)
            
            C = cartesian(repeat(np.arange(aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=substring_length))
            for j in range(aminoacid_count**substring_length):
                s = C[j]
                # Weight on the edge a -> s
                edge_weight = W_weight(s, i, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
                
                a_index = (s_index - s[substring_length-1]) / aminoacid_count
                
                for a_prime in range(aminoacid_count):
                    
                    # a is obtained partly from s
                    roll_right(s,a)
                    a[0] = a_prime
                    
                    # Only edges of the form: spur_node -> s 
                    if all_equal(a,spur_node) and not is_blacklisted(edge_index, edge_blacklist, edge_blacklist_count):
                        # Weight on the edge spur_node -> s
                        length_to[i%2, s_index] = edge_weight
                        
                        predecessor[i, s_index] = a_prime
                    
                    edge_index += 1
                    a_index += aminoacid_count**(substring_length-1)
                    
                s_index += 1
        
        # Edges on the right side of the spur node
        else:
            C = cartesian(repeat(np.arange(aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=substring_length))
            for j in range(aminoacid_count**substring_length):
                s = C[j]
                # Weight on the edge a -> s
                edge_weight = W_weight(s, i, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
                
                a_index = (s_index - s[substring_length-1]) / aminoacid_count
                
                for a_prime in range(aminoacid_count):
                    if not is_blacklisted(edge_index, edge_blacklist, edge_blacklist_count):
                        
                        # Edge a -> s
                        if length_to[i%2, s_index] < length_to[(i-1)%2, a_index] + edge_weight:
                            length_to[i%2, s_index] = length_to[(i-1)%2, a_index] + edge_weight
                            predecessor[i, s_index] = a_prime
                    
                    edge_index += 1
                    a_index += aminoacid_count**(substring_length-1)
                
                s_index += 1
    
    length_to_t = PATH_LENGTH_DTYPE(-np.inf)
    a_index = 0
    
    # Edges heading to the sink node
    C = cartesian(repeat(np.arange(aminoacid_count, dtype=INT_AMINO_ACID_DTYPE),times=substring_length))
    for i in xrange(aminoacid_count**substring_length):
        a = C[i]
        # Their is no point at blacklisting an edge heading
        # to the sink node but it's possible!
        if not is_blacklisted(edge_index, edge_blacklist, edge_blacklist_count):
            edge_weight = Wt_weight(a[1:], n, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)

            if length_to_t < length_to[(n-1)%2, a_index] + edge_weight:
                length_to_t = length_to[(n-1)%2, a_index] + edge_weight
                predecessor_of_t = a
        
        a_index += 1
        edge_index += 1
    
    # There is no path between the source and the sink
    if length_to_t == -np.inf:
        return np.zeros(0, dtype=INT_AMINO_ACID_DTYPE), -np.Inf
    
    # Initialise the peptide sequence
    peptide = np.zeros(peptide_length, dtype=INT_AMINO_ACID_DTYPE)
    peptide[-substring_length:] = predecessor_of_t
    
    # Revert the path using the predecessors to find the peptide
    for i in range(peptide_length - substring_length)[::-1]:
        
        # Compute the prececessor index
        predecessor_index = 0
        for j in xrange(substring_length):
            predecessor_index += peptide[i+j+1] * aminoacid_count**(substring_length-j-1)
        
        # Keep track of the peptide sequence
        peptide[i] = predecessor[i+1,predecessor_index]
    
    # Return the encoded peptide sequence and the binding affinity (path length)
    return np.array(peptide[spur_node_position:]), length_to_t   

#@cython.boundscheck(False)
#@cython.wraparound(False)   
def k_longest_path(self, int k):
    """
    Return the k-longest path, i.e. the k peptides with the greater binding affinty.
    """
    
    # Typing variables from self
    cdef INT_AMINO_ACID_DTYPE_t[:,::1] training_peptides = self.training_peptides
    cdef INT64_t[::1] training_peptides_length = self.training_peptides_length
    cdef FLOAT64_t[:,::1] psi_matrix = self.psi_matrix
    cdef FLOAT64_t[:,::1] position_matrix = self.position_matrix
    cdef FLOAT64_t[::1] beta = self.beta
    cdef EDGES_ID_DTYPE_t[::1] edge_blacklist = self.edge_blacklist
    cdef int aminoacid_count = self.aminoacid_count
    cdef int substring_length = self.substring_length
    cdef int peptide_length = self.peptide_length
    
    cdef INT_AMINO_ACID_DTYPE_t[:,::1] A
    cdef PATH_LENGTH_DTYPE_t[::1] A_length
    cdef INT_AMINO_ACID_DTYPE_t[:,::1] B
    cdef PATH_LENGTH_DTYPE_t[::1] B_length
    
    cdef int B_index, i, j, l, edge_blacklist_count, deviation
    cdef dict lawler_dict
    cdef EDGES_ID_DTYPE_t edge_id
    cdef INT64_t[::1] s
    cdef INT_AMINO_ACID_DTYPE_t[::1] path
    cdef INT_AMINO_ACID_DTYPE_t[::1] spur_path
    cdef PATH_LENGTH_DTYPE_t path_length
    cdef INT_AMINO_ACID_DTYPE_t[::1] spur_node
    cdef INT_AMINO_ACID_DTYPE_t[::1] root_path
    cdef INT_AMINO_ACID_DTYPE_t[::1] total_path
    
    if k > aminoacid_count**peptide_length:
        raise Exception("There is only %i^%i peptides of length %i"%(aminoacid_count, peptide_length, peptide_length))
                
    A = np.zeros((k, peptide_length), dtype=INT_AMINO_ACID_DTYPE)
    A_length = np.zeros(k, dtype=PATH_LENGTH_DTYPE) - PATH_LENGTH_DTYPE(np.inf)
    
    # Initialize the heap to store the potential kth longest path
    B = np.zeros((peptide_length * k, peptide_length), dtype=INT_AMINO_ACID_DTYPE)
    B_length = np.zeros(peptide_length * k, dtype=PATH_LENGTH_DTYPE) - PATH_LENGTH_DTYPE(np.inf)
    B_index = 0
    
    # Initialize an array to store blacklisted edges
    edge_blacklist = np.zeros(peptide_length * k, dtype=EDGES_ID_DTYPE)
    edge_blacklist_count = 0
    
    # Eugene Lawler proposed a modification to Yen's algorithm in which duplicates
    # path are not calculated as opposed to the original algorithm where they are
    # calculated and then discarded when they are found to be duplicates.
    lawler_dict = {}
    
    # Determine the shortest path from the source to the sink
    path, path_length = longest_path(training_peptides, training_peptides_length, psi_matrix, position_matrix, beta, edge_blacklist, edge_blacklist_count, aminoacid_count, substring_length, peptide_length)
    A[0] = path
    A_length[0] = path_length
    
    for i in xrange(1, k):
        
        # New way to implement lawler improvement
        deviation = 0
        if i > 1:
            while deviation < peptide_length and A[i-1,deviation] == A[i-2,deviation]:
                deviation += 1

        if i <= 1 or deviation - substring_length < 0:
            # The spur node is the graph source node
            for j in xrange(i):
                path = A[j]     # Previous shortest path
                edge_id = get_edge_id(path[0:substring_length], 0, substring_length, aminoacid_count)
                if not is_blacklisted(edge_id, edge_blacklist, edge_blacklist_count):
                    edge_blacklist[edge_blacklist_count] = edge_id
                    edge_blacklist_count += 1
            
            # Lawler's modification
            if not lawler_dict.has_key((tuple(edge_blacklist[:edge_blacklist_count]),)):
                spur_path, path_length = longest_path(training_peptides, training_peptides_length, psi_matrix, position_matrix, beta, edge_blacklist, edge_blacklist_count, aminoacid_count, substring_length, peptide_length)
                lawler_dict[(tuple(edge_blacklist[:edge_blacklist_count]),)] = True
            
                # If there is a path
                if path_length > -np.Inf:
                    # Add the path to the heap
                    B[B_index] = spur_path
                    B_length[B_index] = path_length
                    B_index += 1
        
        edge_blacklist_count = 0
        
        # The spur node ranges from the first node to the next to last node in the longest path
        #for j in xrange(0, peptide_length - substring_length):
        for j in xrange(max(0, deviation - substring_length), peptide_length - substring_length):
            
            # Spur node is retrieved from the previous i-shortest path, i − 1
            spur_node = A[i-1][j:j+substring_length]
            
            # The sequence of nodes from the source to the spur node of the previous i-longest path
            root_path = A[i-1][0:j+substring_length]
            
            # For all previous shortest path
            for l in xrange(i):
                path = A[l]
                
                if all_equal(root_path, path[0:j+substring_length]):
                    # Remove the links that are part of the previous shortest paths which share the same root path
                    edge_id = get_edge_id(path[j:j+substring_length+1], j, substring_length, aminoacid_count)
                    if not is_blacklisted(edge_id, edge_blacklist, edge_blacklist_count):
                        edge_blacklist[edge_blacklist_count] = edge_id
                        edge_blacklist_count += 1
            
            # Calculate the spur path from the spur node to the sink.
            # Lawler's modification
            try:
                spur_path, spur_path_length = lawler_dict[ (tuple(edge_blacklist[:edge_blacklist_count]), j+1, tuple(spur_node)) ]
            except:
                spur_path, spur_path_length = longest_path(training_peptides, training_peptides_length, psi_matrix, position_matrix, beta, edge_blacklist, edge_blacklist_count, aminoacid_count, substring_length, peptide_length, j+1, spur_node)
                lawler_dict[(tuple(edge_blacklist[:edge_blacklist_count]), j+1, tuple(spur_node))] = (spur_path[:], spur_path_length)
            
            # Entire path is made up of the root path and spur path
            total_path = np.append(root_path[0:j+1], spur_path)
            total_path_length = spur_path_length + length_to_spur_node(root_path, substring_length, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
                           
            # If there is a path
            if total_path_length > -np.Inf:
                # Add the potential i-longest path to the heap
                # Only if the path is not already in B
                path_in_B = False
                for l in xrange(B_index):
                    if all_equal(B[l], total_path):
                        path_in_B = True
                        break
                
                if not path_in_B:
                    B[B_index] = total_path
                    B_length[B_index] = total_path_length
                    B_index += 1
            
            # Add back the edges that were removed from the graph
            # Here clear the edges blacklist
            edge_blacklist_count = 0
        
        # Sort B
        s = np.argsort(B_length[:B_index])
    
        # Add the maximal cost path becomes the i-longest path
        A[i] = B[s[s.shape[0]-i]]
        A_length[i] = B_length[s[s.shape[0]-i]]
        
    return np.array([self.decode_peptides(x) for x in A]), np.array(A_length)

@cython.boundscheck(False)
@cython.wraparound(False)  
cdef EDGES_ID_DTYPE_t get_edge_id(INT_AMINO_ACID_DTYPE_t[::1] edge_sequence, int position, int substring_length, int aminoacid_count):
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
    cdef EDGES_ID_DTYPE_t edge_id
    cdef int edge_length, i
    
    edge_id = 0
    edge_length = edge_sequence.shape[0]
    
    # Only consider the "self.substring_length" right most amino acid
    for i in xrange(substring_length):
        edge_id += edge_sequence[edge_length-i-1] * aminoacid_count**(i)
    
    # This is an edge from the core of the graph
    if edge_length == substring_length + 1:
        # Account for the first amino acid
        edge_id *= aminoacid_count
        edge_id += edge_sequence[0]
        
        # Account for edges leaving the source node
        edge_id += aminoacid_count**substring_length
        
        if position > 0:
            # Account for edges from the core of the graph i.e. the (position -1)-partite of the graph
            edge_id += position * aminoacid_count**(substring_length+1)

    return edge_id

@cython.boundscheck(False)
@cython.wraparound(False)  
cdef PATH_LENGTH_DTYPE_t length_to_spur_node(   INT_AMINO_ACID_DTYPE_t[::1] path,
                                                int substring_length,
                                                INT_AMINO_ACID_DTYPE_t[:,::1] training_peptides,
                                                INT64_t[::1] training_peptides_length,
                                                FLOAT64_t[:,::1] psi_matrix,
                                                FLOAT64_t[:,::1] position_matrix,
                                                FLOAT64_t[::1] beta):
    """
    Compute the length of the path from the source node
    to the spur_node
    """
    cdef PATH_LENGTH_DTYPE_t length
    cdef int n, k
    
    length = 0.0
    n = path.shape[0] - substring_length + 1
    for k in xrange(n):
        length += W_weight(path[k:k+substring_length], k, training_peptides, training_peptides_length, psi_matrix, position_matrix, beta)
    
    return length
        
