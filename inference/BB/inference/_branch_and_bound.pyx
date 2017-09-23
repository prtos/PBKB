# cython: profile=True
import cython
import heapq
import time

import numpy
cimport numpy

from preimage.inference.node cimport Node
from preimage.inference.node_creator cimport NodeCreator
from preimage.inference.search_stats cimport SearchStatsBuilder

ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int64_t INT64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound(NodeCreator node_creator, int y_length, list alphabet, SearchStatsBuilder stats_builder,
                       INT64_t seed):
    cdef str empty_string = ""
    cdef Node node, best_node
    cdef list heap

    numpy.random.seed(seed)
    heap = create_heap(node_creator, y_length, stats_builder)
    best_node = Node(empty_string, numpy.inf, 0, 0, False)
    while(len(heap) > 0 and (not stats_builder.is_time_or_n_iterations_expired() or best_node.y == empty_string)):
        node = heapq.heappop(heap)
        stats_builder.add_iteration()
        if  best_node < node:
            break
        node = depth_first_search(node, best_node, node_creator, heap, y_length, y_length, alphabet, stats_builder)
        if node < best_node and len(node.y) == y_length:
            best_node = node
            stats_builder.update_solution()
    stats_builder.end()
    return best_node.y, best_node.get_bound()

cdef list create_heap(NodeCreator node_creator, int y_length, SearchStatsBuilder stats_builder):
    cdef list heap
    stats_builder.start()
    heap = node_creator.get_start_nodes(y_length)
    numpy.random.shuffle(heap)
    heapq.heapify(heap)
    stats_builder.end_start_nodes()
    return heap

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound_no_length(NodeCreator node_creator, int min_length, int max_length, list alphabet,
                                 SearchStatsBuilder stats_builder, INT64_t seed):
    cdef str empty_string = ""
    cdef Node best_node
    cdef int key, length
    cdef list heap
    cdef list keys_to_remove
    cdef dict heaps

    numpy.random.seed(seed)
    heaps = get_heap_for_each_length(node_creator, min_length, max_length, stats_builder)
    best_node = Node(empty_string, numpy.inf, 0, 0, False)
    while(len(heaps) > 0 and (not stats_builder.is_time_or_n_iterations_expired() or best_node.y == empty_string)):
        keys_to_remove = []
        for length, heap in heaps.items():
            best_node = find_length_best_node(best_node, node_creator, heap, length, alphabet, keys_to_remove,
                                              stats_builder)
            if stats_builder.is_time_or_n_iterations_expired():
                break
        for key in keys_to_remove:
            heaps.pop(key)
    stats_builder.end()
    return best_node.y, best_node.get_bound()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict get_heap_for_each_length(NodeCreator node_creator, int min_length, int max_length,
                                   SearchStatsBuilder stats_builder):
    cdef list heap
    cdef list keys_to_remove
    cdef dict heaps = {}
    cdef int length

    stats_builder.start()
    for length in range(min_length, max_length + 1):
        heap = node_creator.get_start_nodes(length)
        numpy.random.shuffle(heap)
        heapq.heapify(heap)
        heaps[length] = heap
    stats_builder.end_start_nodes()
    return heaps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node find_length_best_node(Node best_node, NodeCreator node_creator, list heap, int length,
                                   list alphabet, list keys_to_remove, SearchStatsBuilder stats_builder):
    cdef Node node

    if len(heap) == 0:
         keys_to_remove.append(length)
    else:
        node = heapq.heappop(heap)
        stats_builder.add_iteration()
        if best_node < node:
            keys_to_remove.append(length)
        else:
            node = depth_first_search(node, best_node, node_creator, heap, length, length, alphabet, stats_builder)
            if node < best_node and len(node.y) == length:
                best_node = node
                stats_builder.update_solution()
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound_multiple_solutions(NodeCreator node_creator, int y_length, int n_solutions, list alphabet,
                                          SearchStatsBuilder stats_builder, INT64_t seed):
    cdef str empty_string = ""
    cdef Node node, best_node
    cdef list node_heap = node_creator.get_start_nodes(y_length)
    cdef list solution_heap = []
    cdef list solutions, bounds
    cdef int max_depth = y_length - 1

    numpy.random.seed(seed)
    stats_builder.start()
    heapq.heapify(node_heap)
    best_node = Node(empty_string, numpy.inf, 0, 0, False)
    while(len(node_heap) > 0 and (not stats_builder.is_time_or_n_iterations_expired() or best_node.y == empty_string)):
        node = heapq.heappop(node_heap)
        stats_builder.add_iteration()
        if  best_node < node:
            break
        node = depth_first_search(node, best_node, node_creator, node_heap, y_length, max_depth, alphabet,
                                  stats_builder)
        if node < best_node and len(node.y) == max_depth:
            best_node = add_children_to_solution_heap(node, best_node, node_creator, solution_heap, n_solutions,
                                                      y_length, alphabet, stats_builder)
        elif len(node.y) == y_length:
            best_node = add_node_to_solution_heap(node, best_node, solution_heap, n_solutions)
    solutions, bounds = get_sorted_solutions_and_bounds(solution_heap)
    stats_builder.end()
    return solutions, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node depth_first_search(Node node, Node best_node, NodeCreator node_creator, list heap, int y_length,
                             int max_depth, list alphabet, SearchStatsBuilder stats_builder):
    cdef int i
    cdef Node parent_node, child
    cdef str letter, child_y
    for i in range(len(node.y), max_depth):
        stats_builder.add_iteration()
        parent_node = node
        numpy.random.shuffle(alphabet)
        child_y = alphabet[0] + parent_node.y
        node = node_creator.create_node(child_y, parent_node, y_length)
        for letter in alphabet[1:]:
            child_y = letter + parent_node.y
            child = node_creator.create_node(child_y, parent_node, y_length)
            if child < node:
                if node < best_node:
                     heapq.heappush(heap, node)
                node = child
            else:
                if(child < best_node):
                    heapq.heappush(heap, child)
        if best_node < node:
            break
    return node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node add_children_to_solution_heap(Node parent_node, Node best_node, NodeCreator node_creator, list solution_heap,
                                        int n_solutions, int y_length, list alphabet, SearchStatsBuilder stats_builder):
    cdef Node solution_node
    cdef str letter, child_y

    for letter in alphabet:
        stats_builder.add_iteration()
        child_y = letter + parent_node.y
        solution_node = node_creator.create_node(child_y, parent_node, y_length)
        best_node = add_node_to_solution_heap(solution_node, best_node, solution_heap, n_solutions)
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node add_node_to_solution_heap(Node solution_node, Node best_node, list solution_heap, int n_solutions):
    cdef Node new_best_node
    if len(solution_heap) < n_solutions or solution_node < best_node:
        solution_node.invert_bound()
        heapq.heappush(solution_heap, solution_node)
    if len(solution_heap) > n_solutions:
        new_best_node =  heapq.heappop(solution_heap)
        new_best_node.invert_bound()
        best_node = new_best_node
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_sorted_solutions_and_bounds(list solution_heap):
    cdef int i
    cdef list bounds = []
    cdef list solutions = []
    cdef Node node
    for i in range(len(solution_heap)):
        node = heapq.heappop(solution_heap)
        solutions.append(node.y)
        bounds.append(node.get_bound())
    solutions.reverse()
    bounds.reverse()
    return solutions, bounds