# encoding: utf-8
# cython: profile=False
# cython: embedsignature=True
"""
Path counts for canonical hidden Markov models.

"""
from __future__ import absolute_import
from __future__ import division

import cython
cimport cython

import numpy as np
cimport numpy as np

BTYPE = np.bool
ctypedef bint BTYPE_t

ITYPE = np.int64
ctypedef np.int64_t ITYPE_t

__all__ = ['out_arrays', 'path_counts']

def out_arrays(n, k, L, node_path=False):
    """
    Prepares and returns output arrays for `path_counts`.

    Parameters
    ----------
    n : int
        The number of nodes.
    k : int
        The number of symbols in the alphabet.
    L : int
        The number of observations, from which counts are tabulated.
    node_path : bool
        If `True`, prepare an array to hold node paths. If `False`, then
        `node_paths` will be `None`.

    Returns
    -------
    counts : NumPy int array, shape (n, n, k)
        Array for the number of times each edge was traversed.
    final : NumPy int array, shape (n,)
        Array for the final node visitations.
    node_paths : None | NumPy int array, shape (n, L+1)
        Array for node paths.

    """
    counts = np.zeros((n,n,k), dtype=int)
    final = np.zeros(n, dtype=int)
    node_paths = None
    if node_path:
        node_paths = np.zeros((n,L+1), dtype=int)
    return counts, final, node_paths

@cython.boundscheck(False)
@cython.wraparound(False)
def path_counts(np.ndarray[ITYPE_t, ndim=2, mode="c"] tmatrix,
                np.ndarray[ITYPE_t, ndim=1, mode="c"] data,
                BTYPE_t node_path=False,
                out_arrays=None):
    """
    Calculates edge counts, final nodes, and node paths from data.

    Parameters
    ----------
    tmatrix : NumPy int array, shape (n, k)
        The transition matrix of a canonical hidden Markov model. Rows
        correspond to nodes, columns to symbols, and entries correspond to
        the next node. If a transition is not possible, then the entry
        should be -1.
    data : NumPy int array, shape (L,)
        The data points.  The minimum value in the array should be 0, while
        the maximum value in the array should be `k - 1`, where `k` is the
        number of columns in `tmatrix`.
    node_path : bool
        Boolean specifying whether node paths should be stored for each
        possible initial node.
    out_arrays : tuple or None
        If a tuple, then there should be 3 elements, to be used for `counts`,
        `final`, and `node_paths` in the output. Using this parameter will
        reduce the number of memory allocations when doing inference serially
        on a large number of hidden Markov models, all of which have the same
        number of nodes and symbols, and if `node_path` is `True`, also
        the same amount of data. If `node_path` is `False`, then the 3rd
        element can be `None`.

    Returns
    -------
    counts : NumPy int array, shape (n, n, k)
        The number of times each edge was traversed. For both nonedges and
        edges not visited, the count will be zero. Thus, `counts` cannot and
        should not be used to determine the transition structure. First axis
        represents the initial node, second axis the current node, and third
        axis the current symbol.
    final : NumPy int array, shape (n,)
        Entries represent the final node visited, where the index of the entry
        specifies the initial node. If it is not possible to observe the data
        from a particular node, then the value will be -1.
    node_paths : None | NumPy int array, shape (n, L+1)
        If `node_path` is `True`, then this is an integer array of the nodes
        visited from each initial node. If a particular initial node ends up
        not being valid, then the entries are valid only up to the first
        element that is -1. If `node_path` is `False`, then this is `None`.

    Examples
    --------
    >>> m = machines.Even()
    >>> delta, nodes, alphabet = tmatrix(m)
    >>> data = standardize_data(m.symbols(100), alphabet)
    >>> counts, last, node_paths = path_counts(delta, data)

    """
    cdef:
        int initialNode, currentNode, symbol, i, n, k, L
        np.ndarray[ITYPE_t, ndim=3, mode="c"] counts
        np.ndarray[ITYPE_t, ndim=1, mode="c"] final
        np.ndarray[ITYPE_t, ndim=2, mode="c"] node_paths

    n = tmatrix.shape[0]
    k = tmatrix.shape[1]
    L = data.shape[0]

    # Initialize arrays
    if out_arrays is not None:
        # Do not modify the values in counts in case we are updating counts.
        counts = out_arrays[0]
        final = out_arrays[1]
        node_paths = out_arrays[2]
    else:
        counts = np.zeros((n,n,k), dtype=int)
        final = np.zeros(n, dtype=int)
        node_paths = None
        if node_path:
            node_paths = np.zeros((n,L+1), dtype=int)

    for initialNode in range(n):

        currentNode = initialNode

        if node_path:
            # Set the first node in the node path, before seeing any symbols.
            node_paths[initialNode, 0] = currentNode

        for i in range(L):
            symbol = data[i]
            counts[initialNode, currentNode, symbol] += 1
            currentNode = tmatrix[currentNode, symbol]

            if node_path:
                node_paths[initialNode, i+1] = currentNode

            if currentNode == -1:
                # Transition is forbidden.
                break

        # Mark final node, possibly equal to -1.
        final[initialNode] = currentNode

    return counts, final, node_paths
