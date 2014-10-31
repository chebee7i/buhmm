# -*- coding: utf-8 -*-
"""
Functions for making canonical HMMs and transition matrices.

"""
from __future__ import division

import numpy as np

__all__ = [
    'verify_orders', 'make_canonical', 'tmatrix_from_canonical', 'tmatrix'
]

def verify_orders(machine, node_order, symbol_order):
    """
    Helper function to verify node and symbol orders.

    Users can call this before make_canonical(), if they want assurances.

    This requires one pass through all the edges in the machine.

    """
    nodes = set(machine.nodes())
    nodes_ = set(node_order)
    if nodes != nodes_:
        raise Exception("Invalid node order.")

    symbols = machine.alphabet()
    symbols_ = set(symbol_order)
    if symbols != symbols_:
        raise Exception("Invalid symbol order.")

def make_canonical(machine, node_order=None, symbol_order=None):
    """Returns an HMM with canonical node and symbol names.

    Canonical means integer nodes and symbols, starting from 0. Data in the
    transformed machine is shallow copied. So if you need a deep copy, call:

        `make_canonical(machine.copy())`

    Parameters
    ----------
    machine : CMPy machine
        The machine to transform.
    node_order : list
        The desired order for the nodes. If `None`, then the sorted nodes
        determine the new labels.
    symbol_order : list
        The desired order for the symbols. If `None`, then the sorted symbols
        determine the new labels.

    Returns
    -------
    m : CMPy machine
        The canonical form of `machine`.

    Examples
    --------
    >>> m = make_canonical(machines.Even())

    Notes
    -----
    If `node_order` and/or `symbol_order` are not `None`, then they are assumed
    to contain every node and/or symbol. Exceptions will raise otherwise.

    """

    if node_order is None:
        node_order = sorted(machine.nodes())

    if symbol_order is None:
        symbol_order = sorted(machine.alphabet())

    nodes = range(len(node_order))
    symbols = range(len(symbol_order))

    node_map = dict(zip(node_order, nodes))
    symbol_map = dict(zip(symbol_order, symbols))

    m = machine.__class__()
    for u, v, d in machine.edges(data=True):
        new_data = dict(d)
        new_data['output'] = symbol_map[d['output']]
        m.add_edge(node_map[u], node_map[v], **new_data)

    # Add meta information to graph.
    m.graph['nNodes'] = len(node_order)
    m.graph['nSymbols'] = len(symbol_order)

    return m

def tmatrix_from_canonical(machine):
    """
    Returns a NumPy array representing the transition matrix of `machine`.

    This assumes that `machine` is already canonical.

    Parameters
    ----------
    machine : CMPy machine
        A canonical, unifilar CMPy machine, i.e. something which is the
        output of `canonical_machine`.

    Returns
    -------
    delta : NumPy array, shape (n, k)
        A representation of the transition matrix of the machine. There are
        as many rows as there are nodes and as many columns as there are
        symbols. Entries represent the next node. A value of negative one
        means the transition is forbidden.

    Notes
    -----
    Neither unifilarity or canonicity are verified. Results will be incorrect
    if these assumptions are not met.

    Examples
    --------
    >>> m = machines.Even()
    >>> delta = tmatrix_from_canonical(m).tolist()
    [[0, 1], [-1, 0]]

    """
    try:
        shape = (machine.graph['nNodes'], machine.graph['nSymbols'])
    except KeyError:
        # Then the machine is canonical but not created via make_canonical().
        shape = (len(machine), len(machine.alphabet()))

    delta = np.zeros(shape, dtype=int) - 1
    for u, v, data in machine.edges(data=True):
        delta[u, data['output']] = v

    return delta

def tmatrix(machine, node_order=None, symbol_order=None):
    """
    Returns the transition matrix of canonical form of `machine`.

    This canonicalizes the machine while building the transition matrix.

    Parameters
    ----------
    machine : CMPy machine
        The machine to transform.
    node_order : list
        The desired order for the nodes. If `None`, then the sorted nodes
        determine the new labels.
    symbol_order : list
        The desired order for the symbols. If `None`, then the sorted symbols
        determine the new labels.

    Returns
    -------
    delta : NumPy array, shape (n, k)
        A representation of the transition matrix of the machine. There are
        as many rows as there are nodes and as many columns as there are
        symbols. Entries represent the next node. A value of negative one
        means the transition is forbidden.
    node_order : list
        The nodes in row order.
    symbol_order : list
        The symbols in column order.

    Notes
    -----
    Unifilarity is not verified, and the resulting transition matrix will be
    incorrect. Users should verify unifilarity manually, if necessary.

    Examples
    --------
    >>> m = machines.Even()
    >>> delta, nodes, symbols = tmatrix(m)
    >>> delta.tolist()
    [[0, 1], [-1, 0]]
    >>> nodes
    ['A', 'B']
    >>> symbols
    ['0', '1']

    """
    if node_order is None:
        node_order = sorted(machine.nodes())

    if symbol_order is None:
        symbol_order = sorted(machine.alphabet())

    nodes = range(len(node_order))
    symbols = range(len(symbol_order))

    node_map = dict(zip(node_order, nodes))
    symbol_map = dict(zip(symbol_order, symbols))

    shape = (len(nodes), len(symbols))
    delta = np.zeros(shape, dtype=int) - 1
    for u, v, data in machine.edges(data=True):
        delta[node_map[u], symbol_map[data['output']]] = node_map[v]

    return delta, node_order, symbol_order
