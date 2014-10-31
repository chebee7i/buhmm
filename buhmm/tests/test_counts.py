# encoding: utf-8

from nose.tools import *

import numpy as np

from cmpy.inference import standardize_data
from cmpy import machines

from ..canonical import tmatrix
from ..counts import path_counts, out_arrays

def test_path_counts1():
    # Test without state_path
    m = machines.Even()
    delta, nodes, symbols = tmatrix(m)

    prng = np.random.RandomState()
    prng.seed(0)
    d = m.symbols(20, prng=prng)
    d = standardize_data(d)

    counts, final, states = path_counts(delta, d)

    counts_ = [[[4, 8], [0, 8]], [[0, 4], [1, 4]]]
    assert_equal(counts.tolist(), counts_)

    final_ = [0, -1]
    assert_equal(final.tolist(), final_)

    assert_equal(states, None)

def test_path_counts2():
    # Test with node_path
    m = machines.Even()
    delta, nodes, symbols = tmatrix(m)

    prng = np.random.RandomState()
    prng.seed(0)
    d = m.symbols(20, prng=prng)
    d = standardize_data(d)

    counts, final, states = path_counts(delta, d, node_path=True)

    counts_ = [[[4, 8], [0, 8]], [[0, 4], [1, 4]]]
    assert_equal(counts.tolist(), counts_)

    final_ = [0, -1]
    assert_equal(final.tolist(), final_)

    states_ = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert_equal(states.tolist(), states_)

def test_path_counts3():
    # Test with node_path and preallocated arrays
    m = machines.Even()
    delta, nodes, symbols = tmatrix(m)

    prng = np.random.RandomState()
    prng.seed(0)
    d = m.symbols(20, prng=prng)
    d = standardize_data(d)

    counts, final, states = out_arrays(2, 2, 20, node_path=True)
    path_counts(delta, d, node_path=True, out_arrays=(counts, final, states))

    counts_ = [[[4, 8], [0, 8]], [[0, 4], [1, 4]]]
    assert_equal(counts.tolist(), counts_)

    final_ = [0, -1]
    assert_equal(final.tolist(), final_)

    states_ = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert_equal(states.tolist(), states_)



