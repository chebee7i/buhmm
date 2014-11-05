# encoding: utf-8

from nose.tools import *

from cmpy import machines

from ..canonical import (
    verify_orders, make_canonical, tmatrix_from_canonical, tmatrix
)

def test_verify_orders():
    m = machines.Even()
    no = ['A', 'A']
    so = ['0', '1']
    assert_raises(Exception, verify_orders, m, no, so)
    no = ['A', 'B']
    so = ['0', '2']
    assert_raises(Exception, verify_orders, m, no, so)
    so = ['0', '1']
    verify_orders(m, no, so)

def test_make_canonical():
    m = machines.Even()
    m2 = make_canonical(m)
    assert_equal(set(m2.nodes()), set([0,1]))
    assert_equal(m2.alphabet(), set([0,1]))
    # No need to test for topology, as test_tmatrix() covers us.

def test_tmatrix_from_canonical1():
    # Test using a canonical HMM
    m = machines.Even()
    m2 = make_canonical(m)
    delta = tmatrix_from_canonical(m2).tolist()
    assert_equal( delta, [[0,1],[-1,0]])

def test_tmatrix_from_canonical2():
    # Test using a constructed canonical HMM
    m = machines.MealyHMM()
    m.add_edge(0, 0, o=0, p=1)
    m.add_edge(0, 1, o=1, p=1)
    m.add_edge(1, 0, o=1, p=1)
    delta = tmatrix_from_canonical(m).tolist()
    assert_equal( delta, [[0,1],[-1,0]])

def test_tmatrix():
    m = machines.Even()
    delta, nodes, symbols = tmatrix(m)
    d = delta.tolist()
    assert_equal(d, [[0, 1], [-1, 0]])
    assert_equal(nodes, ['A', 'B'])
    assert_equal(symbols, ['0', '1'])
