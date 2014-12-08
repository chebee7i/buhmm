"""
Smoke tests.

"""
from nose.tools import *
from nose import SkipTest

import dit
import buhmm
import numpy as np

class TestSmoke:

    def setUp(cls):
        global machines
        try:
            from cmpy import machines
        except ImportError:
            raise SkipTest('cmpy not available')

    def test_pm_uhmm(self):
        m = machines.Even()
        d = '011110'
        x = buhmm.Infer(m, d)
        xnew = x.get_updated_prior()

        uhmm1 = x.posterior.pm_uhmm('A')
        uhmm2 = xnew.posterior.pm_uhmm('A')
        np.testing.assert_almost_equal(uhmm1, uhmm2)

    def test_pm_uhmm2(self):
        m = machines.Even()
        d = '1111'
        x = buhmm.Infer(m, d)
        xnew = x.get_updated_prior()

        uhmm1 = x.posterior.pm_uhmm('A')
        uhmm2 = xnew.posterior.pm_uhmm('A')
        np.testing.assert_almost_equal(uhmm1, uhmm2)

    def test_predictive_probability(self):
        m = machines.Even()
        d = '11111'
        x = buhmm.Infer(m, d)

        m2A = x.pm_machine('A')
        fnode = m2A.graph['final_node']
        assert_equal(fnode, 'B')
        w = '0'
        p2A = m2A.probability(w, start=fnode)
        assert_almost_equal(p2A, x.predictive_probability(w, 'A'))
        w = '1'
        p2A = m2A.probability(w, start=fnode)
        assert_almost_equal(p2A, x.predictive_probability(w, 'A'))

        m2B = x.pm_machine('B')
        fnode = m2B.graph['final_node']
        assert_equal(fnode, 'A')
        w = '0'
        p2B = m2B.probability(w, start=fnode)
        assert_almost_equal(p2B, x.predictive_probability(w, 'B'))
        w = '1'
        p2B = m2B.probability(w, start=fnode)
        assert_almost_equal(p2B, x.predictive_probability(w, 'B'))

    def test_infer(self):
        m = machines.Even()
        m.prng.seed(0)
        d = m.symbols(100)
        x = buhmm.Infer(m, d)
        evid = x.log_evidence()
        assert_almost_equal(evid, -68.165400496389665)
        pred = x.predictive_probability('1011')
        assert_almost_equal(pred, -2.0372080704707334)
