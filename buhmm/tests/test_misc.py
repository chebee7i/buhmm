from nose.tools import *

import numpy as np

import buhmm.misc as module

def test_logspace_int_smoke():
    # Smoke test
    x = module.logspace_int(2000, 50)
    y = np.array([
          0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   11,
         13,   14,   17,   19,   22,   25,   29,   33,   38,   43,   49,
         56,   65,   74,   84,   96,  110,  125,  143,  164,  187,  213,
        243,  277,  316,  361,  412,  470,  536,  612,  698,  796,  908,
       1035, 1181, 1347, 1537, 1753, 1999])
    np.testing.assert_allclose(x, y)

def test_logspace_int_corner():
    # Corner cases
    x = module.logspace_int(2000, 0)
    np.testing.assert_allclose(x, [])
    x = module.logspace_int(2000, 1)
    np.testing.assert_allclose(x, [0])

    # Must have nonzero limit
    assert_raises(Exception, module.logspace_int, 0)

    # Not enough integers
    assert_raises(Exception, module.logspace_int, 3, 10)
