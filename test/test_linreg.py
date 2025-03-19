import sys
import os
import numpy as np
import pytest

dir = os.path.dirname(__file__)
sys.path.append(os.path.join(dir, '..'))

from linreg.linreg import linreg

def test_linreg():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    assert np.allclose(linreg(x, y), (1.0, 0.0), rtol=1e-4)


