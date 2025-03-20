'''Testing routines for the linreg_num module'''
import sys
import os
import numpy as np
import pytest

directory = os.path.dirname(__file__)
sys.path.append(os.path.join(directory, '..'))

from linreg.linreg_num import linreg_num

def test_linreg():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    assert np.allclose(linreg_num(x, y), (1.0, 0.0), rtol=1e-4)
