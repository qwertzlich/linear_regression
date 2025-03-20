'''Testing routines for the gradient_decent module'''
import sys
import os
import pytest
import numpy as np
from numpy.typing import NDArray

directory = os.path.dirname(__file__)
sys.path.append( os.path.join(directory, '..'))

from linreg.gradient_decent import ml_linreg

@pytest.mark.parametrize('x, y, expected', [
    (np.array([1, 2, 3, 4, 5, 6]), np.array([3, 5, 7, 9, 11, 13]), (2.0, 1.0)),
    (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), (1.0, 0.0)),
    (np.array([-2, -1, 0, 1, 2]), np.array([8.2, 6.6, 5, 3.4, 1.8]), (-1.6, 5.0)),
    (np.array([-2, -1, 0, 1, 2]), np.array([-2, -2.5, -3, -3.5, -4]), (-0.5, -3)),
])
def test_ml_linreg(x: NDArray, y: NDArray, expected: tuple):

    assert np.allclose( ml_linreg(x, y), expected, rtol=0.001)
