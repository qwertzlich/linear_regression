#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module uses gradient descent to perform linear regression on a dataset"""
import argparse
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _cli_defs():
    _description = (
        "This module uses gradient descent to perform linear regression on a dataset"
    )
    parser = argparse.ArgumentParser(description=_description)

    msg = "Path to the input file. The file should contain two columns of data in XY-Format."
    parser.add_argument("-i", "--input", help=msg, required=True)

    msg = "Maximum number of iterations to perfom. Default value is 10000"
    parser.add_argument("-n", "--iterations", help=msg, type=int, default=10000)

    msg = "Step size for the gradient descent. Default value is 0.01"
    parser.add_argument("-s", "--step", help=msg, type=float, default=0.01)

    msg = "Absolute tolerance used as stopping criteria for values of m and b. Default is 1e-6"
    parser.add_argument("-a", "--atol", help=msg, type=float, default=1e-6)

    return parser.parse_args()


def _get_deriv_m(xi: NDArray, yi: NDArray, y_pred: NDArray) -> float:
    """Calculates the partial derivative of MSE regrarding m

    args:
        xi: Independent variable
        yi: Dependent variable
        y: Calculated y values
    returns:
        float: derivative value
    """
    nn = xi.shape[0]
    return np.sum((y_pred - yi) * xi) / nn


def _get_deriv_b(yi: NDArray, y_pred: NDArray) -> float:
    """Calculates the partial derivative of MSE regarding b

    args:
        yi: Dependent variable
        y: Calculated y values
    returns:
        float: derivative value
    """
    nn = yi.shape[0]
    return np.sum(y_pred - yi) / nn


def linreg_grad(
    x: ArrayLike,
    y: ArrayLike,
    step: float = 0.01,
    iterations: int = 10000,
    atol: float = 1e-6,
) -> tuple:
    """
    This function uses gradient descent to perform linear regression on a dataset

    args:
        x: The independent variable
        y: The dependent variable
        step: The step size for the gradient descent
        iterations: The number of iterations to perform
        atol: The absolute tolerance used as stopping criteria for values of m and b
    return:
        tuple: The slope and intercept of the regression line as tuple (a,b) for y = a*x + b
    """
    xi = np.array(x)
    yi = np.array(y)

    mm: float = 1.0
    bb: float = 0.0

    for _ in range(iterations):
        y = mm * xi + bb
        mm_new = mm - step * _get_deriv_m(xi, yi, y)
        bb_new = bb - step * _get_deriv_b(yi, y)

        if np.abs(mm - mm_new) < atol and np.abs(bb - bb_new) < atol:
            break
        mm = mm_new
        bb = bb_new

    return (mm, bb)


if __name__ == "__main__":
    args = _cli_defs()
    x_vals, y_vals = np.loadtxt(args.input, unpack=True)
    slope, intersec = linreg_grad(x_vals, y_vals, args.step, args.iterations, args.atol)

    print(f"The slope is {slope} and the intercept is {intersec}")
