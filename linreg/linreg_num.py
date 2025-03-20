#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains linreg method to perform linear regression on a dataset"""
import argparse
import numpy as np
from numpy.typing import ArrayLike


def _cli_defs():
    _description = (
        "Use numerical methods to perform linear Regression. For more info see Readme."
    )
    parser = argparse.ArgumentParser(description=_description)
    msg = "Path to the input file. The file should contain two columns of data in XY-Format."
    parser.add_argument("-i", "--input", help=msg, required=True)
    return parser.parse_args()


def linreg_num(x: ArrayLike, y: ArrayLike) -> tuple:
    """
    This function performs linear regression on a dataset

    args:
        x: The independent variable
        y: The dependent variable

    return:
        tuple: The slope and intercept of the regression line as tuple (a,b) for y = a*x + b
    """

    x = np.array(x)
    y = np.array(y)

    cc = np.ones((len(x), 2))
    cc[:, 0] = x  # Matrix for LGS cc p = y where p are slope and intercept

    qq, rr = np.linalg.qr(cc)  # QR decomposition of cc
    pp = np.linalg.solve(rr, qq.T @ y)  # Solve LGS cc p = y

    return tuple(pp)


if __name__ == "__main__":
    args = _cli_defs()
    data = np.loadtxt(args.input)
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    slope, intersect = linreg_num(x_vals, y_vals)

    print(
        f"\n The slope of the regression line is {slope} and the intersect is {intersect} \n"
    )
