#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#This module contains linreg method to perform linear regression on a dataset

import numpy as np
from numpy.typing import ArrayLike


def linreg(x : ArrayLike, y: ArrayLike) -> tuple :
    '''
    This function performs linear regression on a dataset

    args:
        x: The independent variable
        y: The dependent variable

    return:
        tuple: The slope and intercept of the regression line as tuple (a,b) for y = a*x + b
    '''

    x = np.array(x)
    y = np.array(y)

    cc = np.ones((len(x), 2))
    cc[:, 0] = x # Matrix for LGS cc p = y where p are slope and intercept

    qq, rr = np.linalg.qr(cc) # QR decomposition of cc
    pp = np.linalg.solve(rr, qq.T @ y) # Solve LGS cc p = y

    return tuple(pp)

