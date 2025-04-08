#! usr/bin/env python3
# --*- coding: utf-8 -*-
'''Utility functions for the linear regression module.'''
import pandas as pd
import numpy as np
from numpy.typing import NDArray

def _read_data(path: str) -> NDArray:
    """Reads data from a CSV file and returns it as a NumPy array.

    Args:
        path (str): Path to the CSV file.

    Returns:
        NDArray: Data read from the CSV file.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=[df.columns[0], df.columns[1]]).to_numpy()
    x_data = df[:, 0].astype(np.float64)
    y_data = df[:, 1].astype(np.float64)
    return x_data, y_data
