# Module for linear Regression

## `linreg(x, y) -> tuple(a,b)`

The linreg method takes two arrays x and y as input und calculates the fit parameters a and b via the ordinary least squares. It solves the linear system of equations $C^T C x = C^T y$ with QR-Decomposition.