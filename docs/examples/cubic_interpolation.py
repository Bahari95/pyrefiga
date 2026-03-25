"""
cubic_interpolation.py

cubic interpolation for a given function in one or two dimensions

Author: M. Bahari
"""
from pyrefiga           import cubic_bsplines
from pyrefiga           import cubic_bspline_interpolation_1D
from pyrefiga           import cubic_bspline_interpolation_2D
# ...
import numpy            as np
from pyrefiga           import pyccel_sol_field_1d
from pyrefiga           import pyccel_sol_field_2d

# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    # uniform grid
    N = 100
    x0, xN = 0.0, 1.0
    xgrid = np.linspace(x0, xN, N+1)
    h = xgrid[1]-xgrid[0]

    # function and derivative at boundaries
    g = np.sin(2*np.pi*xgrid)
    gprime0 = 2*np.pi*np.cos(2*np.pi*xgrid[0])
    gprimeN = 2*np.pi*np.cos(2*np.pi*xgrid[-1])

    # assemble system
    eta, V = cubic_bspline_interpolation_1D(xgrid, g, gprime0, gprimeN, space = True)

    # Evaluate spline at arbitrary points
    xvals = np.linspace(x0, xN, 100)
    cs = pyccel_sol_field_1d(V.knots, eta, mesh = xvals)[0]

    import matplotlib.pyplot as plt
    plt.plot(xvals, cs, label='Cubic spline')
    plt.plot(xgrid, g, 'o', label='Original points')
    plt.legend()
    plt.show()


# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    # uniform grid
    N      = 100
    x0, xN = 0.0, 1.0
    xgrid  = np.linspace(x0, xN, N+1)
    h      = xgrid[1]-xgrid[0]

    X,Y = np.meshgrid(xgrid,xgrid)
    # function and derivative at boundaries
    g       = np.sin(2*np.pi*X.T)*np.sin(2*np.pi*Y.T)
    gprimex = [2*np.pi*np.cos(2*np.pi*xgrid[0])*np.sin(2*np.pi*xgrid), 2*np.pi*np.cos(2*np.pi*xgrid[-1])*np.sin(2*np.pi*xgrid)]
    gprimey = [2*np.pi*np.cos(2*np.pi*xgrid[0])*np.sin(2*np.pi*xgrid), 2*np.pi*np.cos(2*np.pi*xgrid[-1])*np.sin(2*np.pi*xgrid)]

    # assemble system
    eta, Vh = cubic_bspline_interpolation_2D(xgrid, xgrid, g, gprimex, gprimey, space = True)

    S, Sx, Sy, X, Y = pyccel_sol_field_2d((100,100),  eta, Vh.knots, Vh.degree) 

    plt.figure()
    plt.contourf(X, Y, S)
    plt.colorbar()
    plt.title("2D Cubic B-Spline Approximation")
    plt.show()