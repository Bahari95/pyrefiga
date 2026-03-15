"""
cubic_interpolation.py

cubic interpolation for a given function in one dimension

Author: M. Bahari # TODO pass lambda function as parameter update pyccel
"""
import numpy as np
from scipy.linalg import solve
from pyccel   import epyccel
from pyrefiga import make_knots

# ---------------------------------------------
# 1D cubic B-spline generator (uniform knots)
# ---------------------------------------------
def ders_bspline_grid(ne1:'int', p1:'int', spans_1:'int[:]', basis_1:'float64[:,:,:]', points_1:'float64[:]', knots_1:'float64[:]', nders:'int'):
    # Assemble NURBS basis functions and their derivatives at quadrature points for 1D elements.

    # Parameters
    # ----------
    # ne1 : int
    #     Number of elements in the 1D mesh.
    # p1 : int
    #     Degree of the NURBS basis.
    # spans_1 : int[:]
    #     Output vector to store the knot span index for each element.
    # basis_1 : float64[:,:,:,:]
    #     Output array to store basis functions and their derivatives at quadrature points.
    # points_1 : float64[:,:]
    #     Quadrature points for each element.
    # knots_1 : float64[:]
    #     Knot vector.
    # omega : float64[:]
    #     Weights for NURBS basis functions.
    # nders : int
    #     Number of derivatives to compute.

    # Notes
    # -----
    # This function computes the non-zero basis functions and their derivatives at each quadrature point
    # for all elements in a 1D mesh, supporting NURBS basis.
    # ... sizes
    from numpy import zeros
    from numpy import empty
    #   ---Computes All basis in a new points
    degree         = p1
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
        xq = points_1[ie1]
        #~~~~~~~~~~~~~~~
        # Knot index at left/right boundary
        low  = degree
        high = len(knots_1)-1-degree
        # Check if point is exactly on left/right boundary, or outside domain
        if xq <= knots_1[low ]: 
                span = low
        elif xq >= knots_1[high]: 
                span = high-1
        else : 
            # Perform binary search
            span = (low+high)//2
            while xq < knots_1[span] or xq >= knots_1[span+1]:
                if xq < knots_1[span]:
                    high = span
                else:
                    low  = span
                span = (low+high)//2
        # ... 
        spans_1[ie1] = span
        # ...
        ndu[0,0] = 1.0
        for j in range(0,degree):
            left [j] = xq - knots_1[span-j]
            right[j] = knots_1[span+1+j] - xq
            saved    = 0.0
            for r in range(0,j+1):
                # compute inverse of knot differences and save them into lower triangular part of ndu
                ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                # compute basis functions and save them into upper triangular part of ndu
                temp       = ndu[r,j] * ndu[j+1,r]
                ndu[r,j+1] = saved + right[r] * temp
                saved      = left[j-r] * temp
            ndu[j+1,j+1] = saved	
        
        # Compute derivatives in 2D output array 'ders'
        ders[0,:] = ndu[:,degree]
        for r in range(0,degree+1):
            s1 = 0
            s2 = 1
            a[0,0] = 1.0
            for k in range(1,nders+1):
                d  = 0.0
                rk = r-k
                pk = degree-k
                if r >= k:
                    a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                    d = a[s2,0] * ndu[rk,pk]
                j1 = 1   if (rk  > -1 ) else -rk
                j2 = k-1 if (r-1 <= pk) else degree-r
                for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                if r <= pk:
                    a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                    d += a[s2,k] * ndu[r,pk]
                ders[k,r] = d
                j  = s1
                s1 = s2
                s2 = j
        # ...first compute R1
        # ...
        basis_1[ie1, :, 0] = ders[0,:]
        r = degree
        for i_ders in range(1,nders+1):
            # Multiply derivatives by correct factors
            ders[i_ders,:] = ders[i_ders,:] * r
            basis_1[ie1, :, i_ders] = ders[i_ders,:]
            r = r * (degree-i_ders)
def cubic_Hermit_matrix_grid(ne1:'int', points_1:'float64[:]', knots_1:'float64[:]', matrix:'float64[:,:]'):
    # Assemble NURBS basis functions and their derivatives at quadrature points for 1D elements.

    # Parameters
    # ----------
    # ne1 : int
    #     Number of elements in the 1D mesh.
    # points_1 : float64[:,:]
    #     Quadrature points for each element.
    # knots_1 : float64[:]
    #     Knot vector.
    # matrix : float64[:,:]
    #     Output array to store basis functions and bondary values.

    # Notes
    # -----
    # This function computes the non-zero basis functions and their derivatives at each quadrature point
    # for all elements in a 1D mesh, supporting NURBS basis.
    # ... sizes
    from numpy import zeros
    from numpy import empty
    #   ---Computes All basis in a new points
    degree         = 3
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     0+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
        xq = points_1[ie1]
        #~~~~~~~~~~~~~~~
        # Knot index at left/right boundary
        low  = degree
        high = len(knots_1)-1-degree
        # Check if point is exactly on left/right boundary, or outside domain
        if xq <= knots_1[low ]: 
                span = low
        elif xq >= knots_1[high]: 
                span = high-1
        else : 
            # Perform binary search
            span = (low+high)//2
            while xq < knots_1[span] or xq >= knots_1[span+1]:
                if xq < knots_1[span]:
                    high = span
                else:
                    low  = span
                span = (low+high)//2
        # ...
        ndu[0,0] = 1.0
        for j in range(0,degree):
            left [j] = xq - knots_1[span-j]
            right[j] = knots_1[span+1+j] - xq
            saved    = 0.0
            for r in range(0,j+1):
                # compute inverse of knot differences and save them into lower triangular part of ndu
                ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                # compute basis functions and save them into upper triangular part of ndu
                temp       = ndu[r,j] * ndu[j+1,r]
                ndu[r,j+1] = saved + right[r] * temp
                saved      = left[j-r] * temp
            ndu[j+1,j+1] = saved	
        
        # Compute derivatives in 2D output array 'ders'
        ders[0,:] = ndu[:,degree]
        for r in range(0,degree+1):
            s1 = 0
            s2 = 1
            a[0,0] = 1.0
            for k in range(1,0+1):
                d  = 0.0
                rk = r-k
                pk = degree-k
                if r >= k:
                    a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                    d = a[s2,0] * ndu[rk,pk]
                j1 = 1   if (rk  > -1 ) else -rk
                j2 = k-1 if (r-1 <= pk) else degree-r
                for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                if r <= pk:
                    a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                    d += a[s2,k] * ndu[r,pk]
                ders[k,r] = d
                j  = s1
                s1 = s2
                s2 = j
        # ...first compute R1
        # ...
        matrix[ie1+1, span-degree:span+1] = ders[0,:]
    h = knots_1[degree+1]-knots_1[degree]
    matrix[0,0] = -1/(2*h)    # eta_{N-1}
    matrix[0,2] =  1/(2*h)    # eta_{N-1}
    h = knots_1[degree+1]-knots_1[degree]
    matrix[ne1+1,ne1-1] = 1/(2*h)    # eta_{N-1}
    matrix[ne1+1,ne1+1] = -1/(2*h)    # eta_{N-1}

cubic_bsplines = epyccel(ders_bspline_grid)
cubic_Hmatrix  = epyccel(cubic_Hermit_matrix_grid)


# ---------------------------------------------
# Assemble linear system with derivative BCs
# ---------------------------------------------
def assemble_spline_system(xgrid, g, gprime0, gprimeN):
    """
    Assemble the system A * eta = rhs for cubic spline interpolation
    with derivative boundary conditions.
    """
    N = len(xgrid)-1
    h = xgrid[1]-xgrid[0]
    ncoef = N+3
    
    # Matrix and RHS
    A = np.zeros((ncoef, ncoef))
    rhs = np.zeros(ncoef)
    
    # 1. Left boundary derivative
    # A[0,0] = -1/(2*h)   # eta_-1
    # A[0,2] = 1/(2*h)    # eta_1
    rhs[0] = gprime0
    
    # 2. Interpolation at nodes
    knots = make_knots(xgrid, 3, periodic=False, multiplicity = 1)
    cubic_Hmatrix(N+1, xgrid, knots, A)
    # basis = np.zeros((len(xgrid), 4, 1))# degree +1 = 4
    # spans = np.zeros(len(xgrid), dtype=int)
    # cubic_bsplines(N+1,3, spans, basis, xgrid, knots, 0)
    for i in range(N+1):
        # j = spans[i]
        # A[i+1,j-3:j+1] = basis[i,:,0]
        rhs[i+1] = g[i]
    
    # 3. Right boundary derivative
    # A[-1,-3] = 1/(2*h)    # eta_{N-1}
    # A[-1,-1] = -1/(2*h)   # eta_{N+1}
    rhs[-1] = gprimeN
    
    return A, rhs

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
    A, rhs = assemble_spline_system(xgrid, g, gprime0, gprimeN)

    # solve for spline coefficients
    eta = solve(A, rhs)

    # Evaluate spline at arbitrary points
    xvals = np.linspace(x0, xN, 100)
    knots = make_knots(xgrid, 3, periodic=False, multiplicity = 1)
    basis = np.zeros((len(xvals), 1+3,1))
    spans = np.zeros(len(xvals), dtype=int)
    cubic_bsplines(len(xvals),3, spans, basis, xvals, knots, 0)
    # cs = basis[:,:,0] @ eta
    cs = np.zeros(len(xvals))
    for i, xi in enumerate(xvals):
        s = spans[i]       # knot span index
        cs[i] = basis[i,:,0] @ eta[s-3 : s+1]   # degree = 3
    import matplotlib.pyplot as plt
    plt.plot(xvals, cs, label='Cubic spline')
    plt.plot(xgrid, g, 'o', label='Original points')
    plt.legend()
    plt.show()