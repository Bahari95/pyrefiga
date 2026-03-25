#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes L2 projection of 1D function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def least_square_NURBspline(degree, knots, omega, f):
    """
    Computes the least squares projection of a 1D function or vector onto a NURB-spline basis.
    """
    from numpy     import zeros, linspace
    from .bsplines import find_span
    from .bsplines import basis_funs
    from scipy.sparse import csc_matrix, linalg as sla
    
    n       = len(knots) - degree - 1

    if callable(f):
        # ... in the case where f is a function
        m       = n + degree + 100
        u_k     = linspace(knots[0], knots[degree+n], m)
        # ...
        Pc      = zeros(n)
        Q       = zeros(m)
        for i in range(0,m):
            Q[i] = f(u_k[i])
    else:
        # .. in the case of f is a vector
        m       = len(f)
        u_k     = linspace(knots[0], knots[degree+n], m)
        Pc      = zeros(n)
        Q       = zeros(m)
        Q[:]    = f[:]
    
    Pc[0]   = Q[0]
    Pc[n-1] = Q[m-1]  
    #... Assembles matrix N of non vanishing basis functions in each u_k value
    N       = zeros((m-2,n))
    #... Right hand side of least square Approximation
    R       = zeros(m-2)
    R[:]    = Q[1:-1]
    for k in range(1, m-1):
       span                           = find_span( knots, degree, u_k[k] )
       b                              = basis_funs( knots, degree, u_k[k], span )*omega[span-degree:span+1]
       b                             /= sum(b)
       N[k-1,span-degree:span+1]     = b[:]
      #  if span-degree ==0 :
      #     N[k-1,span-degree:span]     = b[1:]
      #     R[k-1]      -= b[0]*Q[0]
      #  elif span+1 == n :
      #     N[k-1,span-degree-1:span-1] = b[:-1]
      #     R[k-1]      -= b[degree]*Q[-1]
      #  else :
      #     N[k-1,span-degree-1:span]   = b
    R -= N[:,0] *Pc[0]     # left
    R -= N[:,-1]*Pc[-1]    # right
    N = N[:,1:-1]
    #... Solve least squares system
    R      = N.T.dot(R)
    M      = (N.T).dot(N)
    #print(N,'\n M = ',M)
    lu       = sla.splu(csc_matrix(M))
    Pc[1:-1] = lu.solve(R)    
    return Pc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes L2 projection of 1D function
def least_square_Bspline(degree, knots, f):
    """
    Computes the least squares projection of a 1D function or vector onto a B-spline basis.
    """
    from numpy     import zeros, linspace
    from .bsplines import find_span
    from .bsplines import basis_funs
    from scipy.sparse import csc_matrix, linalg as sla
    
    n       = len(knots) - degree - 1

    if callable(f):
        # ... in the case where f is a function
        m       = n + degree + 100
        u_k     = linspace(knots[0], knots[degree+n], m)
        # ...
        Pc      = zeros(n)
        Q       = zeros(m)
        for i in range(0,m):
            Q[i] = f(u_k[i])
    else:
        # .. in the case of f is a vector
        m       = len(f)
        u_k     = linspace(knots[0], knots[degree+n], m)
        Pc      = zeros(n)
        Q       = zeros(m)
        Q[:]    = f[:]
    
    Pc[0]   = Q[0]
    Pc[n-1] = Q[m-1]  
    #... Assembles matrix N of non vanishing basis functions in each u_k value
    N       = zeros((m-2,n-2))
    for k in range(1, m-1):
       span                           = find_span( knots, degree, u_k[k] )
       b                              = basis_funs( knots, degree, u_k[k], span )
       if span-degree ==0 :
          N[k-1,span-degree:span]     = b[1:]
       elif span+1 == n :
          N[k-1,span-degree-1:span-1] = b[:-1]
       else :
          N[k-1,span-degree-1:span]   = b

    #... Right hand side of least square Approximation
    R       = zeros(m-2)
    for k in range(1,m-1) : 
       span            = find_span( knots, degree, u_k[k] )
       b               = basis_funs( knots, degree, u_k[k], span )
       R[k-1] = Q[k]
       if span - degree == 0 :
          R[k-1]      -= b[0]*Q[0]
       if span + 1 == n :
          R[k-1]      -= b[degree]*Q[m-1]
    R      = N.T.dot(R)
    M      = (N.T).dot(N)
    #print(N,'\n M = ',M)
    lu       = sla.splu(csc_matrix(M))
    Pc[1:-1] = lu.solve(R)    
    return Pc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes L2 projection of 2D function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def least_square_2dNURBspline(p, q, U, V, omega_u, omega_v, Q):
    """
    Computes least squares projection of f(u,v) onto a 2D NURBS surface basis.
    
    Parameters
    ----------
    p, q   : degrees in u and v
    U, V   : knot vectors
    omega  : weights (2D array shaped [n_u, n_v])
    f      : function f(u,v) or sampled data on a grid
    
    Returns
    -------
    Pc     : control point coefficients (n_u x n_v)
    """
    import numpy as np
    from scipy.sparse import csc_matrix, linalg as sla
    from .bsplines import find_span, basis_funs

    n_u = len(U) - p - 1
    n_v = len(V) - q - 1

    # Sampling points (dense grid in parameter domain)
    m_u, m_v = Q.shape 
    u_k = np.linspace(U[0], U[-1], m_u)
    v_l = np.linspace(V[0], V[-1], m_v)


    # Build collocation matrix
    N_u = np.zeros((m_u-2, n_u))
    N_v = np.zeros((m_v-2, n_v))
    R   = np.zeros((m_u-2, m_v-2))
    Pc  = np.zeros((n_u  , n_v  ))
    Pc[0,:]  = least_square_NURBspline(q, V, omega_v, Q[0,:])
    Pc[-1,:] = least_square_NURBspline(q, V, omega_v, Q[-1,:])
    Pc[:,0]  = least_square_NURBspline(p, U, omega_u, Q[:,0])
    Pc[:,-1] = least_square_NURBspline(p, U, omega_u, Q[:,-1])

    for k in range(1, m_u-1):
       span                           = find_span( U, p, u_k[k] )
       b                              = basis_funs( U, p, u_k[k], span )*omega_u[span-p:span+1]
       b                             /= sum(b)
       N_u[k-1,span-p:span+1] = b[:]
    for k in range(1, m_v-1):
       span                           = find_span( V, q, v_l[k] )
       b                              = basis_funs( V, q, v_l[k], span )*omega_v[span-q:span+1]
       b                             /= sum(b)
       N_v[k-1,span-q:span+1] = b[:]
    R[:,:] = Q[1:-1,1:-1]
   # Example: subtract left strip contribution
   # strips
    R -= np.outer(N_u[:,0],   N_v @ Pc[0,:])     # left
    R -= np.outer(N_u[:,-1],  N_v @ Pc[-1,:])    # right
    R -= np.outer(N_u @ Pc[:,0],   N_v[:,0])     # bottom
    R -= np.outer(N_u @ Pc[:,-1],  N_v[:,-1])    # top 

    # corners (add back once)
    R += Pc[0,0]       * np.outer(N_u[:,0],  N_v[:,0])   # bottom-left
    R += Pc[0,-1]      * np.outer(N_u[:,0],  N_v[:,-1])  # top-left
    R += Pc[-1,0]      * np.outer(N_u[:,-1], N_v[:,0])   # bottom-right
    R += Pc[-1,-1]     * np.outer(N_u[:,-1], N_v[:,-1])  # top-right

    # Solve least squares system
    N_u = N_u[:,1:-1]
    N_v = N_v[:,1:-1]
    R   = R.reshape((m_u-2)*(m_v-2))
    N   = np.kron(N_v, N_u)
    M   = N.T.dot(N)
    rhs = N.T.dot(R)
    P   = sla.cg(M,rhs,rtol = 1e-16)[0]
    Pc[1:-1,1:-1] = P.reshape(n_u-2, n_v-2)
    return Pc

def collocation_2dNURBspline(Vh, sol, xmp = None, adxmp = None):
    """
    Collocation of sampled data sol(u,v) onto a 2D NURBS surface basis
    with separable weights omega[i,j] = wu[i] * wV[j].

    Parameters
    ----------
    p, q   : int
        Degrees in u and v.
    U, V   : 1D arrays
        Knot vectors (open, non-decreasing).
    wu, wv : 1D arrays
        Weights in u (length n_u) and v (length n_v).
        Full weight matrix is outer product: omega[i,j] = wu[i] * wV[i].
    Q      : 2D array (n_u, n_v)
        Sampled data at Greville points.

    Returns
    -------
    Pc : 2D array (n_u, n_v)
        Control coefficients.
    """
    import numpy as np
    from .bsplines import find_span, basis_funs, greville
    from .results_f90 import sol_field_NURBS_2d

    p, q     = Vh.degree
    U, V     = Vh.knots
    wu, wv   = Vh.omega
    n_u, n_v = Vh.nbasis

    # --- Greville points ---
    u_k = greville(Vh.knots[0], Vh.degree[0], Vh.spaces[0].periodic)# np.array([np.mean(U[i+1:i+p+1]) for i in range(n_u)])
    v_l = greville(Vh.knots[1], Vh.degree[1], Vh.spaces[1].periodic)#np.array([np.mean(V[j+1:j+q+1]) for j in range(n_v)])

    if xmp is None:
      Q = np.asarray(sol).reshape(n_u, n_v)
    else:
      sx, sy  = np.meshgrid(u_k, v_l, indexing="ij")
      if adxmp is not None:
         #---Compute a image by initial mapping
         sx   = sol_field_NURBS_2d((None, None), adxmp[0], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
         sy   = sol_field_NURBS_2d((None, None), adxmp[1], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
      #---Compute a image by initial mapping
      sx, sy  = xmp.eval(mesh = (sx, sy))
      Q       = np.asarray(sol(sx, sy)).reshape(n_u, n_v)
    

    # --- 1D collocation matrices (non-rational) ---
    Nu = np.zeros((n_u, n_u))
    Nv = np.zeros((n_v, n_v))

    for ku, u in enumerate(u_k):
        span = find_span(U, p, u)
        vals = basis_funs(U, p, u, span)
        Nu[ku, span - p: span + 1] = vals

    for kv, v in enumerate(v_l):
        span = find_span(V, q, v)
        vals = basis_funs(V, q, v, span)
        Nv[kv, span - q: span + 1] = vals

    # --- Rational 1D collocation matrices ---
    Nu_w = Nu * wu[np.newaxis, :]
    Nv_w = Nv * wv[np.newaxis, :]
    Ru = Nu_w / Nu_w.sum(axis=1, keepdims=True)
    Rv = Nv_w / Nv_w.sum(axis=1, keepdims=True)

    # Solve tensor-product system:
    # Q = Ru @ P @ Rv.T
    T = np.linalg.solve(Ru, Q)
    P = np.linalg.solve(Rv, T.T).T

    return P

# ---------------------------------------------
# Assemble linear system with derivative BCs
# ---------------------------------------------
from   .                import results_f90_core as core
from  functools         import partial    
from  .linalg           import StencilMatrix
from  .spaces           import SplineSpace,TensorSpace
# ...
import numpy            as np
from   scipy.sparse     import csc_matrix, linalg as sla
from   functools        import partial    
# ...
def assemble_cubmatrix(core, V, ne, points,  out = None):
    if out is None:
        out = StencilMatrix(V.vector_space, V.vector_space)        
    # ...
    args = []
    args += [ne]
    args += [points]
    args += [V.knots]
    core( *args, out._data )
    return out
cubic_bsplines = core.ders_bspline_grid
cubic_Hmatrix  = partial(assemble_cubmatrix, core.cubic_Hermit_matrix_grid )
# ---------------------------------------------
# 1D case
# ---------------------------------------------
def cubic_bspline_interpolation_1D(xgrid, g, gprime0, gprimeN, space = False):
    """
    Assemble and solve the system A * eta = rhs for cubic spline interpolation
    with derivative boundary conditions.
    ----
    g : evaluation at xgrid X ygrid
    gprime0 : direvative at boundary x = x0
    gprimeN : direvative at boundary x = xN 
    space   : if True we return sol and space
    ----
    return 
    control point of cubic B-spline function interpolate g
    """
    # ...
    N       = len(xgrid)-1
    ncoef   = N+3
    #.. build B-spline space from xgrid
    V       = SplineSpace(degree = 3, grid=xgrid)
    # Matrix and RHS
    A       = StencilMatrix(V.vector_space, V.vector_space)
    rhs     = np.zeros(ncoef)
    
    # 1. Left boundary derivative
    rhs[0]  = gprime0    
    rhs[-1] = gprimeN
    # 2. Interpolation at nodes
    cubic_Hmatrix(V, N+1, xgrid, A)
    rhs[1:N+1] = g[0:N]    

    # solve for spline coefficients
    lu     = sla.splu(csc_matrix(A.tosparse()))
    eta    = lu.solve(rhs)
    if space:
       return eta, V
    return eta

# ---------------------------------------------
# 2D
# ---------------------------------------------
def cubic_bspline_interpolation_2D(xgrid, ygrid, g, gprimex, gprimey, space = False):
    '''
    Assemble and solve the system A * eta = rhs for cubic spline interpolation
    with derivative boundary conditions.
    ----
    g : evaluation at xgrid X ygrid
    gprimex0,gprimexN  = gprimex 
    gprimey0, gprimeyN = gprimey 
    where
    gprimex0 : direvative at boundary x = x0
    gprimexN : direvative at boundary x = xN 
    gprimey0 : direvative at boundary y = y0 
    gprimeyN : direvative at boundary y = yN
    ----
    return 
    control point of cubic B-spline function interpolate g
    '''
    gprimex0,gprimexN  = gprimex 
    gprimey0, gprimeyN = gprimey 
    # ...
    Nx = len(xgrid)-1
    # ...
    Ny = len(xgrid)-1
    ncoefx = Nx+3
    ncoefy = Ny+3
    
    # Matrix and RHS
    rhs = np.zeros((ncoefx,ncoefy))
        
    #.. build B-spline space from xgrid
    Vx = SplineSpace(degree = 3, grid=xgrid)
    Ax = StencilMatrix(Vx.vector_space, Vx.vector_space)
    cubic_Hmatrix(Vx, Nx+1, xgrid, Ax)
    #.. build B-spline space from ygrid
    Vy  = SplineSpace(degree = 3, grid=ygrid)
    Ay = StencilMatrix(Vy.vector_space, Vy.vector_space)
    cubic_Hmatrix(Vy, Ny+1, ygrid, Ay)
    # ... rhs
    rhs[0, 1:-1] = gprimex0[0]
    rhs[-1,1:-1] = gprimexN[1]
    rhs[1:-1, 0] = gprimey0[0]
    rhs[1:-1,-1] = gprimeyN[1]
    rhs[1:-1,1:-1] = g[:,:]

    # reshape not needed if already matrix
    # Solve Ax * Z = rhs
    lu     = sla.splu(csc_matrix(Ax.tosparse()))
    lv     = sla.splu(csc_matrix(Ay.tosparse()))
    # Z = np.linalg.solve(Ax.toarray(), rhs)
    Z = lu.solve(rhs)
    # Solve Ay * eta^T = Z^T
    # eta = np.linalg.solve(Ay.toarray(), Z.T).T
    eta    = lv.solve(Z.T).T
    if space:
       return eta, TensorSpace(Vx, Vy)
    return eta