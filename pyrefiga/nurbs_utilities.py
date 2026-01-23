"""
This module provides utilities for NURBS-based computations, including solution field evaluation,
prolongation, and Paraview post-processing for multi-patch domains.

@author : M. BAHARI
"""
from   .results_f90     import sol_field_NURBS_2d, sol_field_NURBS_3d, least_square_NURBspline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Prolongate NURBS mapping from VH to Vh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
def prolongate_NURBS_mapping(VH, Vh, w, Cp):
    #.. Prologation by knots insertion matrix
    from   pyrefiga.utilities    import prolongation_matrix

    M  = prolongation_matrix(VH, Vh)
    if Vh.dim == 2 :
        px, py = Cp
        # ... Prolongate the wieghts first
        z  = M.dot(w.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis)

        # ... 
        px = w * px
        py = w * py

        # ...
        Px = (M.dot(px.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis))/z
        Py = (M.dot(py.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis))/z

        return Px, Py, z[:,0], z[0,:]
    else :
        px, py, pz = Cp
        # ... Prolongate the wieghts first
        z  = M.dot(w.reshape(VH.nbasis[0] * VH.nbasis[1]* VH.nbasis[2])).reshape(Vh.nbasis)

        # ... 
        px = w * px
        py = w * py
        pz = w * pz

        # ...
        Px = (M.dot(px.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z
        Py = (M.dot(py.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z
        Pz = (M.dot(pz.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z

        return Px, Py, Pz, z[:,0,0], z[0,:,0], z[0,0,:]


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
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from .bsplines import find_span, basis_funs
    from .  import Poisson

    p, q     = Vh.degree
    U, V     = Vh.knots
    wu, wv   = Vh.omega
    n_u, n_v = Vh.nbasis
    # --- Right-hand side ---
    Q        = np.zeros((n_u *n_v))

    # --- Greville points ---
    u_k = np.array([np.mean(U[i+1:i+p+1]) for i in range(n_u)])
    v_l = np.array([np.mean(V[j+1:j+q+1]) for j in range(n_v)])

    if xmp is None:
      print("Using provided mesh for collocation")
      Q[:]    =  sol[:]
    else:
      sx, sy  = np.meshgrid(u_k, v_l)
      if adxmp is not None:
         #---Compute a image by initial mapping
         sx   = sol_field_NURBS_2d((None, None), adxmp[0], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
         sy   = sol_field_NURBS_2d((None, None), adxmp[1], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
      #---Compute a image by initial mapping
      sx      = sol_field_NURBS_2d((None, None), xmp[0], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
      sy      = sol_field_NURBS_2d((None, None), xmp[1], Vh.omega, Vh.knots, Vh.degree, mesh=(sx, sy))[0]
      Q[:]    =  sol(sx, sy).reshape(n_u*n_v)[:]
    

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

    # --- Weighted basis ---
    Nu_w = Nu * wu[np.newaxis, :]
    Nv_w = Nv * wv[np.newaxis, :]

    denom = (Nu_w @ np.ones(n_u))[:, None] * (Nv_w @ np.ones(n_v))[None, :]

    # Kronecker product
    N_tilde = sp.kron(Nv_w, Nu_w, format="csr")

    denom_flat = denom.ravel(order="C")
    Dinv = sp.diags(1.0 / denom_flat)

    Cmat = Dinv @ N_tilde   # final sparse collocation matrix
    rhs = Q.ravel(order="C")

    # --- Solve square system ---
    sol = sla.spsolve(Cmat, rhs)

    return sol.reshape((n_u, n_v))

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# ... Post processing using Paraview 
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
import numpy                        as     np
import pyvista                      as     pv
from   .spaces                      import TensorSpace
from   .utilities                   import pyref_multipatch, pyref_patch
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsAdMeshMultipatch(nbpts, V, pyrefGeometry, xad, yad, zad = None, adspace = None, solution = None, functions = None, precomputed = None, Analytic = None, filename = "figs/admultipatch_multiblock", plot = True): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   V : TensorSpace 
      Tensor spline space for finite element analysis otherewise for mapping if Vg is None.
   pyrefGeometry: pyref_multipatch or pyref_patch
   xad, yad : list
       Lists of control points for the adaptive mesh in x and y directions.
   zad : list, optional
       List of control points for the adaptive mesh in z direction (for 3D).
   adSpace: TensorSpace space 
      used for the adaptive mapping is defined independently of the solution space V.
   solution : list, optional
       List of solution control points for each patch and its name.
       solutions = [
         {"name": "displacement", "data": xuh = list},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh = list},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   functions : callable, optional
       Analytic function to evaluate on the mesh (signature depends on dimension).
       functions = [
         {"name": "solution", "expression": 'cos(x)+sin(y)'},
         {"name": "rhs", "expression": 'cos(x)+sin(y)' },
         # Add more solution fields as needed
      ]    
   filename : str, optional
       Path to save the output VTM file (default: "figs/admultipatch_multiblock.vtm").
   plot : bool, optional
       If True, enables plotting (not used in this function).
   precomputed:
       if user already computes solution in (nbpts^d) mesh
        precomputed = [
         {"name": "displacement", "data": xuh},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   Analytic = [
   {"name": "Anlytic", "x": '(1.+sx*np.cos(2.*np.pi*sy))**0.5', "y": 'sx*np.sin(2.*np.pi*sy)*0.5'}
   ]
   Returns
   -------
   None
       The function saves the multi-block dataset to the specified output path.
   """
   assert isinstance(pyrefGeometry, (pyref_multipatch, pyref_patch)), \
   "Geomapping must be pyref_multipatch or pyref_patch class"
   #---
   numPaches = pyrefGeometry.nb_patches
   # ...
   assert isinstance(V, TensorSpace)
   # ...
   if adspace is None:
      if zad is None:
         Vrefx = V
         Vrefy = V
      else:
         Vrefx = V
         Vrefy = V     
         Vrefz = V      
   else:
      #... different spaces for adaptive mapping
      if zad is None:
         Vrefx = adspace[0]
         Vrefy = adspace[1]
      else:
         Vrefx = adspace[0]
         Vrefy = adspace[1]      
         Vrefz = adspace[2]      
   if Analytic is not None and V.dim != 2:
      raise TypeError('Not implemented')
   
   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()
   #F3 = [] 
   if pyrefGeometry.geo_dim == 2:
      for i in range(numPaches):
         #... computes adaptive mesh
         sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], Vrefx.omega, Vrefx.knots, Vrefx.degree)[0:3]
         sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], Vrefy.omega, Vrefy.knots, Vrefy.degree)[0:3]
         #---Compute a image by initial mapping
         x, y  = pyrefGeometry.eval(i+1, mesh=(sx, sy))
         [[F1x, F1y], [F2x, F2y]] =pyrefGeometry.gradient(i+1, mesh=(sx, sy))
         #...Compute analytic
         if Analytic is not None:
            A = Analytic[i]
            x = eval(A["x"])
            y = eval(A["y"])
         #...Compute a Jacobian
         #Jf = (F1x*F2y - F1y*F2x)
         Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
         #...
         z = np.zeros_like(x)
         points = np.stack((x, y, z), axis=-1)

         nx, ny = x.shape
         grid = pv.StructuredGrid()
         grid.points = points.reshape(-1, 3)
         grid.dimensions = [nx, ny, 1]

         # Flatten the solution and attach as a scalar field
         grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
         # ... image bu analytic function
         if functions is not None:
            for Funct in functions:
               fnc = eval(Funct["expression"])
               grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
         if solution is not None :
            for sol in solution:
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   else:
      if pyrefGeometry.dim == 2 : # 2D adaptive mesh in 3D mapping
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], Vrefx.omega, Vrefx.knots, Vrefx.degree)[0:3]
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], Vrefy.omega, Vrefy.knots, Vrefy.degree)[0:3]
               #---Compute a image by initial mapping
               x, y, z  = pyrefGeometry.eval(i+1, mesh=(sx, sy))
               [[F1x, F1y],[F2x, F2y],[F3x, F3y]]   = pyrefGeometry.gradient(i+1, mesh=(sx, sy))
               #...Compute dirivatives
               xu = sxx*F1x + syx*F1y
               yu = sxx*F2x + syx*F2y
               zu = sxx*F3x + syx*F3y
               xv = sxy*F1x + syy*F1y
               yv = sxy*F2x + syy*F2y
               zv = sxy*F3x + syy*F3y
               #
               r_u = np.stack((xu, yu, zu), axis=-1)  # shape: (200, 200, 3)
               r_v = np.stack((xv, yv, zv), axis=-1)  # shape: (200, 200, 3)
               n   = np.cross(r_u, r_v)
               n  /= np.linalg.norm(n, axis=-1, keepdims=True)
               pts = np.stack((x, y, z), axis=-1)   # (nx, ny, 3)
               center = pts.reshape(-1,3).mean(axis=0)
               vec = pts - center                   # (nx, ny, 3)
               dot = np.sum(n * vec, axis=-1)      # (nx, ny)
               flip = dot < 0
               n[flip] *= -1
               #n[...,0] = x
               #n[...,1] = y
               #n[...,2] = z
               # --- Light direction (unit vector) ---
               l = np.array([0.2, 0.6, 1.0])
               l /= np.linalg.norm(l)

               # --- Brightness field ---
               I = n[...,0]*l[0] + n[...,1]*l[1] + n[...,2]*l[2]
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]
               # ------------------------------------------------------
               # Flatten the solution and attach as a scalar field
               grid["isophotes"] = I.flatten(order='C')  # or 'F' if needed (check your ordering)
               
               # ... image bu analytic function
               if functions is not None:
                  for Funct in functions:
                     fnc = eval(Funct["expression"])
                     grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               # .... 
               if solution is not None :
                  for sol in solution:
                     U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
               # .... 
               if precomputed is not None :
                  for sol in precomputed:
                     grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
               multiblock[f"patch_{i}"] = grid
      else: #... zad
         for i in range(numPaches):
            #... computes adaptive mesh
            sx, uxx, uxy, uxz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xad[i], Vrefx.omega, Vrefx.knots, Vrefx.degree)[0:4]
            sy, uyx, uyy, uyz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), yad[i], Vrefy.omega, Vrefy.knots, Vrefy.degree)[0:4]
            sz, uzx, uzy, uzz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zad[i], Vrefz.omega, Vrefz.knots, Vrefz.degree)[0:4]
            #---Compute a image by initial mapping
            x, y, z  = pyrefGeometry.eval(i+1, mesh=(sx, sy, sz))
            #...Compute a Jacobian in  i direction
            Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
            # .... 
            points = np.stack((x, y, z), axis=-1)

            nx, ny, nz = x.shape
            grid = pv.StructuredGrid()
            grid.points = points.reshape(-1, 3)
            grid.dimensions = [nx, ny, nz]

            # Flatten the solution and attach as a scalar field
            grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

            # Flatten the solution and attach as a scalar field
            # ... image bu analytic function
            if functions is not None:
               for Funct in functions:
                  fnc = eval(Funct["expression"])
                  grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
            # .... 
            if solution is not None:
               for sol in solution:
                  U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
            # .... 
            if precomputed is not None :
               for sol in precomputed:
                  grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
            multiblock[f"patch_{i}"] = grid

   # Save multiblock dataset
   multiblock.save(filename+".vtm")
   print(f"Saved all patches with solution to {filename}.vtm")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsSolutionMultipatch(nbpts, V, pyrefGeometry, solution = None, functions = None, precomputed = None, Analytic = None, filename = "figs/multipatch_solution", plot = True): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   V : list
      Tensor spline space for finite element analysis otherewise for mapping if Vg is None.
   xmp, ymp : list
       Lists of control points for the initial mapping in x and y directions.
   omega : list
       List of weights in one direction for each patch
   pyrefGeometry: pyref_multipatch or pyref_patch

   solution : list, optional
       List of solution control points for each patch and name.
       solutions = [
         {"name": "displacement", "data": xuh},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   functions : callable, optional
       Analytic function to evaluate on the mesh (signature depends on dimension).
       functions = [
         {"name": "solution", "expression": 'cos(x)+sin(y)'},
         {"name": "rhs", "data": 'cos(x)+sin(y)' },
         # Add more solution fields as needed
      ]   
   filename : str, optional
       Path to save the output VTM file (default: "figs/multipatch_solution.vtm").
   plot : bool, optional
       If True, enables plotting (not used in this function).
   precomputed:
       if user already computes solution in (nbpts^d) mesh
        = [
         {"name": "displacement", "data": xuh},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   Analytic = [
   {"name": "Anlytic", "x": '(1.+sx*np.cos(2.*np.pi*sy))**0.5', "y": 'sx*np.sin(2.*np.pi*sy)*0.5'}
   ]
   Returns
   -------
   None
       The function saves the multi-block dataset to the specified output path.
       
   """
   assert isinstance(pyrefGeometry, (pyref_multipatch, pyref_patch)), \
   "Geomapping must be pyref_multipatch or pyref_patch class"
   #---
   numPaches = pyrefGeometry.nb_patches
   #...
   assert isinstance(V, TensorSpace)

   if Analytic is not None and V.dim != 2:
         raise TypeError('Not implemented')

   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()
   if pyrefGeometry.geo_dim == 2:
      for i in range(numPaches):
         #---Compute a physical domain
         x, y  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts))
         [[F1x, F1y], [F2x, F2y]] = pyrefGeometry.gradient(i+1, nbpts=(nbpts, nbpts))
         #...Compute a Jacobian
         Jf = F1x*F2y - F1y*F2x
         #...Compute analytic
         if Analytic is not None:
            A = Analytic[i]
            x, y = eval(A["x"]), eval(A["y"])
         #...
         z = np.zeros_like(x)
         points = np.stack((x, y, z), axis=-1)

         nx, ny = x.shape
         grid = pv.StructuredGrid()
         grid.points = points.reshape(-1, 3)
         grid.dimensions = [nx, ny, 1]

         # Flatten the solution and attach as a scalar field
         grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
         # ... image bu analytic function
         if functions is not None :
            for Funct in functions:
               fnc = eval(Funct["expression"])
               grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
         if solution is not None :
            for sol in solution:
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   elif pyrefGeometry.dim == 3: #.. z is not none 3D case
      for i in range(numPaches):
         #---Compute a physical domain
         x, y, z  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts, nbpts))
         [[uxx, uxy, uxz],[uyx, uyy, uyz],[uzx, uzy, uzz]]   = pyrefGeometry.gradient(i+1, nbpts=(nbpts, nbpts, nbpts))
         #...Compute a Jacobian
         Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
         # .... 
         points = np.stack((x, y, z), axis=-1)

         nx, ny, nz = x.shape
         grid = pv.StructuredGrid()
         grid.points = points.reshape(-1, 3)
         grid.dimensions = [nx, ny, nz]

         # Flatten the solution and attach as a scalar field
         grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

         # Flatten the solution and attach as a scalar field
         # ... image bu analytic function
         if functions is not None :
            for Funct in functions:
               fnc = eval(Funct["expression"])
               grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
         # .... 
         if solution is not None :
            for sol in solution:
               U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
         # .... 
         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   else: #.. z is not none
      for i in range(numPaches):
         #---Compute a physical domain
         x, y, z  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts))
         [[F1x, F1y],[F2x, F2y],[F3x, F3y]]   = pyrefGeometry.gradient(i+1, nbpts=(nbpts, nbpts))
         #...
         r_u = np.stack((F1x, F2x, F3x), axis=-1)  # shape: (200, 200, 3)
         r_v = np.stack((F1y, F2y, F3y), axis=-1)  # shape: (200, 200, 3)
         n   = np.cross(r_u, r_v)
         n  /= np.linalg.norm(n, axis=-1, keepdims=True)
         pts = np.stack((x, y, z), axis=-1)   # (nx, ny, 3)
         center = pts.reshape(-1,3).mean(axis=0)
         vec = pts - center                   # (nx, ny, 3)
         dot = np.sum(n * vec, axis=-1)      # (nx, ny)
         flip = dot < 0
         n[flip] *= -1
         # --- Light direction (unit vector) ---
         l = np.array([0.2, 0.6, 1.0])
         l /= np.linalg.norm(l)

         # --- Brightness field ---
         I = n[...,0]*l[0] + n[...,1]*l[1] + n[...,2]*l[2]

         # ....
         points = np.stack((x, y, z), axis=-1)

         nx, ny = x.shape
         grid = pv.StructuredGrid()
         grid.points = points.reshape(-1, 3)
         grid.dimensions = [nx, ny, 1]

         # Flatten the solution and attach as a scalar field
         grid["isphotes"] = I.flatten(order='C')  # or 'F' if needed (check your ordering)

         # ... image bu analytic function
         if functions is not None :
            for Funct in functions:
               fnc = eval(Funct["expression"])
               grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
         # ...
         if solution is not None :
            for sol in solution:
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], V.omega, V.knots, V.degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
         # ...
         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   
   # Save multiblock dataset
   multiblock.save(filename+".vtm")
   print(f"Saved all patches with solution to {filename}.vtm")

def ViewGeo(geometry, Nump, nbpts=50, functions = None, plot = True, Analytic = None):
   """
   Example on how one can use nurbs mapping and prolongate it in fine grid
   """

   print('#---: ', geometry, Nump)
   mp  = pyref_multipatch(geometry, Nump)
   print("geom dim = ",mp.geo_dim)
  
   # ... save a solution as .vtm for paraview
   paraview_nurbsSolutionMultipatch(nbpts, mp.space, mp, functions = functions, Analytic=Analytic)      
   #------------------------------------------------------------------------------
   # Show or close plots depending on argument
   if plot :
      import subprocess

      # Load the multipatch VTM
      subprocess.run(["paraview", "figs/multipatch_solution.vtm"])


#--------------------------------------------------------------------------------------------------------------------
# ... Time post-processes TODO
#--------------------------------------------------------------------------------------------------------------------
def paraview_TimeSolutionMultipatch(nbpts, V, xmp, ymp, zmp = None, Vg = None, LStime = None, solution = None, functions = None, precomputed = None, filename = "figs/multipatch_solution", output_pvd_path="figs/solution.pvd"): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   V : list
      Tensor spline space for finite element analysis otherewise for mapping if Vg is None.
   xmp, ymp : list
       Lists of control points for the initial mapping in x and y directions.
   zmp : list, optional
       List of control points for the initial mapping in z direction (for 3D).
   Vg: TensorSpace 
      for mapping is defined independently of the solution space V.
   LStime: liste of times
   solution : list, optional
       List of solution control points for each patch and name.
       solutions = [
         {"name": "displacement", "data": xuh},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   functions : callable, optional
       Analytic function to evaluate on the mesh (signature depends on dimension).
       functions = [
         {"name": "solution", "expression": 'cos(x)+sin(y)'},
         {"name": "rhs", "data": 'cos(x)+sin(y)' },
         # Add more solution fields as needed
      ]   
   filename : str, optional
       Path to save the output VTM file (default: "figs/multipatch_solution.vtm").
   plot : bool, optional
       If True, enables plotting (not used in this function).
   precomputed:
       if user already computes solution in (nbpts^d) mesh
        = [
         {"name": "displacement", "data": xuh},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   Returns
   -------
   None
       The function saves the multi-block dataset to the specified output path.
       
   """
   #---Compute a solution
   numPaches = len(Vg)
   assert isinstance(V, TensorSpace)
   if Vg is None:
      Vg = V
   assert isinstance(Vg, TensorSpace)
   # Handle the case where the user uses a B-spline space and wants to plot using this function
   for j in range(numPaches):
      if Vg.omega is None or all(x is None for x in Vg.omega):
         for i in range(Vg.dim):
            Vg.spaces[i]._omega = np.ones(Vg.nbasis[i])
   # --- Create PVD file header
   with open(output_pvd_path, 'w') as f:
      f.write('<?xml version="1.0"?>\n')
      f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
      f.write('  <Collection>\n')
      if zmp is None:
         if solution is None:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     #...Compute a Jacobian
                     Jf = F1x*F2y - F1y*F2x
                     #...
                     z = np.zeros_like(x)
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     #...Compute a Jacobian
                     Jf = F1x*F2y - F1y*F2x
                     #...
                     z = np.zeros_like(x)
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
         else:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     #...Compute a Jacobian
                     Jf = F1x*F2y - F1y*F2x
                     #...
                     z = np.zeros_like(x)
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
                     for sol in solution:
                        U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     #...Compute a Jacobian
                     Jf = F1x*F2y - F1y*F2x
                     #...
                     z = np.zeros_like(x)
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                     for sol in solution:
                        U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
      elif V.dim == 3: #.. z is not none 3D case
         if solution is None:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, uxx, uxy, uxz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:4]
                     y, uyx, uyy, uyz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:4]
                     z, uzx, uzy, uzz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0:4]
                     #...Compute a Jacobian
                     Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
                     # .... 
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny, nz = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, nz]

                     # Flatten the solution and attach as a scalar field
                     grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     # .... 
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny, nz = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, nz]

                     # Flatten the solution and attach as a scalar field
                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
         else:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     # .... 
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny, nz = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, nz]

                     # Flatten the solution and attach as a scalar field
                     # .... 
                     for sol in solution:
                        U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     # .... 
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny, nz = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, nz]

                     # Flatten the solution and attach as a scalar field
                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
                     # .... 
                     for sol in solution:
                        U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
      else: #.. z is not none
         if solution is None:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     z, F3x, F3y = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0:3]
                     #...Compute a Jacobian
                     Jf = F1x*F2y - F1y*F2x
                     #...
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     #...
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)

                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
         else:
            if functions is None:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     #...
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     for sol in solution:
                        U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
                     if precomputed is not None :
                           for sol in precomputed:
                              grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one .vtm per time step
                  vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
                  multiblock.save(vtm_filename)


                  rel_path = os.path.basename(vtm_filename)
                  f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
            else:
               for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
                  multiblock = pv.MultiBlock()
                  for i in range(numPaches):
                     #---Compute a physical domain
                     x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], Vg.omega, Vg.knots, Vg.degree)[0]
                     #...
                     points = np.stack((x, y, z), axis=-1)

                     nx, ny = x.shape
                     grid = pv.StructuredGrid()
                     grid.points = points.reshape(-1, 3)
                     grid.dimensions = [nx, ny, 1]

                     # Flatten the solution and attach as a scalar field
                     # ... image bu analytic function
                     t = LStime[t_ix]
                     for Funct in functions:
                        fnc = eval(Funct["expression"])
                        grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
                     for sol in solution:
                        U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], V.omega, V.knots, V.degree)[0]
                        grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                     if precomputed is not None :
                        for sol in precomputed:
                           grid[sol["name"]]  = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
                     multiblock[f"patch_{i}"] = grid
                  # --- Save one VTM file per time step
                  filename_t = f"{filename}_t{t_ix:03d}.vtm"   
                  multiblock.save(filename_t)
      # --- Close PVD file
      f.write('  </Collection>\n')
      f.write('</VTKFile>\n')
   # Save multiblock dataset
   print(f"Saved all patches with solution to {filename}.vtm")
   print(f"PVD time-series file written to {output_pvd_path}")
   print("👉 Open this .pvd file in ParaView to view the animation.")