"""
results_f90.py: A fast post-processing script for visualizing the solution and its derivatives using B-spline or Nurbs functions.

Author: M. Mustapha Bahari TODO DELETE
"""

from   .               import results_f90_core as core
from   .               import nurbs_utilities_core as nurbs_core

#==============================================================
from numpy import zeros, linspace, meshgrid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In one dimension
def sol_field_NURBS_1d(knots, omega, uh, Npoints = None, mesh = None, bound_val = None):
   """
   Computes the solution and its gradient in one dimension.
   """
   Tu = knots
   nu = uh.shape[0]
   pu = len(Tu) - nu -1
   
   if mesh is None:

      if Npoints is None:
         nx     = nu-pu+1
         mesh = Tu[pu:-pu] 
      else :
         '''
         x0_v  : min val in x direction
         x1_v  : max val in x direction
         '''
         nx   = Npoints
         if bound_val is not None:
            x0_v = bound_val[0]
            x1_v = bound_val[1] 

         else :
            x0_v = Tu[pu]
            x1_v = Tu[-pu-1]
         # ...
         mesh  = linspace(x0_v, x1_v, nx)
   # ...
   nx       = mesh.shape[0]
   Q        = zeros((nx, 3))
   Q[:,2]   = mesh[:]
   nurbs_core.sol_field_1D_meshes(nx, uh, Tu, pu, omega, Q)
   return Q[:,0], Q[:,1]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_NURBS_2d( Npoints, uh, omega, knots, degree, mesh = None, bound_val = None):
    '''
    Using computed control points uh we compute solution
    in new discretisation by Npoints
    The solution can be determined within the provided mesh
    The solution can be calculated within a particular domain : bound_val    
    '''
    pu, pv = degree
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    
    if mesh is None:

        if Npoints is None:

            nx = nu-pu+1
            ny = nv-pv+1
        
            xs = Tu[pu:-pu] 
            ys = Tv[pv:-pv] 
            
        else :
            '''
            x0_v  : min val in x direction
            x1_v  : max val in x direction
            y0_v  : min val in y direction
            y1_v  : max val in y direction
            '''
            nx, ny  = Npoints
            if bound_val is not None:

                x0_v = bound_val[0]
                x1_v = bound_val[1] 
                y0_v = bound_val[2] 
                y1_v = bound_val[3]

            else :
                x0_v = Tu[pu]
                x1_v = Tu[-pu-1]
                y0_v = Tv[pv]
                y1_v = Tv[-pv-1]
        # ...
        xs                     = linspace(x0_v, x1_v, nx)
        ys                     = linspace(y0_v, y1_v, ny)
        # ...
        w1, w2 = omega
        Q      = zeros((nx, ny, 3)) 
        nurbs_core.sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, w1, w2, Q)
        # ...
        X, Y   = meshgrid(xs, ys)
        return Q[:,:,0], Q[:,:,1], Q[:,:,2], X.T, Y.T
    else :
       w1, w2 = omega
       # ...
       nx, ny   = mesh[0].shape
       Q        = zeros((nx, ny, 5))
       Q[:,:,3] = mesh[0][:,:]
       Q[:,:,4] = mesh[1][:,:] 
       nurbs_core.sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, w1, w2, Q)       
       return Q[:,:,0], Q[:,:,1], Q[:,:,2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In three dimension
def sol_field_NURBS_3d(Npoints,  uh, omega, knots, degree, mesh = None):
    '''
    Compute solution in new discretisation by Npoints using  control points uh
    '''

    pu, pv, pz = degree
    Tu, Tv, Tz = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    nz = len(Tz) - pz - 1    
    if mesh is None:
       if Npoints is None:

            nx = nu-pu+1
            ny = nv-pv+1
            nz = nz-pz+1
    
            xs = Tu[pu:-pu] #linspace(Tu[pu], Tu[-pu-1], nx)
    
            ys = Tv[pv:-pv] #linspace(Tv[pv], Tv[-pv-1], ny)
      
            zs = Tz[pz:-pz] #linspace(Tv[pv], Tv[-pv-1], ny)
      
       else :
            nx, ny, nz = Npoints

            xs = linspace(Tu[pu], Tu[-pu-1], nx)
          
            ys = linspace(Tv[pv], Tv[-pv-1], ny)
       
            zs = linspace(Tz[pz], Tz[-pz-1], nz)
       # ...
       w1, w2, w3 = omega
       Q    = zeros((nx, ny, nz, 7)) 
       nurbs_core.sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tz, pu, pv, pz, w1, w2, w3, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3], Q[:,:,:,4], Q[:,:,:,5], Q[:,:,:,6],
    else :
       nx, ny, nz = Npoints
       # ...
       w1, w2, w3 = omega
       # ...
       Q          = zeros((nx, ny, nz, 7))
       Q[:,:,:,4] = mesh[0][:,:,:]
       Q[:,:,:,5] = mesh[1][:,:,:]
       Q[:,:,:,6] = mesh[2][:,:,:]
       nurbs_core.sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tz, pu, pv, pz, w1, w2, w3, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3]    
    
# Computes Solution and its gradien In one dimension
def pyccel_sol_field_1d(knots, uh, Npoints = None, mesh = None, bound_val = None):
   """
   Computes the solution and its gradient in one dimension.
   """
   Tu = knots
   nu = uh.shape[0]
   pu = len(Tu) - nu -1
   
   if mesh is None:

      if Npoints is None:
         nx     = nu-pu+1
         mesh = Tu[pu:-pu] 
      else :
         '''
         x0_v  : min val in x direction
         x1_v  : max val in x direction
         '''
         nx   = Npoints
         if bound_val is not None:
            x0_v = bound_val[0]
            x1_v = bound_val[1] 

         else :
            x0_v = Tu[pu]
            x1_v = Tu[-pu-1]
         # ...
         mesh  = linspace(x0_v, x1_v, nx)
   # ...
   nx       = mesh.shape[0]
   Q        = zeros((nx, 3))
   Q[:,2]   = mesh[:]
   core.sol_field_1D_meshes(nx, uh, Tu, pu, Q)
   return Q[:,0], Q[:,1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_2d( Npoints, uh, knots, degree, mesh = None, bound_val = None):
    '''
    Using computed control points uh we compute solution
    in new discretisation by Npoints
    The solution can be determined within the provided mesh
    The solution can be calculated within a particular domain : bound_val    
    '''
    pu, pv = degree
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    
    if mesh is None:

      if Npoints is None:

         nx = nu-pu+1
         ny = nv-pv+1
      
         xs = Tu[pu:-pu] 
         ys = Tv[pv:-pv] 
      
      else :
         '''
         x0_v  : min val in x direction
         x1_v  : max val in x direction
         y0_v  : min val in y direction
         y1_v  : max val in y direction
         '''
         nx, ny  = Npoints
         if bound_val is not None:

            x0_v = bound_val[0]
            x1_v = bound_val[1] 
            y0_v = bound_val[2] 
            y1_v = bound_val[3]

         else :
            x0_v = Tu[pu]
            x1_v = Tu[-pu-1]
            y0_v = Tv[pv]
            y1_v = Tv[-pv-1]
         # ...
         xs                     = linspace(x0_v, x1_v, nx)
         ys                     = linspace(y0_v, y1_v, ny)
      # ...
      Q    = zeros((nx, ny, 3)) 
      core.sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, Q)
      # ...
      X, Y = meshgrid(xs, ys)
      return Q[:,:,0], Q[:,:,1], Q[:,:,2], X.T, Y.T
    else :
       # ...
       nx, ny   = mesh[0].shape
       Q        = zeros((nx, ny, 5))
       Q[:,:,3] = mesh[0][:,:]
       Q[:,:,4] = mesh[1][:,:] 
       core.sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, Q)       
       return Q[:,:,0], Q[:,:,1], Q[:,:,2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In three dimension
def pyccel_sol_field_3d(Npoints,  uh , knots, degree, mesh = None):
    '''
    Using computed control points uh we compute solution
    in new discretisation by Npoints    
    '''

    pu, pv, pz = degree
    Tu, Tv, Tz = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    nz = len(Tz) - pz - 1    
    if mesh is None:
       if Npoints is None:

            nx = nu-pu+1
            ny = nv-pv+1
            nz = nz-pz+1
    
            xs = Tu[pu:-pu] #linspace(Tu[pu], Tu[-pu-1], nx)
    
            ys = Tv[pv:-pv] #linspace(Tv[pv], Tv[-pv-1], ny)
      
            zs = Tz[pz:-pz] #linspace(Tv[pv], Tv[-pv-1], ny)
      
       else :
            nx, ny, nz = Npoints

            xs = linspace(Tu[pu], Tu[-pu-1], nx)
          
            ys = linspace(Tv[pv], Tv[-pv-1], ny)
       
            zs = linspace(Tz[pz], Tz[-pz-1], nz)
       Q    = zeros((nx, ny, nz, 7)) 
       core.sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tz, pu, pv, pz, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3], Q[:,:,:,4], Q[:,:,:,5], Q[:,:,:,6]
    else :
       nx, ny, nz = Npoints
       # ...
       Q          = zeros((nx, ny, nz, 7))
       Q[:,:,:,4] = mesh[0][:,:,:]
       Q[:,:,:,5] = mesh[1][:,:,:]
       Q[:,:,:,6] = mesh[2][:,:,:]
       core.sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tz, pu, pv, pz, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3]

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

import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
import numpy                        as     np
import pyvista                      as     pv
import os
colors = ['b', 'k', 'r', 'g', 'm', 'c', 'y', 'orange']
markers = ['v', 'o', 's', 'D', '^', '<', '>', '*']  # Different markers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_SolutionMultipatch(nbpts, xuh, V, xmp, ymp, savefig = None, plot = True, Jacfield = None): 
   """
   Plot the solution of the problem in the whole multi-patch domain
   """
   #---Compute a solution
   numPaches = len(V)
   u   = []
   F1  = []
   F2  = []
   JF = []
   for i in range(numPaches):
      u.append(pyccel_sol_field_2d((nbpts, nbpts), xuh[i], V[i].knots, V[i].degree)[0])
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      #...Compute a Jacobian
      F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[1:3]
      F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[1:3]
      JF.append(F1x*F2y - F1y*F2x)
   if Jacfield is not None:
       u = JF

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

   # --- Create Figure ---
   fig, axes = plt.subplots(figsize=(8, 6))

   # --- Contour Plot for First Subdomain ---
   im = []
   for i in range(numPaches):
      im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
      # --- Colorbar ---
      divider = make_axes_locatable(axes)
      cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
      cbar = plt.colorbar(im[i], cax=cax)
      cbar.ax.tick_params(labelsize=15)
      cbar.ax.yaxis.label.set_fontweight('bold')
   # --- Formatting ---
   axes.set_title("Solution the in whole domain ", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_JacobianMultipatch(nbpts, V, xmp, ymp, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   u   = []
   F1  = []
   F2  = []
   for i in range(numPaches):
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      #...Compute a Jacobian
      F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[1:3]
      F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[1:3]
      u.append(F1x*F2y - F1y*F2x)

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

   # --- Create Figure ---
   fig, axes = plt.subplots(figsize=(8, 6))

   # --- Contour Plot for First Subdomain ---
   im = []
   for i in range(numPaches):
      im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
      # --- Colorbar ---
      divider = make_axes_locatable(axes)
      cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
      cbar = plt.colorbar(im[i], cax=cax)
      cbar.ax.tick_params(labelsize=15)
      cbar.ax.yaxis.label.set_fontweight('bold')
   # --- Formatting ---
   #axes.set_title("Jacobian the in whole domain ", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_MeshMultipatch(nbpts, V, xmp, ymp, cp = True, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])

   # --- Create Figure ---
   fig =plt.figure() 
   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if ii == 1:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '-g', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      else :
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '-g', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      #''
      phidx = F1[ii][0,:]
      phidy = F2[ii][0,:]
      plt.plot(phidx, phidy, '-r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
      # ...
      phidx = F1[ii][nbpts-1,:]
      phidy = F2[ii][nbpts-1,:]
      plt.plot(phidx, phidy, '-r', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_FunctMultipatch(nbpts, V, xmp, ymp, functions, cp = True, savefig = None, plot = True): 
   """
   Plot the function in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1     = []
   F2     = []
   values = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      values.append(functions(F1[i], F2[i]))

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(values[0]), np.min(values[1]))
   u_max  = max(np.max(values[0]), np.max(values[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(values[i]))
      u_max  = max(u_max, np.max(values[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots
   # --- Create Figure ---
   # ... Analytic Density function
   fig, axes =plt.subplots() 
   for i in range(numPaches):
      im2 = plt.contourf( F1[i], F2[i], values[i], levels, cmap= 'plasma')
   #divider = make_axes_locatable(axes) 
   #cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
   #plt.colorbar(im2, cax=cax) 
   fig.tight_layout()

   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_AdMeshMultipatch(nbpts, V, xmp, ymp, xad, yad, cp = True, savefig = None, plot = True, patchesInterface = False): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
      sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, mesh=(sx, sy))[0])
      F2.append(pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, mesh=(sx, sy))[0])

   # --- Create Figure ---
   fig =plt.figure() 

   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if patchesInterface:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25 ,label = '$Im([0,1]^2_{y=1})$')

         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25,label = '$Im([0,1]^2_{y=1})$')
         #''
         phidx = F1[ii][0,:]
         phidy = F2[ii][0,:]
         plt.plot(phidx, phidy, '--k',  linewidth=0.25, label = '$Im([0,1]^2_{x=0})$')
         # ...
         phidx = F1[ii][nbpts-1,:]
         phidy = F2[ii][nbpts-1,:]
         plt.plot(phidx, phidy, '--k', linewidth= 0.25, label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0