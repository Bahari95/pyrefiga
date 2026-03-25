"""
results_f90.py: A fast processing script for computing the solution and its derivatives using B-spline or Nurbs functions.

Author: M. Mustapha Bahari
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

