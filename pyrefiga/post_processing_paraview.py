"""
This module provides utilities for NURBS-based computations, including solution field evaluation,
prolongation, and Paraview post-processing for multi-patch domains.

@author : M. BAHARI
"""
from   .results_f90     import sol_field_NURBS_2d, sol_field_NURBS_3d

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# ... Post processing using Paraview 
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
import numpy                        as     np
import pyvista                      as     pv
from   .utilities                   import pyref_multipatch, pyref_patch
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsAdMeshMultipatch(nbpts, pyrefGeometry, moving_mesh, solution = None, functions = None, precomputed = None, Analytic = None, filename = "figs/admultipatch_multiblock", plot = False): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   pyrefGeometry: pyref_multipatch or pyref_patch
   moving_mesh : collable , optional
       List of solution control points for each patch and its name.
       solutions = [
         {"name": "x", "data": xuh = list, V},   # e.g., x/mapping field control points
         {"name": "y", "data": yuh = list, V},   # e.g., y/mapping field control points
         {"name": "z", "data": zuh = list, V},   # e.g., z/mapping field control points
      ]
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
   nb_Patches = pyrefGeometry.nb_patches
   # ...
   if Analytic is not None and pyrefGeometry.geo_dim != 2:
      raise TypeError('Not implemented')
   
   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()
   #F3 = [] 
   if pyrefGeometry.geo_dim == 2:
      for i in range(nb_Patches):
         # ....
         mm  =  moving_mesh[0]
         assert mm["name"] == 'x', "First moving mesh must be x mapping"
         sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:3]
         mm  =  moving_mesh[1]
         assert mm["name"] == 'y', "Second moving mesh must be y mapping"
         sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:3]
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
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   else:
      if pyrefGeometry.dim == 2 : # 2D adaptive mesh in 3D mapping
            for i in range(nb_Patches):
               #... computes adaptive mesh
               mm  =  moving_mesh[0]
               assert mm["name"] == 'x', "First moving mesh must be x mapping"
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:3]
               mm  =  moving_mesh[1]
               assert mm["name"] == 'y', "Second moving mesh must be y mapping"
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:3]
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
                     U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
               # .... 
               if precomputed is not None :
                  for sol in precomputed:
                     grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
               multiblock[f"patch_{i}"] = grid
      else: #... zad
         for i in range(nb_Patches):
            #... computes adaptive mesh
            mm  =  moving_mesh[0]
            assert mm["name"] == 'x', "First moving mesh must be x mapping"
            sx, uxx, uxy, uxz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:4]
            mm  =  moving_mesh[1]
            assert mm["name"] == 'y', "Second moving mesh must be y mapping"
            sy, uyx, uyy, uyz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:4]
            mm  =  moving_mesh[2]
            assert mm["name"] == 'z', "Third moving mesh must be z mapping"
            sz, uzx, uzy, uzz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), mm["data"][i], mm["space"].omega, mm["space"].knots, mm["space"].degree)[0:4]
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
            grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

            # Flatten the solution and attach as a scalar field
            # ... image bu analytic function
            if functions is not None:
               for Funct in functions:
                  fnc = eval(Funct["expression"])
                  grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
            # .... 
            if solution is not None:
               for sol in solution:
                  U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
            # .... 
            if precomputed is not None :
               for sol in precomputed:
                  grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
            multiblock[f"patch_{i}"] = grid

   # Save multiblock dataset
   multiblock.save(filename+".vtm")
   print(f"Saved all patches with solution to {filename}.vtm")
   if plot:
    import subprocess
    # Load the multipatch VTM
    subprocess.run(["paraview", filename+".vtm"])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsSolutionMultipatch(nbpts, pyrefGeometry, solution = None, functions = None, precomputed = None, Analytic = None, filename = "figs/multipatch_solution", plot = False): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   pyrefGeometry: pyref_multipatch or pyref_patch

   solution : list, optional
       List of solution control points for each patch and name.
       solutions = [
         {"name": "displacement", "data": xuh, "space": V},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh, "space": V},   # e.g., velocity field control points
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
   nb_Patches = pyrefGeometry.nb_patches

   if Analytic is not None and pyrefGeometry.geo_dim != 2:
         raise TypeError('Not implemented')

   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()
   if pyrefGeometry.geo_dim == 2:
      for i in range(nb_Patches):
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
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   elif pyrefGeometry.dim == 3: #.. z is not none 3D case
      for i in range(nb_Patches):
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
         grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

         # Flatten the solution and attach as a scalar field
         # ... image bu analytic function
         if functions is not None :
            for Funct in functions:
               fnc = eval(Funct["expression"])
               grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
         # .... 
         if solution is not None :
            for sol in solution:
               U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
         # .... 
         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   else: #.. z is not none
      for i in range(nb_Patches):
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
               U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
               grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
         # ...
         if precomputed is not None :
            for sol in precomputed:
               grid[sol["name"]] = sol["data"][i].flatten(order='C')  # or 'F' if needed (check your ordering)
         multiblock[f"patch_{i}"] = grid
   
   # Save multiblock dataset
   multiblock.save(filename+".vtm")
   print(f"Saved all patches with solution to {filename}.vtm")
   if plot:
    import subprocess
    # Load the multipatch VTM
    subprocess.run(["paraview", filename+".vtm"])

def ViewGeo(geometry, Nump, nbpts=50, functions = None, Analytic = None, filename="figs/multipatch_geometry", plot = True):
   """
   Example on how one can use nurbs mapping and prolongate it in fine grid
   """

   print('#---: ', geometry, Nump)
   mp  = pyref_multipatch(geometry, Nump)
   print("geom dim = ",mp.geo_dim)
   # ... save a solution as .vtm for paraview
   paraview_nurbsSolutionMultipatch(nbpts, mp, functions = functions, Analytic=Analytic, filename=filename)      
   #------------------------------------------------------------------------------
   # Show or close plots depending on argument
   if plot :
      import subprocess

      # Load the multipatch VTM
      subprocess.run(["paraview", filename+".vtm"])


#--------------------------------------------------------------------------------------------------------------------
# ... Time post-processes TODO
#--------------------------------------------------------------------------------------------------------------------
def paraview_TimeSolutionMultipatch(nbpts, pyrefGeometry, LStime = None, solution = None, functions = None, precomputed = None, filename = "figs/multipatch_solution", plot = False): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.

   pyrefGeometry: pyref_multipatch or pyref_patch
   LStime: liste of times
   solution : list, optional
       List of solution control points for each patch and name.
       solutions = [
         {"name": "displacement", "data": xuh, "space": V},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh, "space": V},   # e.g., velocity field control points
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
   assert isinstance(pyrefGeometry, (pyref_multipatch, pyref_patch)), \
   "Geomapping must be pyref_multipatch or pyref_patch class"
   #---
   nb_Patches       = pyrefGeometry.nb_patches
   # ...
   output_pvd_path = os.path.abspath(filename + ".pvd")
   # --- Create PVD file header
   with open(output_pvd_path, 'w') as f:
      f.write('<?xml version="1.0"?>\n')
      f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
      f.write('  <Collection>\n')
      if pyrefGeometry.geo_dim ==2:
         for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
            multiblock = pv.MultiBlock()
            for i in range(nb_Patches):
               #---Compute a physical domain
               x, y  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts))
               [[F1x, F1y], [F2x, F2y]] = pyrefGeometry.gradient(i+1, nbpts=(nbpts, nbpts))
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
               if functions is not None :
                  for Funct in functions:
                     fnc = eval(Funct["expression"])
                     grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               # ...
               if solution is not None :
                  for sol in solution:
                     U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
               # ...
               if precomputed is not None :
                  for sol in precomputed:
                     grid[sol["name"]] = sol["data"][i][t_ix].flatten(order='C')  # or 'F' if needed (check your ordering)
               multiblock[f"patch_{i}"] = grid
            # --- Save one .vtm per time step
            vtm_filename = f"{filename}_t{t_ix:03d}.vtm"
            multiblock.save(vtm_filename)


            rel_path = os.path.basename(vtm_filename)
            f.write(f'    <DataSet timestep="{t_ix}" group="" part="0" file="{rel_path}"/>\n')
      elif pyrefGeometry.dim == 3: #.. z is not none 3D case
         for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
            multiblock = pv.MultiBlock()
            for i in range(nb_Patches):
               #---Compute a physical domain
               x, y, z  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts, nbpts))
               # [[uxx, uxy, uxz],[uyx, uyy, uyz],[uzx, uzy, uzz]]   = pyrefGeometry.gradient(i+1, nbpts=(nbpts, nbpts, nbpts))
               #...Compute a Jacobian
               # Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # ... image bu analytic function
               t = LStime[t_ix]
               if functions is not None :
                  for Funct in functions:
                     fnc = eval(Funct["expression"])
                     grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               # .... 
               if solution is not None :
                  for sol in solution:
                     U                 = sol_field_NURBS_3d((nbpts, nbpts, nbpts), sol["data"][i][t_ix], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
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
         for t_ix in range(len(LStime)): # assuming time is the 2nd dimension of solution["data"][i][t]
            multiblock = pv.MultiBlock()
            for i in range(nb_Patches):
               #---Compute a physical domain
               x, y, z  = pyrefGeometry.eval(i+1, nbpts=(nbpts, nbpts))
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # ... image bu analytic function
               t = LStime[t_ix]
               if functions is not None :
                  for Funct in functions:
                     fnc = eval(Funct["expression"])
                     grid[Funct["name"]] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               # ...
               if solution is not None :
                  for sol in solution:
                     U                 = sol_field_NURBS_2d((nbpts, nbpts), sol["data"][i][t_ix], sol["space"].omega, sol["space"].knots, sol["space"].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)
               # ...
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
   if plot:
    import subprocess
    # Load the multipatch VTM
    subprocess.run(["paraview", output_pvd_path])
