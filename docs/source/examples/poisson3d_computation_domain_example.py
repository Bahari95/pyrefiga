"""
poisson3d_computation_domain_example.py

Example: Solving Poisson's Equation on a 3D using B-spline or NURBS representation.

author :  M. BAHARI
"""

from pyrefiga import compile_kernel, apply_dirichlet

from pyrefiga import SplineSpace
from pyrefiga import TensorSpace
from pyrefiga import StencilMatrix
from pyrefiga import StencilVector
from pyrefiga import pyccel_sol_field_3d
from pyrefiga import Poisson
from pyrefiga import getGeometryMap
# ... Using Matrices accelerated with Pyccel
from   pyrefiga                    import assemble_stiffness1D
from   pyrefiga                    import assemble_mass1D     

#---In Poisson equation
from gallery.gallery_section_03 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery.gallery_section_03 import assemble_matrix_un_ex01 #---1 : In uniform mesh
from gallery.gallery_section_03 import assemble_norm_ex01      #---1 : In uniform mesh

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time
import argparse
from tabulate import tabulate
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2, V3, V):
       u                   = StencilVector(V.vector_space)
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                  = assemble_stiffness1D(V1)
       K1                  = K1.tosparse()
       K1                  = K1.toarray()[1:-1,1:-1]
       K1                  = csr_matrix(K1)

       M1                  = assemble_mass1D(V1)
       M1                  = M1.tosparse()
       M1                  = M1.toarray()[1:-1,1:-1]
       M1                  = csr_matrix(M1)

       # Stiffness and Mass matrix in 1D in the second deriction
       K2                  = assemble_stiffness1D(V2)
       K2                  = K2.tosparse()
       K2                  = K2.toarray()[1:-1,1:-1]
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = M2.tosparse()
       M2                  = M2.toarray()[1:-1,1:-1]
       M2                  = csr_matrix(M2)
       
       # Stiffness and Mass matrix in 1D in the thrd deriction
       K3                  = assemble_stiffness1D(V3)
       K3                  = K3.tosparse()
       K3                  = K3.toarray()[1:-1,1:-1]
       K3                  = csr_matrix(K3)

       M3                  = assemble_mass1D(V3)
       M3                  = M3.tosparse()
       M3                  = M3.toarray()[1:-1,1:-1]
       M3                  = csr_matrix(M3)

       # ...
       #M                   = kron(K1,kron(M2,M3))+kron(M1,kron(K2,M3))+kron(M1,kron(M2,K3))
       #lu                  = sla.splu(csc_matrix(M))
       mats_1              = [M1, K1]
       mats_2              = [M2, K2]
       mats_3              = [M3, K3]
       # ...Fast Solver
       poisson             = Poisson(mats_1, mats_2, mats_3)

       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V )
       b                   = rhs.toarray()
       b                   = b.reshape(V.nbasis)
       b                   = b[1:-1, 1:-1, 1:-1]      
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis-2)*(V3.nbasis-2))
       # ...
       #xkron               = lu.solve(b)       
       xkron               = poisson.solve(b)
       
       xkron               = xkron.reshape([V1.nbasis-2,V2.nbasis-2,V3.nbasis-2])
       # ...
       x                   = np.zeros(V.nbasis)
       x[1:-1, 1:-1, 1:-1] = xkron
       x                   = x.reshape(V.nbasis)
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()

nbpts       = 100 #for plot
nelements   = 16
# Test 1
g           = ['np.sin(np.pi*x)*y**2*z*3*np.sin(4.*np.pi*(1.-y))*(1.-z)']
geometry = '../fields/cube.xml'
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)

# Extract geometry mapping
mp             = getGeometryMap(geometry,0)
degree         = mp.degree[0] # Use same degree as geometry
xmp, ymp, zmp  = mp.RefineGeometryMap(Nelements=(nelements,nelements,nelements))

#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
quad_degree = degree + 1
# create the spline space for each direction
V1   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V2   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V3   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V    = TensorSpace(V1, V2, V3)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, V3, V)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
from pyrefiga    import paraview_nurbsSolutionMultipatch
solutions = [
    {"name": "Solution", "data": [xuh]}
]
functions = [
    {"name": "Exact solution", "expression": g[0]},
]
paraview_nurbsSolutionMultipatch(nbpts, [V], [xmp], [ymp], zmp=[zmp], solution = solutions, functions = functions)
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])