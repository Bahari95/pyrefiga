"""
poisson3d_computation_domain_example.py

Example: Solving Poisson's Equation on a 3D using B-spline on the computational domain.

author :  M. BAHARI
"""

from  pyrefiga                      import compile_kernel
from  pyrefiga                      import apply_dirichlet
# ...
from  pyrefiga                     import SplineSpace
from  pyrefiga                     import TensorSpace
from  pyrefiga                     import StencilVector
from  pyrefiga                     import Poisson
from  pyrefiga                     import pyref_patch
# ... Using Matrices accelerated with Pyccel
from   pyrefiga                    import assemble_stiffness1D
from   pyrefiga                    import assemble_mass1D     

#---In Poisson equation
from gallery.gallery_section_03 import assemble_vector_ex01
from gallery.gallery_section_03 import assemble_norm_ex01  

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
import numpy                        as     np
import time
import argparse
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
       K1                  = apply_dirichlet(V1, K1)
       K1                  = csr_matrix(K1)

       M1                  = assemble_mass1D(V1)
       M1                  = apply_dirichlet(V1, M1)
       M1                  = csr_matrix(M1)

       # Stiffness and Mass matrix in 1D in the second deriction
       K2                  = assemble_stiffness1D(V2)
       K2                  = apply_dirichlet(V2, K2)
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = apply_dirichlet(V2, M2)
       M2                  = csr_matrix(M2)
       
       # Stiffness and Mass matrix in 1D in the thrd deriction
       K3                  = assemble_stiffness1D(V3)
       K3                  = apply_dirichlet(V3, K3)
       K3                  = csr_matrix(K3)

       M3                  = assemble_mass1D(V3)
       M3                  = apply_dirichlet(V3, M3)
       M3                  = csr_matrix(M3)

       # ...Fast diag solver based on kronecker product
       mats_1              = [M1, K1]
       mats_2              = [M2, K2]
       mats_3              = [M3, K3]
       # ...Fast Solver
       poisson             = Poisson(mats_1, mats_2, mats_3)

       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V )
       b                   = apply_dirichlet(V, rhs)      
       # ...
       xkron               = poisson.solve(b)
       
       u                   = apply_dirichlet(V, xkron, update = u)# zero Dirichlet
       # ...
       x                   = u.toarray().reshape(V.nbasis)
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

nbpts       = 50 #for plot
nelements   = 8
degree      = 2
# Test 1
from pyrefiga import load_xml
geometry    = load_xml('cube.xml')
g           = ['np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)']
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)
# Extract geometry mapping
mp             = pyref_patch(geometry,0)

#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
# create the spline space for each direction
V1   = SplineSpace(degree=degree, nelements= nelements)
V2   = SplineSpace(degree=degree, nelements= nelements)
V3   = SplineSpace(degree=degree, nelements= nelements)
V    = TensorSpace(V1, V2, V3)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, V3, V)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))
print(np.min(xuh), np.max(xuh))
#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
from pyrefiga    import paraview_nurbsSolutionMultipatch
solutions = [
    {"name": "Solution", "data": [xuh], "space": V},
]
functions = [
    {"name": "Exact solution", "expression": g[0]},
]
paraview_nurbsSolutionMultipatch(nbpts, mp, solution = solutions, functions = functions, plot = args.plot)