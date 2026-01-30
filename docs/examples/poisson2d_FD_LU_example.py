"""
poisson2d_FD_LU_example.py

       example on how can fast diagonalization method be used to solve
       a Poisson equation in 2D on a square domain using isogeometric analysis
author :  M. BAHARI
"""

from pyrefiga import compile_kernel
from pyrefiga import apply_dirichlet
from pyrefiga import SplineSpace
from pyrefiga import TensorSpace
from pyrefiga import StencilVector
from pyrefiga import build_dirichlet
from pyrefiga import assemble_mass1D, assemble_stiffness1D
from pyrefiga import Poisson
#...
import time
start = time.time()

#---In Poisson equation
from gallery.gallery_section_04 import assemble_vector_ex01 #---1 : In uniform mesh
from gallery.gallery_section_04 import assemble_matrix_ex01 #---1 : In uniform mesh
from gallery.gallery_section_04 import assemble_norm_ex01 #---1 : In uniform mesh

assemble_stiffness2D = compile_kernel(assemble_matrix_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)
print('time to import utilities of Poisson equation =', time.time()-start)

from scipy.sparse        import csr_matrix, linalg as sla
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists
#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2 , V, x_d = None, u_d = None):


       u   = StencilVector(V.vector_space)
       K1         = assemble_stiffness1D(V1)
       K1         = apply_dirichlet(V1,K1)
       K1         = csr_matrix(K1)
       #___
       M1         = assemble_mass1D(V1)
       M1         = apply_dirichlet(V1,M1)
       M1         = csr_matrix(M1)

       #..Stiffness and Mass matrix in 1D in the second deriction
       K2         = assemble_stiffness1D(V2)
       K2         = apply_dirichlet(V2,K2)
       K2         = csr_matrix(K2)
       #___
       M2         = assemble_mass1D(V2)
       M2         = apply_dirichlet(V2,M2)
       M2         = csr_matrix(M2)

       #...step 0.1
       mats_1     = [M1, K1]
       mats_2     = [M2, K2]

       lu         = Poisson(mats_1, mats_2)
       # stiffness           = assemble_stiffness2D(V)
       # stiffness           = apply_dirichlet(V, stiffness)
       # #--Assembles matrix
       # lu                  = sla.splu(csc_matrix(stiffness))
       #--Assembles right hand side of Poisson equation
       rhs                 = assemble_rhs( V, fields = [u_d] )
       b                   = apply_dirichlet(V, rhs)
       # ...
       x                   = lu.solve(b)         

       # Rassembles Direcjlet boundary conditions
       u                   = apply_dirichlet(V, x, update = u_d)
       xsol                = u.tensor
       
       #--Computes error l2 and H1
       Norm                = assemble_norm_l2(V, fields=[u])
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]
       print('<.> l2_norm = {}  ||u||_H1= {} using nelement={} degree={} '.format(l2_norm, H1_norm, V.nelements, V.degree))
       return u, xsol, l2_norm, H1_norm


degree     = 4
nelements  = 64
#----------------------
# create the spline space for each direction
V1 = SplineSpace(degree=degree, nelements=nelements, nderiv = 2)
V2 = SplineSpace(degree=degree, nelements=nelements, nderiv = 2)

# create the tensor space
Vh = TensorSpace(V1, V2)

#..
g        = ['0.','2.* np.cos(np.pi*y)','2.* x ','-2.* x']
u_exact  = ['2.*x*np.cos(np.pi*y)']

x_d, u_d = build_dirichlet(Vh, g)
#----------------------
#---Solve Poisson equation
timer_start = time.time()
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, Vh, x_d = x_d, u_d = u_d)
xuh_uni = xuh
print('#---IN-UNIFORM--MESH: Poisson equation solved in {} seconds'.format(time.time() - timer_start))

# ...
plot = True
if plot:
       #---Compute a solution
       nbpts = 50
       from pyrefiga import pyref_patch, load_xml, paraview_nurbsSolutionMultipatch
       geometry = load_xml('unitSquare.xml')
       patch    = pyref_patch(geometry, 0)
       solutions = [
              {"name": "Solution", "data": [xuh], "space": Vh},
       ]
       functions = [
              {"name": "Exact solution", "expression": u_exact[0]},
       ]
       paraview_nurbsSolutionMultipatch(nbpts, patch, solution=solutions, functions=functions, plot = True)