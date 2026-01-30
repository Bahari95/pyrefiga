"""
poisson2d_neumann_example.py

A basic test for Neumann boundary conditions.(2D)

author :  M. BAHARI
"""

from pyrefiga import compile_kernel 
from pyrefiga import apply_dirichlet

from pyrefiga import SplineSpace
from pyrefiga import TensorSpace
from pyrefiga import StencilVector
from pyrefiga import pyccel_sol_field_2d

# ... Using Matrices accelerated with Pyccel
from   pyrefiga                    import assemble_stiffness1D
from   pyrefiga                    import assemble_mass1D     

#---In Poisson equation
from gallery.gallery_section_02 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery.gallery_section_02 import assemble_norm_ex01      #---1 : In uniform mesh

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
import time

#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2, V):
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
       K2                  = K2.tosparse()
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = M2.tosparse()
       M2                  = csr_matrix(M2)
       
       # Stiffness and Mass matrix in 1D in the thrd deriction
       M                   = kron(K1,M2)+kron(M1,K2)+kron(M1,M2)
       lu                  = sla.splu(csc_matrix(M))
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V, fields=[u] )
       b                   = apply_dirichlet(V, rhs, dirichlet=[[True, True], [False, False]])
       # ...
       xkron               = lu.solve(b)       
       xkron               = xkron.reshape([V1.nbasis-2,V2.nbasis])
       # ...
       u                   = apply_dirichlet(V, xkron, dirichlet=[[True, True], [False, False]], update = u)#zero Dirichlet
       x                   = u.tensor
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1
#----------------------
u_exact = [ 'np.sin(np.pi*x)* np.sin(np.pi*y)']
#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
nelements  = 64
# create the spline space for each direction
V1   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V2   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
Vh   = TensorSpace(V1, V2)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, Vh)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))

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