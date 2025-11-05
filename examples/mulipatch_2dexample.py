"""
mulipatch_2dexample.py

Example multipatch: Solving Poisson's Equation on a 2D complex geometry using B-spline or NURBS representation.

Author: M. Bahari
"""
from   pyrefiga                         import compile_kernel
from   pyrefiga                         import apply_dirichlet

from   pyrefiga                         import SplineSpace
from   pyrefiga                         import TensorSpace
from   pyrefiga                         import StencilMatrix
from   pyrefiga                         import StencilVector

from   pyrefiga                         import pyccel_sol_field_2d
from   pyrefiga                         import prolongation_matrix
from   pyrefiga                         import least_square_Bspline
from   pyrefiga                         import getGeometryMap
from   pyrefiga                         import build_dirichlet
from   pyrefiga                         import pyrefInterface

# Import Poisson assembly tools for uniform mesh
from gallery.gallery_section_10 import assemble_vector_un_ex01
from gallery.gallery_section_10 import assemble_norm_un_ex01

assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

# ... nitsche assembly tools
from gallery.gallery_nitsche_00 import assemble_matrix_nitsche_ex00
assemble_stiffness_nitsche  = compile_kernel(assemble_matrix_nitsche_ex00, arity=2)
from gallery.gallery_nitsche_00 import assemble_matrix_nitsche_ex02
assemble_stiffness2_nitsche = compile_kernel(assemble_matrix_nitsche_ex02, arity=2)

from   scipy.sparse                      import csr_matrix
from   scipy.sparse                      import csc_matrix, linalg as sla
from   numpy                             import zeros
from   tabulate                          import tabulate
import numpy                             as     np
import timeit
# import math # for debugging purposes
# # Check manually
# has_nan = np.argwhere(np.isnan(M))
# print("NaN found at indices:", has_nan)
# print("Contains NaN  =========================: ", has_nan)
import time
import argparse

#------------------------------------------------------------------------------
# Create directory for figures if it doesn't exist
#------------------------------------------------------------------------------
import os
os.makedirs("figs", exist_ok=True)

#------------------------------------------------------------------------------
# Poisson solver algorithm for two patches
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# The convention for the patches is as follows:
#                                                 ___4___
#                                                |       |
#                                               1|       |2
#                                                |_______|
#                                                    3
#------------------------------------------------------------------------------
def poisson_solve(V, u11_mph, u12_mph, u21_mph, u22_mph, u_d1, u_d2, interface, dirichlet_1, dirichlet_2):
    # Build global linear system
    n_basis                    = V.nbasis[0] * V.nbasis[1]
    M                          = zeros((n_basis*2,n_basis*2))
    #... computes coeffs for Nitsche's method
    stab                       = 4.*( V.degree[0] + V.dim ) * ( V.degree[0] + 1 )
    m_h                        = (V.nbasis[0]*V.nbasis[1])
    Kappa                      = 2.*stab*m_h
    print("Kappa = ", Kappa)
    # ...
    normS                      = 0.5

    # Assemble stiffness matrix 11
    stiffness11                = assemble_stiffness_nitsche(V, fields=[u11_mph, u12_mph], knots=True, value=[Vh.omega[0],Vh.omega[1], interface[0], Kappa, normS])
    stiffness11                = apply_dirichlet(V, stiffness11, dirichlet=dirichlet_1)
    M[:n_basis,:n_basis]       = (stiffness11.tosparse()).toarray()[:,:]

    # Assemble stiffness matrix 22
    stiffness22                = assemble_stiffness_nitsche(V, fields=[u21_mph, u22_mph], knots=True, value=[Vh.omega[0],Vh.omega[1], interface[1], Kappa, normS])
    stiffness22                = apply_dirichlet(V, stiffness22, dirichlet=dirichlet_2)
    M[n_basis:,n_basis:]       = (stiffness22.tosparse()).toarray()[:,:]

    # # # Assemble Nitsche's method matrices
    print("Assembling Nitsche matrices...", interface[0])
    stiffness12                = assemble_stiffness2_nitsche(V, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, value=[Vh.omega[0],Vh.omega[1], interface[0], Kappa, normS])
    stiffness12                = apply_dirichlet(V, stiffness12, dirichlet=dirichlet_1)
    M[:n_basis,n_basis:]       = (stiffness12.tosparse()).toarray()[:,:]

    # Assemble Nitsche's method matrices
    stiffness21                = assemble_stiffness2_nitsche(V, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, value=[Vh.omega[0],Vh.omega[1], interface[1], Kappa, normS])
    stiffness21                = apply_dirichlet(V, stiffness21, dirichlet=dirichlet_2)
    M[n_basis:,:n_basis]       = (stiffness21.tosparse()).toarray()[:,:]

    print("Test if Nitsche matrix is symmetric: ", np.allclose(stiffness12.tosparse().toarray(), stiffness21.tosparse().toarray().T))


    # Assemble right-hand side vector
    rhs                       = assemble_rhs_un( V, fields=[u11_mph, u12_mph, u_d1])
    rhs1                      = apply_dirichlet(V, rhs, dirichlet=dirichlet_1)
    # Assemble right-hand side vector
    rhs                       = assemble_rhs_un( V, fields=[u21_mph, u22_mph, u_d2])
    rhs2                      = apply_dirichlet(V, rhs, dirichlet=dirichlet_2)
    # Assemble right-hand side vector
    b                          = zeros(n_basis*2)
    b[:n_basis]                = rhs1.toarray()[:]
    b[n_basis:]                = rhs2.toarray()[:]

    # Solve the linear system using CGS
    x, inf          = sla.cg(M, b)

    # ... Extract solution
    u1              = StencilVector(V.vector_space)
    u2              = StencilVector(V.vector_space)
    # ... Convert solution to StencilVector format
    x1              = (x[:n_basis]+u_d1.toarray()).reshape(V.nbasis)
    u1.from_array(V, x1)
    x2              = (x[n_basis:]+u_d2.toarray()).reshape(V.nbasis)
    u2.from_array(V, x2)

    # Compute L2 and H1 errors
    norm0   = assemble_norm_un(V, fields=[u11_mph, u12_mph, u1]).toarray()
    norm1   = assemble_norm_un(V, fields=[u21_mph, u22_mph, u2]).toarray()
    l2_norm = np.sqrt(norm0[0]**2+norm1[0]**2)
    H1_norm = np.sqrt(norm0[1]**2+norm1[1]**2)
    print("L2 norm:", l2_norm, "H1 norm:", H1_norm, "area: ", norm0[2], "+", norm1[2])
    return x1, x2, l2_norm, H1_norm

#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()

#------------------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------------------
nbpts       = 100 # Number of points for plotting
RefinNumber = 1   # Number of global mesh refinements
nelements   = 8  # Initial mesh size
table       = zeros((RefinNumber+1,5))
i           = 1
times       = []

print("(#=assembled Dirichlet, #=solve poisson)\n")

#------------------------------------------------------------------------------
# Define exact solution and Dirichlet boundary condition
#------------------------------------------------------------------------------
# Test 0
g         = ['np.sin(2.*np.pi*x)*np.sin(2.*np.pi*y)']
# Test 1
#g         = ['1./(1.+np.exp((x + y  - 0.5)/0.01) )']

#------------------------------------------------------------------------------
# Load CAD geometry
#------------------------------------------------------------------------------
#geometry = '../fields/unitSquare.xml'
#geometry = '../fields/circle.xml'
geometry = '../fields/quart_annulus.xml'
# geometry = '../fields/annulus.xml'
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)

# Extract geometry mapping
mp              = getGeometryMap(geometry,0)# .. First patch 
mp1             = getGeometryMap(geometry,1)# .. Second patch
degree          = mp.degree # Use same degree as geometry
quad_degree     = max(degree[0],degree[1])+1 # Quadrature degree
mp.nurbs_check  = True # Activate NURBS if geometry uses NURBS
mp1.nurbs_check = True # Activate NURBS if geometry uses NURBS

#------------------------------------------------------------------------------
# Initialize spaces and mapping for initial mesh
#------------------------------------------------------------------------------
Nelements        = (nelements,nelements)
weight, xmp, ymp = mp.RefineGeometryMap(Nelements=Nelements)
wm1, wm2         = weight[:,0], weight[0,:]
xmp1, ymp1       = mp1.RefineGeometryMap(Nelements=Nelements)[1:]

#------------------------------------------------------------------------------
# Detect interface between patches
#------------------------------------------------------------------------------
rInt             = pyrefInterface(xmp, ymp, xmp1, ymp1)
rInt.printInterface() # Print detected interface

# Create spline spaces for each direction
V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
# Create tensor product space
Vh = TensorSpace(V1, V2)
print("degree = {}  nelements = {}".format(Vh.degree, Vh.nelements))
# Initialize mapping vectors
u11_mph        = StencilVector(Vh.vector_space)
u12_mph        = StencilVector(Vh.vector_space)
u11_mph.from_array(Vh, xmp)
u12_mph.from_array(Vh, ymp)

# Initialize mapping vectors
u21_mph        = StencilVector(Vh.vector_space)
u22_mph        = StencilVector(Vh.vector_space)
u21_mph.from_array(Vh, xmp1)
u22_mph.from_array(Vh, ymp1)

#------------------------------------------------------------------------------
# Assemble Dirichlet boundary conditions
#------------------------------------------------------------------------------
xd1, u_d1 = build_dirichlet(Vh, g, map = (xmp, ymp))
xd2, u_d2 = build_dirichlet(Vh, g, map = (xmp1, ymp1))
xd1, xd2  = rInt.setInterface(xd1, xd2)
u_d1.from_array(Vh, xd1)
u_d2.from_array(Vh, xd2)
print('#')

# Solve Poisson equation on coarse grid
start = time.time()
xuh1, xuh2, l2_error,  H1_error = poisson_solve(Vh, u11_mph, u12_mph, u21_mph, u22_mph, u_d1, u_d2, rInt.interface, rInt.dirichlet_1, rInt.dirichlet_2)
times.append(time.time()- start)
print('#')

# Store results in table
table[0,:] = [degree[0], nelements, l2_error, H1_error, times[-1]]

#------------------------------------------------------------------------------
# Mesh refinement loop 
#------------------------------------------------------------------------------
i_save = 1
for nbne in range(4,4+RefinNumber):
    # Refine mesh
    nelements  = 2**nbne
    Nelements  = (nelements,nelements)
    print('#---IN-UNIFORM--MESH', nelements)
    # Refine geometry mapping
    weight, xmp, ymp  = mp.RefineGeometryMap(Nelements=Nelements)
    wm1, wm2 = weight[:,0], weight[0,:]
    xmp1, ymp1  = mp1.RefineGeometryMap(Nelements=Nelements)[1:]
    # Create spline spaces for refined mesh
    V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
    Vh = TensorSpace(V1, V2)
    print('#spaces')
    # Update mapping vectors
    u11_mph         = StencilVector(Vh.vector_space)
    u12_mph         = StencilVector(Vh.vector_space)
    u11_mph.from_array(Vh, xmp)
    u12_mph.from_array(Vh, ymp)
    # Update mapping vectors
    u21_mph         = StencilVector(Vh.vector_space)
    u22_mph         = StencilVector(Vh.vector_space)
    u21_mph.from_array(Vh, xmp1)
    u22_mph.from_array(Vh, ymp1)
    # Assemble Dirichlet boundary conditions
    xd1, u_d1 = build_dirichlet(Vh, g, map = (xmp, ymp))
    xd2, u_d2 = build_dirichlet(Vh, g, map = (xmp1, ymp1))
    xd1, xd2  = rInt.setInterface(xd1, xd2)
    u_d1.from_array(Vh, xd1)
    u_d2.from_array(Vh, xd2)
    print('#')
    # Solve Poisson equation on refined mesh
    start = time.time()
    xuh1, xuh2, l2_error,  H1_error = poisson_solve(Vh, u11_mph, u12_mph, u21_mph, u22_mph, u_d1, u_d2, rInt.interface, rInt.dirichlet_1, rInt.dirichlet_2)
    times.append(time.time()- start)
    print('#')
    # Store results
    table[i_save,:]                 = [degree[0], nelements, l2_error, H1_error, times[-1]]
    i_save                         += 1

#------------------------------------------------------------------------------
# Print error results in LaTeX table format
#------------------------------------------------------------------------------
if True :
    print("	\subcaption{Degree $p =",degree,"$}")
    print("	\\begin{tabular}{c|ccc|ccc}")
    print("		\hline")
    print("		 $\#$cells & $L^2$-Err & $H^1$-Err & cpu-time\\\\")
    print("		\hline")
    for i in range(0,RefinNumber+1):
        print("		",int(table[i,1]),"$\\times$", int(table[i,1]), "&", np.format_float_scientific(table[i,2], unique=False, precision=2), "&", np.format_float_scientific(table[i,3], unique=False, precision=2), "&", np.format_float_scientific(table[i,4], unique=False, precision=2),"\\\\")
    print("		\hline")
    print("	\end{tabular}")
print('\n')

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
from pyrefiga    import paraview_nurbsSolutionMultipatch
solutions = [
    {"name": "Solution", "data": [xuh1, xuh2]}
]
functions = [
    {"name": "Exact solution", "expression": g[0]},
]
paraview_nurbsSolutionMultipatch(nbpts, [Vh, Vh], [xmp, xmp1], [ymp, ymp1],  solution = solutions, functions = functions)
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])