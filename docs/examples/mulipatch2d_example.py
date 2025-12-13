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
from   pyrefiga                         import StencilNitsche
from   pyrefiga                         import pyccel_sol_field_2d
from   pyrefiga                         import prolongation_matrix
from   pyrefiga                         import least_square_Bspline
from   pyrefiga                         import getGeometryMap
from   pyrefiga                         import build_dirichlet
from   pyrefiga                         import pyrefInterface
from   pyrefiga                         import load_xml
# Import Poisson assembly tools for uniform mesh
from gallery.gallery_section_10 import assemble_matrix_un_ex00
from gallery.gallery_section_10 import assemble_vector_un_ex01
from gallery.gallery_section_10 import assemble_norm_un_ex01

assemble_stiffness   = compile_kernel(assemble_matrix_un_ex00, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

from   scipy.sparse                      import linalg as sla
from   numpy                             import zeros
from   tabulate                          import tabulate
import numpy                             as     np
import time
import argparse

#------------------------------------------------------------------------------
# Create directory for figures if it doesn't exist
#------------------------------------------------------------------------------
import os
os.makedirs("figs", exist_ok=True)
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
parser.add_argument("--nbpts", type=int, default=100, help="Number of elements used for plot(default: 50)")
parser.add_argument("--last", action="store_true", help="Enable iterations")
parser.add_argument("--h", type=int, default=2, help="Number of elements to elevalte the grid (default: 2)")
parser.add_argument("--i", type=int, default=1, help="Number of elements to elevalte the grid befor resolution (default: 1)")
parser.add_argument("--e", type=int, default=0, help="Number of elements to elevalte the degree (default: 0)")
args = parser.parse_args()

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

    Ni            = StencilNitsche(V, V, [dirichlet_1,dirichlet_2], interfaces= interface)
    #...

    # Assemble stiffness matrix 11
    stiffness11   = assemble_stiffness(V, fields=[u11_mph, u12_mph])
    Ni.applyNitsche(stiffness11,u11_mph, u12_mph, 1)
    stiffness11   = apply_dirichlet(V, stiffness11, dirichlet = dirichlet_1)
    Ni.appendBlock(stiffness11, 1)

    # Assemble stiffness matrix 22
    stiffness22   = assemble_stiffness(V, fields=[u21_mph, u22_mph])
    Ni.applyNitsche(stiffness22, u21_mph, u22_mph, 2)
    stiffness22   = apply_dirichlet(V, stiffness22, dirichlet = dirichlet_2)
    Ni.appendBlock(stiffness22, 2)

    #=======================================
    # # # Assemble Nitsche's method matrices
    #=======================================
    Ni.addNitscheoffDiag(u11_mph, u12_mph, u21_mph, u22_mph)

    # Assemble right-hand side vector
    rhs                          = assemble_rhs_un( V, fields=[u11_mph, u12_mph, u_d1])
    rhs1   = apply_dirichlet(V, rhs, dirichlet_1)    
    # Assemble right-hand side vector
    rhs                       = assemble_rhs_un( V, fields=[u21_mph, u22_mph, u_d2])
    rhs2   = apply_dirichlet(V, rhs, dirichlet_2)    
    # Assemble right-hand side vector
    b                          = zeros(Ni._nbasis[0]+Ni._nbasis[1])
    b[:Ni._nbasis[0]]                = rhs1[:]
    b[Ni._nbasis[0]:]                = rhs2[:]

    # Solve the linear system using CGS
    x, inf          = sla.cg(Ni.tosparse(), b)

    # ... Extract solution
    u1              = apply_dirichlet(V, x[:Ni._nbasis[0]], dirichlet = dirichlet_1, update= u_d1)#StencilVector(V.vector_space)
    u2              = apply_dirichlet(V, x[Ni._nbasis[0]:], dirichlet = dirichlet_2, update= u_d2)#StencilVector(V.vector_space)
    # ... to array
    x1              = u1.toarray().reshape(V.nbasis)
    x2              = u2.toarray().reshape(V.nbasis)

    # Compute L2 and H1 errors
    norm0   = assemble_norm_un(V, fields=[u11_mph, u12_mph, u1]).toarray()
    norm1   = assemble_norm_un(V, fields=[u21_mph, u22_mph, u2]).toarray()
    l2_norm = np.sqrt(norm0[0]**2+norm1[0]**2)
    H1_norm = np.sqrt(norm0[1]**2+norm1[1]**2)
    print("L2 norm:", l2_norm, "H1 norm:", H1_norm, "area: ", norm0[2], "+", norm1[2])

    return x1, x2, l2_norm, H1_norm

#------------------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------------------
#------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------
nbpts       = args.nbpts # FOR PLOT
RefinNumber = args.h    # Number of global mesh refinements
refGrid     = args.i    # Initial mesh size
degree      = [0, 0]#[args.e, args.e] Not yet TODO
if args.last:
    refGrid     = RefinNumber  # Initial mesh size
    RefinNumber = 0
table       = zeros((RefinNumber+1,7))
times       = []


print("(#=assembled Dirichlet, #=solve poisson)\n")
#------------------------------------------------------------------------------
# Define exact solution and Dirichlet boundary condition
#------------------------------------------------------------------------------
# Test 0
# g         = ['np.sin(2.*np.pi*x)*np.sin(2.*np.pi*y)']
# Test 1
#g         = ['1./(1.+np.exp((x + y  - 0.5)/0.01) )']
# Test 1
g         = ['x**2+y**2']

#------------------------------------------------------------------------------
# Load CAD geometry
#------------------------------------------------------------------------------
# geometry = load_xml('unitSquare.xml')
# idmp = (1,0)
geometry = load_xml('lshape.xml')
idmp = (0,1)
# geometry = load_xml('quart_annulus.xml')
# idmp     = (0,1)
# geometry = load_xml('annulus.xml')
# idmp     = (0,1)
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)

#------------------------------------------------------------------------------
# Extract geometry mapping
#------------------------------------------------------------------------------
mp              = getGeometryMap(geometry,idmp[0])# .. First patch 
mp1             = getGeometryMap(geometry,idmp[1])# .. Second patch
degree[0]        += mp.degree[0] # we suppose that we have same degree
degree[1]        += mp.degree[1]
mp.nurbs_check    = True # Activate NURBS if geometry uses NURBS
mp1.nurbs_check   = True # Activate NURBS if geometry uses NURBS
nb_ne             = mp.nelements[0]*2**refGrid #16  # number of elements after refinement
quad_degree       = max(degree[0],degree[1]) # Quadrature degree

#------------------------------------------------------------------------------
# Initialize mapping
#------------------------------------------------------------------------------
xmp, ymp         = mp.coefs()
wm1, wm2         = mp.weights()
xmp1, ymp1       = mp1.coefs()

#------------------------------------------------------------------------------
# Detect interface between patches
#------------------------------------------------------------------------------
rInt             = pyrefInterface(xmp, ymp, xmp1, ymp1)
rInt.printInterface() # Print detected interface

#------------------------------------------------------------------------------
# Mesh refinement loop 
#------------------------------------------------------------------------------
i_save = 0
for ne in range(refGrid,refGrid+RefinNumber+1):
    #-----------------------------------------------------------
    # Refine mesh
    #-----------------------------------------------------------
    nb_ne           = 2**ne
    print('#---IN-UNIFORM--MESH', nb_ne)

    # Refine geometry mapping
    weight, xmp, ymp  = mp.RefineGeometryMap(numElevate=nb_ne)
    wm1, wm2 = weight[:,0], weight[0,:]
    xmp1, ymp1  = mp1.RefineGeometryMap(numElevate=nb_ne)[1:]
    # Create spline spaces for refined mesh
    V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,None, numElevate=nb_ne), nderiv = 1, omega = wm1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,None, numElevate=nb_ne), nderiv = 1, omega = wm2, quad_degree = quad_degree)
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
    xd1, u_d1 = build_dirichlet(Vh, g, map = (xmp, ymp,Vh))
    xd2, u_d2 = build_dirichlet(Vh, g, map = (xmp1, ymp1,Vh))
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
    table[i_save,:]                 = [V1.degree,V2.degree, V1.nelements, V2.nelements, l2_error, H1_error, times[-1]]
    i_save                         += 1

#------------------------------------------------------------------------------
# Print error results in LaTeX table format
#------------------------------------------------------------------------------
print("Degree $p =",degree,"\n")
print("cells & $L^2$-Err & $H^1$-Err & cpu-time")
print("----------------------------------------")
for i in range(0,RefinNumber+1):
    print("",int(table[i,2]),"X", int(table[i,3]), "&", np.format_float_scientific(table[i,4], unique=False, precision=2), "&", np.format_float_scientific(table[i,5], unique=False, precision=6), "&", np.format_float_scientific(table[i,4], unique=False, precision=2))
print('\n')

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :

    from pyrefiga    import paraview_nurbsSolutionMultipatch
    solutions = [
        {"name": "Solution", "data": [xuh1, xuh2]}
    ]
    functions = [
        {"name": "Exact solution", "expression": g[0]},
    ]
    paraview_nurbsSolutionMultipatch(nbpts, [Vh, Vh], [xmp, xmp1], [ymp, ymp1],  solution = solutions, functions = functions)
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])