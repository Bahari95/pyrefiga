"""
poisson3d_example.py TODO 

Example: Solving Poisson's Equation on a 2D complex geometry using B-spline or NURBS representation.

Author: M. Bahari
"""
from   pyrefiga                    import compile_kernel
from   pyrefiga                    import apply_dirichlet

from   pyrefiga                    import SplineSpace
from   pyrefiga                    import TensorSpace
from   pyrefiga                    import StencilMatrix
from   pyrefiga                    import StencilVector
from   pyrefiga                    import pyccel_sol_field_2d
from   pyrefiga                    import sol_field_NURBS_2d
from   pyrefiga                    import paraview_nurbsSolutionMultipatch
from   pyrefiga                    import getGeometryMap
from   pyrefiga                    import load_xml
from   pyrefiga                    import build_dirichlet

# Import Poisson assembly tools for uniform mesh
from gallery.gallery_section_06    import assemble_matrix_un_ex01
from gallery.gallery_section_06    import assemble_vector_un_ex01
from gallery.gallery_section_06    import assemble_norm_un_ex01

assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros
import numpy                        as     np
import timeit
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
# Poisson solver algorithm
#------------------------------------------------------------------------------
def poisson_solve(V, VT, u11_mph, u12_mph, u_d):
    u   = StencilVector(V.vector_space)

    # Assemble stiffness matrix
    stiffness  = StencilMatrix(V.vector_space, V.vector_space)
    stiffness  = assemble_matrix_un(VT, fields=[u11_mph, u12_mph], out=stiffness)
    stiffness1 = apply_dirichlet(V, stiffness)

    # Assemble right-hand side vector
    rhs        = StencilVector(V.vector_space)
    rhs        = assemble_rhs_un( VT, fields=[u11_mph, u12_mph, u_d], out= rhs)

    #... apply Dirichlet
    stiffness  = apply_dirichlet(V, stiffness)
    rhs        = apply_dirichlet(V, rhs)

    # Solve linear system
    lu         = sla.splu(csc_matrix(stiffness1))
    x          = lu.solve(rhs)

    # Apply Dirichlet boundary conditions
    u          = apply_dirichlet(V, x, update = u_d)
    x_s        = u.toarray().reshape(V.nbasis)

    # Compute L2 and H1 errors
    Norm      = StencilVector(V.vector_space)
    Norm      = assemble_norm_un(VT, fields=[u11_mph, u12_mph, u], out= Norm)
    norm      = Norm.toarray()
    l2_norm   = norm[0]
    H1_norm   = norm[1]
    return u, x_s, l2_norm, H1_norm, 0.

#------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------
nbpts       = args.nbpts # FOR PLOT
RefinNumber = args.h    # Number of global mesh refinements
refGrid     = args.i    # Initial mesh size
degree      = [args.e, args.e]
if args.last:
    refGrid     = RefinNumber  # Initial mesh size
    RefinNumber = 0
table       = zeros((RefinNumber+1,8))
times       = []

print("(#=spaces, #=assembled Dirichlet, #=solve poisson)\n")

#------------------------------------------------------------------
# Load CAD geometry
#------------------------------------------------------------------
#geometry = load_xml('unitSquare.xml')
geometry = load_xml('rotor_2d.xml')
# geometry = load_xml('circle.xml')
# geometry = load_xml('quart_annulus.xml')
# geometry = load_xml('annulus.xml')
id_mp    = 0
print('#---Poisson equation', geometry)

#------------------------------------------------------------------
# Define exact solution and Dirichlet boundary condition
#------------------------------------------------------------------
# Test 0
# g         = ['np.sin(2.*np.pi*x)*np.sin(2.*np.pi*y)']
# Test 1
g         = ['x**2+y**2']
print("Dirichlet boundary conditions", g)

#------------------------------------------------------------------
#... extract geometry from xml file
#------------------------------------------------------------------
geometry         = load_xml(geometry)  # Load geometry from pyrefiga
# ... Assembling mapping
mp               = getGeometryMap(geometry,id_mp)
degree[0]        += mp.degree[0]
degree[1]        += mp.degree[1]
mp.nurbs_check   = True # Activate NURBS if geometry uses NURBS
nb_ne            = 2**refGrid # number of elements after refinement
quad_degree      = max(degree[0],degree[1]) # Quadrature degree
# ... Assembling mapping
xmp, ymp         = mp.coefs()
wm1, wm2         = mp.weights()

# Create spline spaces for each direction for mapping
V1mp            = SplineSpace(degree=mp.degree[0], grid = mp.grids[0], omega = wm1, quad_degree = quad_degree)
V2mp            = SplineSpace(degree=mp.degree[1], grid = mp.grids[1], omega = wm2, quad_degree = quad_degree)
Vmp             = TensorSpace(V1mp, V2mp)
# ...
u11_mp          = StencilVector(Vmp.vector_space)
u12_mp          = StencilVector(Vmp.vector_space)
u11_mp.from_array(Vmp, xmp)
u12_mp.from_array(Vmp, ymp)

i_save = 0
for ne in range(refGrid,refGrid+RefinNumber+1):
    #-----------------------------------------------------------
    # Refine mesh
    #-----------------------------------------------------------
    nb_ne           = 2**ne
    print('#---IN-UNIFORM--MESH', nb_ne)

    #-----------------------------------------------------------
    # Create spline spaces for refined mesh
    #-----------------------------------------------------------
    V1              = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0, numElevate=nb_ne), quad_degree = quad_degree)
    V2              = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1, numElevate=nb_ne), quad_degree = quad_degree)
    Vh              = TensorSpace(V1, V2)

    #-----------------------------------------------------------
    # Create spline spaces for each direction for mapping (compute basis in new integral grid)
    #-----------------------------------------------------------
    V1mp            = SplineSpace(degree=mp.degree[0], grid = mp.grids[0], omega = wm1, mesh =  V1.mesh, quad_degree = quad_degree)
    V2mp            = SplineSpace(degree=mp.degree[1], grid = mp.grids[1], omega = wm2, mesh =  V2.mesh, quad_degree = quad_degree)
    VT              = TensorSpace(V1, V2, V1mp, V2mp)
    print('#spces; VT.nelements:',VT.nelements)
    #-----------------------------------------------------------
    # Assemble Dirichlet boundary conditions
    #-----------------------------------------------------------
    u_d   = build_dirichlet(Vh, g, map = (xmp, ymp, Vmp))[1]
    print('#')

    #-----------------------------------------------------------
    # Solve Poisson equation on refined mesh
    #-----------------------------------------------------------
    start = time.time()
    u, xuh, l2_error,  H1_error, cond  = poisson_solve(Vh, VT, u11_mp, u12_mp, u_d)
    times.append(time.time()- start)
    print('#')

    #-----------------------------------------------------------
    # Store results
    #-----------------------------------------------------------
    table[i_save,:]                     = [V1.degree,V2.degree, V1.nelements, V2.nelements, l2_error, H1_error, times[-1], cond]
    i_save                             += 1

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
if args.plot :
    # -----------------------------END OF THE SHARED PART FOR ALL GEOMETRY
    #---Solution in uniform mesh
    un = pyccel_sol_field_2d((nbpts,nbpts),  xuh , Vh.knots, Vh.degree)[0]
    #---Compute a mapping
    x = sol_field_NURBS_2d((nbpts,nbpts),  xmp, Vmp.omega, Vmp.knots, Vmp.degree)[0]
    y = sol_field_NURBS_2d((nbpts,nbpts),  ymp, Vmp.omega, Vmp.knots, Vmp.degree)[0]
    # ...
    Sol_un = eval(g[0])

    #---Compute a solution
    functions = [
            {"name": "exact_solution", "expression": g[0]},
    ]
    precomputed = [
            {"name": "numerical_solution", "data": [un]},
            {"name": "Error", "data": [np.absolute(un-Sol_un)]},
    ]
    paraview_nurbsSolutionMultipatch(nbpts, [Vh], [xmp], [ymp], Vg = [Vmp], functions = functions, precomputed= precomputed,filename = 'figs/poisson_un_2dexample')
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/poisson_un_2dexample.vtm"])

print("End")