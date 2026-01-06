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
from   pyrefiga                         import pyrefMultpatch
from   pyrefiga                         import build_dirichlet
from   pyrefiga                         import load_xml
# Import Poisson assembly tools for uniform mesh
from gallery.gallery_section_06         import assemble_matrix_un_ex01
from gallery.gallery_section_10         import assemble_vector_un_ex02
from gallery.gallery_section_10         import assemble_norm_un_ex02

assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex02, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex02, arity=1)

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
def poisson_solve(V, VT, pyrefMP, g):

    assert isinstance( pyrefMP, pyrefMultpatch)
    assert isinstance( V,  TensorSpace)
    assert isinstance( VT, TensorSpace)

    #... Nitsche's class
    Ni            = StencilNitsche(V, V, pyrefMP, Isoparametric= False)
    # ... Assemble Nitsche's global matrix
    Ni.assembleNitsche()
    # Assemble right-hand side vector
    b             = zeros(Ni._Nitshedim[0])
    #...
    u_d           = []
    # Assemble stiffness matrix
    for patch_nb in range(1, pyrefMP.getNumPatches()+1):
        #... mapping in Stencil format
        u11_mph, u12_mph = pyrefMP.getStencilMapping(patch_nb)
        #... mapping as arrays
        xmp, ymp         = pyrefMP.getcoefs(patch_nb)
        # Assemble Dirichlet boundary conditions
        u_d1 = build_dirichlet(V, g, map = (xmp, ymp, pyrefMP.getTensorSpace()), Boundaries  = pyrefMP.getDirichletBoundaries(patch_nb))[1]
        u_d.append(u_d1)
        #...
        stiffness  = StencilMatrix(V.vector_space, V.vector_space)
        stiffness  = assemble_matrix_un(VT, fields=[u11_mph, u12_mph], out=stiffness)
        stiffness  = apply_dirichlet(V, stiffness, dirichlet = pyrefMP.getDirPatch(patch_nb))
        #...
        Ni.appendBlock(stiffness, patch_nb)
        # Assemble right-hand side vector
        rhs        = StencilVector(V.vector_space)
        rhs        = assemble_rhs_un( VT, fields=[u11_mph, u12_mph, u_d1], out= rhs)
        rhs        = apply_dirichlet(V, rhs, pyrefMP.getDirPatch(patch_nb))
        # ...
        Ni.assembleNitsche_Dirichlet(u_d1, patch_nb)
        #...
        b[Ni._rhnb[patch_nb-1]:Ni._rhnb[patch_nb]] = rhs[:]
    #=============================================
    # # # Assemble Nitsche's off diagonal matrices
    #=============================================
    # Solve the linear system using CGS
    x, inf          = sla.cg(Ni.tosparse(), b + Ni.rhs())
    l2_norm = 0.
    H1_norm = 0.
    x_sol   = []
    u_sol   = []
    # ... Extract solution
    for patch_nb in range(1,pyrefMP.getNumPatches()+1):
        u1              = apply_dirichlet(V, x[Ni._rhnb[patch_nb-1]:Ni._rhnb[patch_nb]], dirichlet = pyrefMP.getDirPatch(patch_nb), update= u_d[patch_nb-1])#StencilVector(V.vector_space)
        # ... to array
        x1              = u1.toarray().reshape(V.nbasis)
        x_sol.append(x1)
        u_sol.append(u1)
        #... mapping in Stencil format
        u11_mph, u12_mph = pyrefMP.getStencilMapping(patch_nb)
        # Compute L2 and H1 errors
        Norm      = StencilVector(V.vector_space)
        Norm      = assemble_norm_un(VT, fields=[u11_mph, u12_mph, u1], out= Norm).toarray()
        l2_norm += Norm[0]**2
        H1_norm += Norm[1]**2
    l2_norm = np.sqrt(l2_norm)
    H1_norm = np.sqrt(H1_norm)    
    print("L2 norm:", l2_norm, "H1 norm:", H1_norm)

    return x_sol, l2_norm, H1_norm

#------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------
nbpts       = args.nbpts # FOR PLOT
RefinNumber = args.h    # Number of global mesh refinements
refGrid     = args.i    # Initial mesh size
degree      = [args.e, args.e] #Not yet TODO
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
geometry = load_xml('unitSquare.xml')
idmp = (0,1)
# geometry = load_xml('lshape.xml')
# idmp = (0,1)
# geometry = load_xml('quart_annulus.xml')
# idmp     = (0,1)
# geometry = load_xml('annulus.xml')
# idmp     = (0,1,2,3)
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)

#------------------------------------------------------------------------------
# Extract geometry mapping
#------------------------------------------------------------------------------
pyrefMP           = pyrefMultpatch(geometry,idmp)# .. First patch 
degree[0]        += pyrefMP.degree()[0]
degree[1]        += pyrefMP.degree()[1]
nb_ne             = refGrid #16  # number of elements after refinement
quad_degree       = max(degree[0],degree[1]) # Quadrature degree
#... Print multipatch info
pyrefMP.detail()

#------------------------------------------------------------------------------
# Mesh refinement loop 
#------------------------------------------------------------------------------
i_save = 0
for ne in range(refGrid,refGrid+RefinNumber+1):
    #-----------------------------------------------------------
    # Refine mesh
    #-----------------------------------------------------------
    nb_ne           = ne
    print('#---IN-UNIFORM--MESH', nb_ne)

    # Refine geometry mapping
    # Create spline spaces for refined mesh
    V1 = SplineSpace(degree=degree[0], grid = pyrefMP.getRefinegrid(0, numElevate=nb_ne), quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = pyrefMP.getRefinegrid(1, numElevate=nb_ne), quad_degree = quad_degree)
    # ... mapping spaces
    V1mp, V2mp = pyrefMP.UnifSplineSpace(mesh=(V1.mesh, V2.mesh), quad_degree= (quad_degree,quad_degree), nders=1)
    Vh = TensorSpace(V1, V2)
    VT = TensorSpace(V1, V2, V1mp, V2mp)
    print('#spaces')
    # Solve Poisson equation on refined mesh
    start = time.time()
    xuh, l2_error,  H1_error = poisson_solve(Vh, VT, pyrefMP, g)
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
    print("",int(table[i,2]),"X", int(table[i,3]), "&", np.format_float_scientific(table[i,4], unique=False, precision=2), "&", np.format_float_scientific(table[i,5], unique=False, precision=6), "&", np.format_float_scientific(table[i,6], unique=False, precision=2))
print('\n')

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :

    from pyrefiga    import paraview_nurbsSolutionMultipatch
    solutions = [
        {"name": "Solution", "data": xuh}
    ]
    functions = [
        {"name": "Exact solution", "expression": g[0]},
    ]
    paraview_nurbsSolutionMultipatch(nbpts, Vh, pyrefMP.getAllcoefs('x'), pyrefMP.getAllcoefs('y'), Vg = pyrefMP.getTensorSpace(),  solution = solutions, functions = functions)
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])