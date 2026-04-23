"""
# porous_medium.py is a test file for a convection in porous medium
    
    This model describes the buoyancy-driven horizontal spreading of heat and chemical species through a fluid saturated
porous medium.
D. A. Neid and A. Bejan. Convection in Porous Media. pringer-Verlag, New York, 1992

author: M Bahari
"""
from   pyrefiga                    import compile_kernel
from   pyrefiga                    import apply_dirichlet
from   pyrefiga                    import SplineSpace
from   pyrefiga                    import StencilMatrix
from   pyrefiga                    import TensorSpace
from   pyrefiga                    import StencilVector
from   pyrefiga                    import load_xml
from   pyrefiga                    import pyref_multipatch
from   pyrefiga                    import plot_results, compute_eoc
#-----------
from   pyrefiga                    import paraview_TimeSolutionMultipatch
from   pyrefiga                    import pyccel_sol_field_2d
from   pyrefiga                    import eval_bsplines, StencilNitsche

# Import convection in porous medium assembly tools for uniform mesh
from gallery.gallery_section_00             import assemble_matrix_un_ex01
from gallery.gallery_section_00             import assemble_vector_un_ex01
from gallery.gallery_section_00             import assemble_norm_un_ex01
assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

# Import convection in porous medium assembly tools for uniform mesh ..
from gallery.gallery_section_04    import assemble_matrix_diff_ex01
from gallery.gallery_section_04    import assemble_vector_diff_ex01
from gallery.gallery_section_04    import assemble_vector_mass_ex01

assemble_matrix_diff   = compile_kernel(assemble_matrix_diff_ex01, arity=2)
assemble_rhs_diff      = compile_kernel(assemble_vector_diff_ex01, arity=1)
assemble_rhs_mass      = compile_kernel(assemble_vector_mass_ex01, arity=1)

from gallery.gallery_section_04         import assemble_matrix_DiffSpacediagnitsche
from gallery.gallery_section_04         import assemble_matrix_DiffSpaceoffdiagnitsche
#.
assemble_nitscheDiag        = compile_kernel(assemble_matrix_DiffSpacediagnitsche, arity=2)
assemble_nitscheUnderDiag   = compile_kernel(assemble_matrix_DiffSpaceoffdiagnitsche, arity=2)

from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi
import numpy                        as     np
import time
import argparse

#------------------------------------------------------------------------------
# Create directory for figures if it doesn't exist
#------------------------------------------------------------------------------
import os
os.makedirs("figs", exist_ok=True)

def assemble_Nitesche_terms(Ni, dt):
    assert isinstance(Ni, StencilNitsche), 'EXPECT StencilNitsche'
    # assert isinstance(pyref_MP, pyref_multipatch), 'EXPECT pyref_multipatch'
    '''
    Docstring pour assemble_nitsche for Laplace operator
    assemble global Nitsche matrix in uniform mesh

    :param self: Description
    '''
    stab         = 4.*( Ni._domain.degree[0] + Ni._domain.dim ) *( Ni._domain.degree[1] + Ni._domain.dim ) * ( Ni._domain.degree[0] + 1 )
    m_h          = (Ni._domain.nbasis[0]*Ni._domain.nbasis[1])
    Kappa        = dt*2.*stab*m_h*1e+6
    normS        = dt*Ni.normS
    # ...
    for patch_nb in range(1,Ni.mp.nb_patches+1):
        #==================================
        # ... Diagonal nitsche matrix part
        #==================================
        # ... initial diag stiffness matrix
        stiffness  = StencilMatrix(Ni._domain.vector_space, Ni._domain.vector_space)
        # ... assemble 
        # assemble mappings for patches
        u11_mph, u12_mph = Ni.mp.stencil_mapping(patch_nb)
        #... get interfaces for a given patch
        interfaces_like = Ni.mp.getInterfacePatch(patch_nb)
        # ... assemble diagonal matrix
        assemble_nitscheDiag(Ni._Alldomain, fields=[u11_mph, u12_mph], knots=True, value=[Ni._mpdomain.omega[0],Ni._mpdomain.omega[1], interfaces_like, Kappa, normS], out = stiffness)
        # ... apply dirichlet
        stiffness  = apply_dirichlet(Ni._domain, stiffness, dirichlet = Ni.mp.getDirPatch(patch_nb))
        assert not np.isnan(stiffness.data).any(), "Sparse matrix contains NaNs"
        # ... assemble it into globel matrix
        Ni.append_block(stiffness, patch_nb)
    # ...Ni.add_nitsche_off_diag()
    for interface in Ni.mp.getInterfaces():
        #======================================
        # ... Off diagonal nitsche matrix part
        #======================================
        # ... get interface and patch numbers
        patch_nb         = interface[0] 
        patch_nb_n       = interface[1]
        # ... get interface mappings
        interface_like   = interface[2][0]
        # assemble mappings for patches
        u11_mph, u12_mph = Ni.mp.stencil_mapping(patch_nb)
        u21_mph, u22_mph = Ni.mp.stencil_mapping(patch_nb_n)
        #... assemble off diagonal matrix
        stiffnessoffdiag = StencilMatrix(Ni._domain.vector_space, Ni._domain.vector_space)
        assemble_nitscheUnderDiag(Ni._Alldomain, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, 
                                        value=[Ni._mpdomain.omega[0],Ni._mpdomain.omega[1], interface_like, Kappa, normS], 
                                        out = stiffnessoffdiag)
        #... correct coo matrix
        stiffnessoffdiag = Ni.collect_offdiag_stencil_matrix(stiffnessoffdiag, interface)
        Ni.append_block(stiffnessoffdiag, patch_nb_n, patch_nb)
        Ni.append_block(stiffnessoffdiag.T, patch_nb, patch_nb_n)
    #...
    return 0

#------------------------------------------------------------------------------
# convection in porous medium solver algorithm
#------------------------------------------------------------------------------
class convection_in_porous_medium(object):
    
    def __init__(self, V, pyref_MP, Le, Ra, ratioPH, dt, x0=0., y0=0.):
        assert isinstance(pyref_MP, pyref_multipatch), "pyref_MP must be a pyref_patch instance"
        assert isinstance(V, TensorSpace), "only accept TensorSpace"

        #...+++++++++++++++++++++++++++++
        VT            = pyref_MP.getspace(V)
        #... Nitsche's class assemble stream matrix
        Ni            = StencilNitsche(V, VT, pyref_MP)
        # ... Assemble Nitsche's global matrix
        assemble_Nitesche_terms(Ni, 1.)
        # Assemble stiffness matrix
        for patch_nb in range(1, pyref_MP.nb_patches+1):
            #... mapping in Stencil format
            u11_mph, u12_mph = pyref_MP.stencil_mapping(patch_nb)
            #...
            stiffness  = StencilMatrix(V.vector_space, V.vector_space)
            stiffness  = assemble_matrix_un(VT, fields=[u11_mph, u12_mph], value=[1], out=stiffness)
            stiffness  = apply_dirichlet(V, stiffness, dirichlet = pyref_MP.getDirPatch(patch_nb))
            # print("shape in ", patch_nb, "is", stiffness.shape)
            #...
            Ni.append_block(stiffness, patch_nb)
            # ...
        #=============================================
        # # # Assemble Nitsche's off diagonal matrices
        #=============================================
        M            = Ni.nitsche_merge()
        # x, inf  = sla.cg(M, b, rtol=1e-30)
        self.Ni      = Ni
        self.lu      = sla.splu(csc_matrix(M))
        # ...
        self.V       = V
        self.pyref_MP= pyref_MP
        self.x0      = x0 
        self.y0      = y0
        self.x_last  = 0.
        self.dt      = dt
        self.Le      = Le
        self.Ra      = Ra
        self.ratioPH = ratioPH  
    def project(self, u_sol = None, time = 0.):
        '''
        Project field into the physical space
        '''
        V = self.V
        assert isinstance( self.pyref_MP, pyref_multipatch)
        assert isinstance( V,  TensorSpace)
        pyrefMP       = self.pyref_MP.clone(Dirichlet_all=False) # Clone multipatch for visualization with all patches having Dirichlet BCs

        #... space FE&Mapping
        VT            = pyrefMP.getspace(V) 
        #... Nitsche's class
        Ni            = StencilNitsche(V, VT, pyrefMP)
        # Assemble stiffness matrix 
        for patch_nb in range(1, pyrefMP.nb_patches+1):
            #... mapping in Stencil format
            u11_mph, u12_mph = pyrefMP.stencil_mapping(patch_nb)
            # Assemble Dirichlet boundary conditions
            #...
            stiffness  = StencilMatrix(V.vector_space, V.vector_space)
            stiffness  = assemble_matrix_un(VT, fields=[u11_mph, u12_mph], value=[0], out=stiffness)
            stiffness  = apply_dirichlet(V, stiffness, dirichlet = False)
            # print("shape in ", patch_nb, "is", stiffness.shape)
            #...
            Ni.append_block(stiffness, patch_nb)
            # Assemble right-hand side vector
            rhs        = StencilVector(V.vector_space)
            rhs        = assemble_rhs_un( VT, fields=[u11_mph, u12_mph], value = [self.x0, self.y0], out= rhs)
            rhs        = apply_dirichlet(V, rhs, dirichlet = False)
            # print("shape in ", patch_nb, "is", rhs.shape)
            # ...
            Ni.assemble_nitsche_dirichlet(rhs, patch_nb, False)
            # ...
        #=============================================
        # # # Assemble Nitsche's off diagonal matrices
        #=============================================
        M       = Ni.nitsche_merge()
        b       = Ni.nitsche_merge_rhs()
        # print("after merge", b.shape, b)
        lu         = sla.splu(csc_matrix(M))
        x          = lu.solve(b)
        l2_norm = 0.
        H1_norm = 0.
        l1_norm = 0.
        u_sol   = []
        # ... Extract solution
        for patch_nb in range(1,pyrefMP.nb_patches+1):
            # ... extract solution
            u1              = Ni.extract_sol(x, patch_nb)
            # ...
            u_sol.append(u1)
            #... mapping in Stencil format
            u11_mph, u12_mph = pyrefMP.stencil_mapping(patch_nb)
            # Compute L2 and H1 errors
            Norm      = StencilVector(V.vector_space)
            Norm      = assemble_norm_un(VT, fields=[u11_mph, u12_mph, u1], value=[self.x0, self.y0, time], out= Norm).toarray()
            l2_norm += Norm[0]**2
            H1_norm += Norm[1]**2
            l1_norm += Norm[2]
        l2_norm = np.sqrt(l2_norm)
        H1_norm = np.sqrt(H1_norm)    
        return u_sol, l2_norm, H1_norm, l1_norm
    def stream(self, u_t, u_c, Rbuoyancy = 0., ttime = 0.):
        V                = self.V
        # ...
        assert isinstance( self.pyref_MP, pyref_multipatch)
        assert isinstance( V,  TensorSpace)
        # ...
        u_d              = [StencilVector(V.vector_space) for _ in range(self.pyref_MP.nb_patches)]
        # ...
        VT            = self.pyref_MP.getspace(V)
        #... Nitsche's class
        self.Ni.b_dir  *= 0. # set rhs to zero
        for patch_nb in range(1, self.pyref_MP.nb_patches+1):
            #... mapping in Stencil format
            u11_mph, u12_mph = self.pyref_MP.stencil_mapping(patch_nb)
            # Assemble Dirichlet boundary conditions
            u_d1 = u_d[patch_nb-1]
            # Assemble right-hand side vector
            rhs        = StencilVector(V.vector_space)
            rhs        = assemble_rhs_diff( VT, fields=[u11_mph, u12_mph, u_d1, u_t[patch_nb-1], u_c[patch_nb-1]], value= [self.Ra, Rbuoyancy], out= rhs)
            rhs        = apply_dirichlet(V, rhs, dirichlet = self.pyref_MP.getDirPatch(patch_nb))
            # ...
            self.Ni.assemble_nitsche_dirichlet(rhs, patch_nb)
            # ...
        #=============================================
        # # # Assemble Nitsche's off diagonal matrices
        #=============================================
        b       = self.Ni.nitsche_merge_rhs()
        # x, inf  = sla.cg(M, b, rtol=1e-30)
        x          = self.lu.solve(b)
        l2_norm = 0.
        H1_norm = 0.
        l1_norm = 0.
        u_str   = []
        # ... Extract solution
        for patch_nb in range(1,self.pyref_MP.nb_patches+1):
            # ... extract solution
            u1              = self.Ni.extract_sol(x, patch_nb)
            # ...
            u_str.append(u1)
            #... mapping in Stencil format
            u11_mph, u12_mph = self.pyref_MP.stencil_mapping(patch_nb)
            # Compute L2 and H1 errors
            Norm      = StencilVector(V.vector_space)
            Norm      = assemble_norm_un(VT, fields=[u11_mph, u12_mph, u1], value=[self.x0, self.y0, ttime], out= Norm).toarray()
            l2_norm += Norm[0]**2
            H1_norm += Norm[1]**2
            l1_norm += Norm[2]
        l2_norm = np.sqrt(l2_norm)
        H1_norm = np.sqrt(H1_norm)    
        return u_str, l2_norm, H1_norm, l1_norm
    def temperature(self, u_tmp, u_sol, ttime = 0.):
        V                = self.V
        # ...
        assert isinstance( self.pyref_MP, pyref_multipatch)
        assert isinstance( V,  TensorSpace)
        # ...
        pyrefMP       = self.pyref_MP.clone(Dirichlet_all=False) # Clone multipatch for visualization with all patches having Dirichlet BCs
        VT            = pyrefMP.getspace(V)
        #... Nitsche's class
        # ... Assemble Nitsche's global matrix
        Ni            = StencilNitsche(V, VT, pyrefMP)
        # ... Assemble Nitsche's global matrix
        assemble_Nitesche_terms(Ni, self.dt)
        for patch_nb in range(1, pyrefMP.nb_patches+1):
            #... mapping in Stencil format
            u11_mph, u12_mph = pyrefMP.stencil_mapping(patch_nb)
            # print("shape in ", patch_nb, "is", stiffness.shape)
            stiffness  = StencilMatrix(V.vector_space, V.vector_space)
            stiffness  = assemble_matrix_diff(VT, fields=[u_sol[patch_nb-1], u11_mph, u12_mph], value= [self.dt, self.Le, self.ratioPH], out=stiffness)
            stiffness  = apply_dirichlet(V, stiffness, dirichlet = pyrefMP.getDirPatch(patch_nb))
            #...
            Ni.append_block(stiffness, patch_nb)
            # Assemble right-hand side vector
            rhs        = StencilVector(V.vector_space)
            rhs        = assemble_rhs_mass( VT, fields=[u11_mph, u12_mph, u_tmp[patch_nb-1]], out= rhs)
            rhs        = apply_dirichlet(V, rhs, dirichlet = pyrefMP.getDirPatch(patch_nb))
            # print("shape in ", patch_nb, "is", rhs.shape)
            # ...
            Ni.assemble_nitsche_dirichlet(rhs, patch_nb)
            # ...
        #=============================================
        # # # Assemble Nitsche's off diagonal matrices
        #=============================================
        M       = Ni.nitsche_merge()
        b       = Ni.nitsche_merge_rhs()
        # x, inf  = sla.gmres(M, b, rtol=1e-30, x0=self.x_last)
        lu         = sla.splu(csc_matrix(M))
        x          = lu.solve(b)
        # self.x_last = x
        l2_norm = 0.
        H1_norm = 0.
        l1_norm = 0.
        u_sol   = []
        # ... Extract solution
        for patch_nb in range(1,pyrefMP.nb_patches+1):
            # ... extract solution
            u1              = Ni.extract_sol(x, patch_nb)
            # ...
            u_sol.append(u1)
            #... mapping in Stencil format
            u11_mph, u12_mph = pyrefMP.stencil_mapping(patch_nb)
            # Compute L2 and H1 errors
            Norm      = StencilVector(V.vector_space)
            Norm      = assemble_norm_un(VT, fields=[u11_mph, u12_mph, u1], value=[self.x0, self.y0, ttime], out= Norm).toarray()
            l2_norm += Norm[0]**2
            H1_norm += Norm[1]**2
            l1_norm += Norm[2]
        l2_norm = np.sqrt(l2_norm)
        H1_norm = np.sqrt(H1_norm)    
        return u_sol, l2_norm, H1_norm, l1_norm

#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()

#------------------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------------------
Le          = 1.
ratioPH     = 1.
Ra          = 1e3
nbpts       = 75     # Number of points for plotting
Ntimes      = 100    # Number of time step
RefinNumber = 5      # Number of global mesh refinements
nbRefineNbr = 0      # Number of global mesh refinements loop
degree      = [1,0]  # Elevate Degree of the spline space
Ltime0      = 0.
time_max    = 1.#2.*pi
dt          = 1e-5
N_plot      = 5
table       = zeros((nbRefineNbr+1,7))
LStimes     = []
sol_app     = []
sol_ex      = []
error_i     = []
error_l2    = []
error_h1    = []
error_mass  = []
x0, y0      = 1.5, 0.
print("(#=assembled Dirichlet, #=solve convection in porous medium)\n")

#------------------------------------------------------------------------------
# Define exact solution and Dirichlet boundary condition
#------------------------------------------------------------------------------
# Test 0
# g         = ['np.sin(2.*np.pi*x)*np.sin(2.*np.pi*y)']
# Test 1
g       = ['0.*x+0.*y']
# g       = ['10.*np.exp(-100.*((x-1.5)**2+(y-0.0)**2))']
# Test 2
# g       = ['np.exp(0 - ((x**2 + y**2 - 1.5**2)**2)/0.01)']

# ...
f_ex    = lambda x,y, t: 0. *x + 0. * y #10.*np.exp(-100.*((x*cos(t)+y*sin(t)-x0)**2+(y*cos(t)-x*sin(t)-y0)**2))
g0      = lambda x,y : 0. *x + 0. * y #10.*np.exp(-100.*((x-x0)**2+(y-y0)**2))
# ...
# f_ex    = lambda x,y, t: np.exp(t - ((x**2 + y**2 - 1.5**2)**2)/0.01)
# g0      = lambda x,y : np.exp(0. - ((x**2 + y**2 -1.5**2)**2)/0.01)

#------------------------------------------------------------------------------
# Load CAD geometry
#------------------------------------------------------------------------------
geometry = load_xml('ushape.xml')

id_mp    = [0,1,2]
print('#---IN-UNIFORM--MESH-convection in porous medium equation', geometry)
print("Dirichlet boundary conditions", g)

# Extract geometry mapping
pyref_MP       = pyref_multipatch(geometry,id_mp)
degree[0]     += pyref_MP.degree[0] # Add degree from geometry
degree[1]     += pyref_MP.degree[1]
#...
print(degree)
pyref_MP.detail()

quad_degree    = max(degree[0],degree[1])**2 # Quadrature degree

for nbRefine in range(nbRefineNbr+1):
    Elevatene   = RefinNumber+nbRefine
    #... Initialize lists to store results
    Ltime       = Ltime0
    LStimes     = []
    sol_app     = []
    str_app     = []
    sol_ex      = []
    error_i     = []
    error_l2    = []
    error_h1    = []
    error_mass  = []
    #------------------------------------------------------------------------------
    # Initialize FE space
    #------------------------------------------------------------------------------
    # Create spline spaces for each direction
    V1 = SplineSpace(degree=degree[0], grid = pyref_MP.Refinegrid(0,numElevate=Elevatene), nderiv = 1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = pyref_MP.Refinegrid(1,numElevate=Elevatene), nderiv = 1, quad_degree = quad_degree)
    # Create tensor product space
    Vh = TensorSpace(V1, V2)
    # ...
    u_str = [StencilVector(Vh.vector_space) for _ in range(pyref_MP.nb_patches)]

    print('#nbasis: ', Vh.nbasis)

    #------------------------------------------------------------------------------
    # Mesh refinement loop
    #------------------------------------------------------------------------------
    AN = convection_in_porous_medium(Vh, pyref_MP, Le, Ra, ratioPH, dt)
    # ...
    nt = 0
    # Project initial solution
    u_tmp, l2_error,  H1_error, mass_error = AN.project()
    # ...  stream function
    u_str, l2_error,  H1_error, mass_error = AN.stream(u_tmp, u_tmp)
    print(f"Step {nt}: Time = {Ltime:.4e} s | L2 Error = {l2_error:.4e} | H1 Error = {H1_error:.4e} | l1 mass = {mass_error:.4e}")
    # Store results
    x_ap    = []
    x_str   = []
    x_ex    = []
    x_err   = []
    for i in range(pyref_MP.nb_patches):
        x_ap.append( pyccel_sol_field_2d((nbpts, nbpts), u_tmp[i].tensor,  Vh.knots, Vh.degree)[0])
        x_str.append( pyccel_sol_field_2d((nbpts, nbpts), u_str[i].tensor,  Vh.knots, Vh.degree)[0] )
        x, y  = pyref_MP.eval(patch_nb = i+1, nbpts = (nbpts, nbpts))
        x_ex.append( f_ex(x,y, Ltime))
        x_err.append( np.absolute(x_ap[-1]-x_ex[-1]) )
    # ...
    sol_app.append(x_ap)
    str_app.append(x_str)
    sol_ex.append(x_ex)
    LStimes.append(Ltime)
    error_i.append(x_err)
    error_l2.append(l2_error)
    error_h1.append(H1_error)
    error_mass.append(mass_error)

    nt = 1
    while (Ltime <time_max and nt <=Ntimes):
        # ...
        Ltime     += dt
        # ...
        #print('#')
        # Solve convection in porous medium equation on refined mesh
        start = time.time()
        # ...  temperature
        u_tmp, l2_error,  H1_error, mass_error = AN.temperature( u_tmp, u_str, ttime= Ltime)
        # ...  stream function
        u_str, l2_error,  H1_error, mass_error = AN.stream(u_tmp, u_tmp)
        #print('#')
        print(f"Step {nt}: Time = {Ltime:.4e} s | L2 Error = {l2_error:.4e} | H1 Error = {H1_error:.4e} | l1 mass = {mass_error:.4e}")
        # ...
        if nt%N_plot==0:
            # Store results
            x_ap    = []
            x_str   = []
            x_ex    = []
            x_err   = []
            for i in range(pyref_MP.nb_patches):
                x_ap.append( pyccel_sol_field_2d((nbpts, nbpts), u_tmp[i].tensor,  Vh.knots, Vh.degree)[0])
                x_str.append( pyccel_sol_field_2d((nbpts, nbpts), u_str[i].tensor,  Vh.knots, Vh.degree)[0] )
                x, y =  pyref_MP.eval(patch_nb = i+1, nbpts = (nbpts, nbpts))
                x_ex.append( f_ex(x,y, Ltime))
                x_err.append( np.absolute(x_ap[-1]-x_ex[-1]) )
            # ...
            sol_app.append(x_ap)
            str_app.append(x_str)
            sol_ex.append(x_ex)
            LStimes.append(Ltime)
            error_i.append(x_err)
            error_l2.append(l2_error)
            error_h1.append(H1_error)
            error_mass.append(mass_error)

            # plot_MeshMultipatch(nbpts, [Vh], [xuh1], [xuh2])
        #..
        nt += 1
    table[nbRefine,:]                     = [Vh.nelements[0], Vh.nelements[1], Ltime, l2_error, H1_error, np.max(error_i[-1]), mass_error]
    print(f"Step {nt}: Time = {table[nbRefine,2]:.4e} s | L2 Error = {table[nbRefine,3]:.4e} | H1 Error = {table[nbRefine,4]:.4e} | l1 mass = {table[nbRefine,6]:.4e}")

#np.savetxt('figs/table_un.txt', table, fmt='%.20e')
with open('figs/table.txt', 'a') as f:
    f.write("\n# New data block---------------\n")
    np.savetxt(f, table, fmt='%.20e')

#------------------------------------------------------------------------------
# Print error results in LaTeX table format
#------------------------------------------------------------------------------
print("Degree $p =",Vh.degree,"\n")
print("Cells & Time & $L^2$-Err & eoc & $H^1$-Err & eoc & $L^infty$-Err & eoc & Mass-err")
print("----------------------------------------")
erocl2 = compute_eoc(table[:, 3])
eroch1 = compute_eoc(table[:, 4])
eroci1 = compute_eoc(table[:, 5])
# extract values first
for i in range(0,nbRefineNbr+1):
    rows, cols = int(table[i, 0]), int(table[i, 1])
    val1  = np.format_float_scientific(table[i, 2], unique=False, precision=2)
    val2  = np.format_float_scientific(table[i, 3], unique=False, precision=2)
    val22 = np.format_float_scientific(erocl2[i], unique=False, precision=2)
    val3  = np.format_float_scientific(table[i, 4], unique=False, precision=2)  # if intentional repeat
    val33 = np.format_float_scientific(eroch1[i], unique=False, precision=2)
    val4  = np.format_float_scientific(table[i, 5], unique=False, precision=2)
    val44 = np.format_float_scientific(eroci1[i], unique=False, precision=2)
    val5  = np.format_float_scientific(table[i, 6], unique=False, precision=2)
    # use f-string
    print(f"{rows}X{cols} & {val1} & {val2} & {val22} & {val3} & {val33} & {val4} & {val44} & {val5}")
print('\n')

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
plot_results([LStimes], [error_l2], i = 0, xlabel = '$\mathbf{Time}$', ylabel = '$\mathbf{L^2-error}$', legend = False,  mylocname='figs/L2_error_Conv_order')
#...
plot_results([LStimes], [error_h1], i = 3, xlabel = '$\mathbf{Time}$', ylabel = '$\mathbf{H^1-error}$', legend = False, mylocname='figs/H1_error_Conv_order')
#...
plot_results([LStimes], [error_mass], i = 4, xlabel = '$\mathbf{Time}$', ylabel = '$\mathbf{Mass-error}$', legend = False, mylocname='figs/mass_conservation_error_Conv_order')

# ...
if args.plot:
    precomputed = [
        {"name": "temperature", "data": sol_app},
        {"name": "stream function", "data": str_app},
        {"name": "Exact solution", "data": sol_ex},
        {"name": "Error distribution", "data": error_i},
    ]

    paraview_TimeSolutionMultipatch(nbpts, pyref_MP, LStime=LStimes, precomputed = precomputed, plot=args.plot)
