"""
MFMAE_fast_solver.py

# mixed variational formulation fast diagonalization solver for Monge-Ampere equation

@author : M. BAHARI
"""
from   pyrefiga                    import compile_kernel
from   pyrefiga                    import SplineSpace
from   pyrefiga                    import TensorSpace
from   pyrefiga                    import StencilMatrix
from   pyrefiga                    import StencilVector
from   pyrefiga                    import pyccel_sol_field_2d 
from   pyrefiga                    import sol_field_NURBS_2d
from   pyrefiga                    import quadratures_in_admesh
from   pyrefiga                    import paraview_nurbsAdMeshMultipatch
#.. Prologation by knots insertion matrix
from   pyrefiga                    import prolongation_matrix
# ... Using Kronecker algebra accelerated with Pyccel
from   pyrefiga                    import Poisson
# ... Using Matrices accelerated with Pyccel
from   pyrefiga                    import assemble_stiffness1D
from   pyrefiga                    import assemble_mass1D      
from   pyrefiga                    import assemble_matrix_ex01
from   pyrefiga                    import assemble_matrix_ex02
# ...   load a geometry from xml file 
from   pyrefiga                    import getGeometryMap
from   pyrefiga                    import load_xml

#..
from gallery.gallery_section_06             import assemble_vector_ex01
from gallery.gallery_section_06             import assemble_Quality_ex01

#..
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_Quality     = compile_kernel(assemble_Quality_ex01, arity=1)


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

#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
import argparse
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting results")
parser.add_argument("--nbpts", type=int, default=50, help="Number of elements used for plot(default: 50)")
args = parser.parse_args()

#==============================================================================
#.......Poisson ALGORITHM
def picard_solve(V1, V2, V3, V4, V, V11, V01, V10, u11_mpH = None, u12_mpH = None, times = None, x_2 = None, tol = None):
       niter      = 30   #
       if tol is None :
          tol     = 1e-8  # 
       # .. computes basis and sopans in adapted quadrature
       Quad_adm   = quadratures_in_admesh(V, nders = 0)
       #----------------------------------------------------------------------------------------------
       # ... Strong form of Neumann boundary condition which is Dirichlet because of Mixed formulation
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #..
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)

       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #..
       u_01.from_array(V01, x_D)
       u_10.from_array(V10, y_D)
       del x_D
       del y_D

       #___
       I1         = np.eye(V3.nbasis)
       I2         = np.eye(V4.nbasis)

       #... We delete the first and the last spline function
       #.. as a technic for applying Neumann boundary condition
       #.in a mixed formulation

       #..Stiffness and Mass matrix in 1D in the first deriction
       D1         = assemble_mass1D(V3)
       D1         = D1.tosparse()
       D1         = D1.toarray()
       D1         = csr_matrix(D1)
       #___
       M1         = assemble_mass1D(V1)
       M1         = M1.tosparse()
       m1         = M1
       M1         = M1.toarray()[1:-1,1:-1]
       M1         = csc_matrix(M1)
       m1         = csr_matrix(m1)

       #..Stiffness and Mass matrix in 1D in the second deriction
       D2         = assemble_mass1D(V4)
       D2         = D2.tosparse()
       D2         = D2.toarray()
       D2         = csr_matrix(D2)
       #___
       M2         = assemble_mass1D(V2)
       M2         = M2.tosparse()
       m2         = M2
       M2         = M2.toarray()[1:-1,1:-1]
       M2         = csc_matrix(M2)
       m2         = csr_matrix(m2)

       #...
       R1         = assemble_matrix_ex01(V01)
       R1         = R1.toarray()
       R1         = R1.reshape(V01.nbasis)
       r1         = R1.T
       R1         = R1[1:-1,:].T
       R1         = csr_matrix(R1)
       r1         = csr_matrix(r1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       r2         = R2
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)
       r2         = csr_matrix(r2)

       #...step 0.1
       mats_1     = [M1, M1]
       mats_2     = [D2, D2]

       # ...Fast Solver
       poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, D1]
       mats_2     = [M2, M2]

       # ...Fast Solver
       poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 1
       M1         = sla.inv(M1) #... I don't know if I can avoid the inverse TODO
       A1         = M1.dot(R1.T)
       K1         = R1.dot( A1)
       K1         = csr_matrix(K1)
       #___
       M2         = sla.inv(M2)
       A2         = M2.dot( R2.T)
       K2         = R2.dot( A2)
       K2         = csr_matrix(K2)

       #...step 2
       mats_1     = [D1, K1]
       mats_2     = [D2, K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2)

       #...non homogenoeus Neumann boundary 
       b01        = -kron(r1, D2).dot(u_01.toarray())
       #__
       b10        = -kron(D1, r2).dot( u_10.toarray())
       b_0        = b01 + b10
       #...
       b11        = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       b12        = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())
       
       #___Solve first system
       r_0        =  kron(A1.T, I2).dot(b11) + kron(I1, A2.T).dot(b12)

       #___
       x11_1      = kron(A1, I2)
       x12_1      = kron(I1, A2)
       #___
       C1         = poisson_c1.solve(2.*b11)
       C2         = poisson_c2.solve(2.*b12)

       # ... for Two or Multi grids
       if x_2 is None :    
          u11     = StencilVector(V01.vector_space)
          u12     = StencilVector(V10.vector_space)
          x11     = np.zeros(V01.nbasis) # dx/ appr.solution
          x12     = np.zeros(V10.nbasis) # dy/ appr.solution
          # ...
          u11.from_array(V01, x11)
          u12.from_array(V10, x12)
          # ...

          # .../
          x_2     = zeros(V3.nbasis*V4.nbasis)
       else           :       
          #print( 'one pice is true')
          u11          = StencilVector(V01.vector_space)
          u12          = StencilVector(V10.vector_space)
          x11          = np.zeros(V01.nbasis) # dx/ appr.solution
          x12          = np.zeros(V10.nbasis) # dy/ appr.solution
          # ...Assembles Neumann (Dirichlet) boundary conditions
          x11[-1,:]    = 1.
          x12[:,-1]    = 1.
          # ...
          x11[1:-1,:]  =  (C1 - x11_1.dot(x_2)).reshape([V1.nbasis-2,V3.nbasis])
          u11.from_array(V01, x11)
          #___
          x12[:,1:-1]  =  (C2 - x12_1.dot(x_2)).reshape([V4.nbasis,V2.nbasis-2])
          u12.from_array(V10, x12)      
       # ... for assembling residual
       M_res      = kron(D1, D2)
       #___       
       del poisson_c1, poisson_c2, b01, b10, R1, R2, r1, r2, D1, D2, M1, M2, A1, A2, b11, b12
       
       for i in range(niter):
           
           # ... computes spans and basis in adapted quadrature 
           spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11, u12)
           #---Assembles a right hand side of Poisson equation
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs(V, fields = [u11, u12, u11_mpH, u12_mpH], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2], out= rhs)
           b            = rhs.toarray()
           b            = b_0 + b.reshape(V4.nbasis*V3.nbasis)
           #___
           r            =  r_0 - b
           
           # ... Solve first system
           x2           = poisson.solve(r)
           x2           = x2 -sum(x2)/len(x2)
           #___
           x11[1:-1,:]  =  (C1 - x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])
           u11.from_array(V01, x11)
           #___
           x12[:,1:-1]  =  (C2 - x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
           u12.from_array(V10, x12)

           #..Residual 
           dx           = x2[:]-x_2[:]
           x_2[:]       = x2[:]
           
           #... Compute residual for L2
           l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
           
           if i == 0 :    
             # ...Assembles Neumann boundary conditions
             x11[-1,:]  = 1.
             x12[:,-1]  = 1.
             # ...
             u11.from_array(V01, x11)
             u12.from_array(V10, x12)
           if l2_residual < tol:
              break
       return u11, u12, x11, x12, i, l2_residual, x_2

# # .................................................................
# ....................Using Two or Multi grid method for soving MAE
# #..................................................................

def  Monge_ampere_equation(nb_ne, geometry = 'circle.xml', times = None, id_map = 0, check =None) :


   geometry       = load_xml(geometry)  # Load geometry
   if times is None :
      times           = 0.
   #...=====================
   # ... Assembling mapping
   mp             = getGeometryMap(geometry,id_map)
   degree         = mp.degree # Use same degree as geometry
   quad_degree    = max(degree[0],degree[1])*2+1 # Quadrature degree
   mp.nurbs_check = True # Activate NURBS if geometry uses NURBS
   if mp.nelements[0]*2**nb_ne < 8 and mp.nelements[1]*2**nb_ne <8 :
      raise ValueError('please for the reason of sufficient mesh choose nelemsnts strictly greater than 4')
   # ... Assembling mapping
   xmp, ymp       = mp.coefs()
   wm1, wm2       = mp.weights

   # ... Assembling mapping
   ne               = mp.nelements[0]*4
   # Create spline spaces for each direction
   V1mp            = SplineSpace(degree=degree[0], grid = mp.grids[0], mesh = mp.Refinegrid(0,None, numElevate=ne), omega = wm1, quad_degree = quad_degree)
   V2mp            = SplineSpace(degree=degree[1], grid = mp.grids[1], mesh = mp.Refinegrid(1,None, numElevate=ne), omega = wm2, quad_degree = quad_degree)
   Vmp             = TensorSpace(V1mp, V2mp)
   # ... Initial guess
   #----------------------
   # create the spline space for each direction
   V1H             = SplineSpace(degree=degree[0]+1, grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
   V2H             = SplineSpace(degree=degree[1]+1, grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)
   V3H             = SplineSpace(degree=degree[0],   grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
   V4H             = SplineSpace(degree=degree[1],   grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)

   # create the tensor space
   VH11           = TensorSpace(V3H, V4H)
   VH01           = TensorSpace(V1H, V3H)
   VH10           = TensorSpace(V4H, V2H)

   u11_mp         = StencilVector(Vmp.vector_space)
   u12_mp         = StencilVector(Vmp.vector_space)
   u11_mp.from_array(Vmp, xmp)
   u12_mp.from_array(Vmp, ymp)

   # ... G-space
   VH             = TensorSpace(V1H, V2H, V3H, V4H, V1mp, V2mp)

   #... in coarse grid
   tol            = 1e-5
   start          = time.time()
   x2H            = picard_solve(V1H, V2H, V3H, V4H, VH, VH11, VH01, VH10, u11_mpH = u11_mp, u12_mpH = u12_mp, times = times, tol = tol)[-1]
   MG_time        = time.time()- start
   # ... For multigrid method
   for n in range(4,nb_ne):
      ne          = mp.nelements[0]*2**n
      V1mg        = SplineSpace(degree=degree[0]+1, grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
      V2mg        = SplineSpace(degree=degree[1]+1, grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)
      V3mg        = SplineSpace(degree=degree[0],   grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
      V4mg        = SplineSpace(degree=degree[1],   grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)

      # create the tensor space
      Vh11mg      = TensorSpace(V3mg, V4mg)
      Vh01mg      = TensorSpace(V1mg, V3mg)
      Vh10mg      = TensorSpace(V4mg, V2mg)

      # Create spline spaces for each direction
      V1mp            = SplineSpace(degree=degree[0], grid = mp.grids[0], mesh = mp.Refinegrid(0,None, numElevate=ne), omega = wm1, quad_degree = quad_degree)
      V2mp            = SplineSpace(degree=degree[1], grid = mp.grids[1], mesh = mp.Refinegrid(1,None, numElevate=ne), omega = wm2, quad_degree = quad_degree)
      Vmp             = TensorSpace(V1mp, V2mp)
         
      Vhmg        = TensorSpace(V1mg, V2mg, V3mg, V4mg, V1mp, V2mp)
      
      #.. Prologation by knots insertion matrix
      M           = prolongation_matrix(VH11, Vh11mg)
      x2H         = M.dot(x2H)
      # ...

      # ... in new grid
      #tol       *= 1e-1
      start       = time.time()
      x2H         = picard_solve(V1mg, V2mg, V3mg, V4mg, Vhmg, Vh11mg, Vh01mg, Vh10mg, u11_mpH = u11_mp, u12_mpH = u12_mp, times = times, x_2 = x2H, tol= tol)[-1]
      MG_time    += time.time()- start
      # .. update grids
      V1H         = V1mg
      V2H         = V2mg
      V3H         = V3mg
      V4H         = V4mg

      # create the tensor space
      VH11        = TensorSpace(V3H, V4H)
      VH01        = TensorSpace(V1H, V3H)
      VH10        = TensorSpace(V4H, V2H)
      VH          = TensorSpace(V1H, V2H, V3H, V4H, V1mp, V2mp )

   # ...
   if check is not None :
      if  VH.nelements[0] == mp.nelements[0]*2 and VH.nelements[1] == mp.nelements[1]*2 :
         print(".../!\.. : two-level is activated")
      else : 
         print(".../!\.. : multi-level is activated")

   #----------------------
   # create the spline space for each direction
   ne              = mp.nelements[0]*2**nb_ne
   V1              = SplineSpace(degree=degree[0]+1, grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
   V2              = SplineSpace(degree=degree[1]+1, grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)
   V3              = SplineSpace(degree=degree[0],   grid = mp.Refinegrid(0,None, numElevate=ne), quad_degree = quad_degree)
   V4              = SplineSpace(degree=degree[1],   grid = mp.Refinegrid(1,None, numElevate=ne), quad_degree = quad_degree)

   # create the tensor space
   Vh11            = TensorSpace(V3, V4)
   Vh01            = TensorSpace(V1, V3)
   Vh10            = TensorSpace(V4, V2)

   # Create spline spaces for each direction
   V1mp            = SplineSpace(degree=degree[0], grid = mp.grids[0], mesh = mp.Refinegrid(0,None, numElevate=ne), omega = wm1, quad_degree = quad_degree)
   V2mp            = SplineSpace(degree=degree[1], grid = mp.grids[1], mesh = mp.Refinegrid(1,None, numElevate=ne), omega = wm2, quad_degree = quad_degree)
   Vmp             = TensorSpace(V1mp, V2mp)
         
   Vh              = TensorSpace(V1, V2, V3, V4, V1mp, V2mp)

   #.. Prologation by knots insertion matrix
   M                = prolongation_matrix(VH11, Vh11)
   x2H              = M.dot(x2H)	

   # ... in fine grid
   start            = time.time()
   u11_pH, u12_pH, x11uh, x12uh, iter_N, l2_residualh = picard_solve(V1, V2, V3, V4, Vh, Vh11, Vh01, Vh10, u11_mpH = u11_mp, u12_mpH = u12_mp, times = times, x_2 = x2H)[:-1]
   MG_time         += time.time()- start
   # ...
   # .. computes basis and sopans in adapted quadrature
   Quad_adm         = quadratures_in_admesh(Vh)
   spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11_pH, u12_pH)
   Quality          = StencilVector(Vh11.vector_space)
   Quality          = assemble_Quality(Vh, fields=[u11_pH, u12_pH, u11_mp, u12_mp], value = [times, spans_ad1, spans_ad2, basis_ad1, basis_ad2],  out = Quality)
   norm             = Quality.toarray()
   l2_Quality       = norm[0]
   l2_displacement  = norm[1]
   return mp.nelements[0]*2**nb_ne, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vmp

# # ........................................................
# ....................For generating tables
# #.........................................................
# ... unite-squar
# geometry = 'unitSquare.xml'
# id_map   = 122
# ... Circular domain
geometry = 'circle.xml'
id_map   = 0
# ... Quartert-annulus
#geometry = 'quart_annulus.xml'
# id_map   = 0
# ... Butterfly
#geometry = 'butterfly.xml'
# id_map   = 0

# ... new discretization for plot
nbpts           = args.nbpts
print("	\\begin{tabular}{r c c c c c}")
print("		\hline")
print("		$\#$cells & Err & CPU-time (s) & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
print("		\hline")
for nb_ne in range(5,6):

   nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp = Monge_ampere_equation(nb_ne, geometry= geometry, id_map = id_map)

   #---Compute a solution
   sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
   sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]

   #---Compute a mapping
   F1 = sol_field_NURBS_2d((nbpts,nbpts),  xmp , Vhmp.omega, Vhmp.knots, Vhmp.degree)[0]
   F2 = sol_field_NURBS_2d((nbpts,nbpts),  ymp , Vhmp.omega, Vhmp.knots, Vhmp.degree)[0]
   # ... in adaped mesh
   ux = sol_field_NURBS_2d( None,  xmp , Vhmp.omega, Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
   uy = sol_field_NURBS_2d( None,  ymp , Vhmp.omega, Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
   # ... Jacobian function of Optimal mapping
   det = uxx*uyy-uxy*uyx
   # ...
   det_min          = np.min( det[1:-1,1:-1])
   det_max          = np.max( det[1:-1,1:-1])

   # ... scientific format
   l2_Quality       = np.format_float_scientific(l2_Quality, unique=False, precision=3)
   l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
   MG_time          = round(MG_time, 3)
   det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
   det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
   print("		",nelements, "&", l2_Quality,"&",  MG_time, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
print("		\hline")
print("	\end{tabular}")
print('\n')
	
#~~~~~~~~~~~~~~~~~~~~~~~
for i in range(nbpts):
  for j in range(nbpts):
     if det[i,j] < 0.:
         print('Npoints =',nbpts,'min_Jac-F in the entire domain = ', det[i,j] ,'index =', i, j)

print('..../!\...: min~max value of the Jacobian function =', np.min(det),'~', np.max(det) )

#         -++++++++++++++++++++++++++++++++++++++++++++++++++++ End of sharing part of any geometry-----------------------------------------------------------
from numpy import pi, cos, sin, sqrt, arctan2, exp, cosh
#.. Analytic Density function 
#rho = lambda x,y : 1.+ 9./(1.+(10.*sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.5-0.25*0.) -10.*((x-0.5-0.25*0.)**2+(y-0.5)**2)))**2)
#rho = lambda x,y : (1. + 7./cosh( 5.*((x-sqrt(4)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 7./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2)
#rho = lambda x,y :1+5*np.exp(-100*np.abs((x-0.45)**2+(y-0.4)**2-0.09))+5.*np.exp(-100.*np.abs(x**2+y**2-0.2))+5.*np.exp(-100*np.abs((x+0.45)**2 +(y-0.4)**2-0.1))+7.*np.exp(-100.*np.abs(x**2+(y+1.25)**2-0.4))
#.. Test 1  circle
#def rho(x,y):
#   return 1. + 5./np.cosh( 5.*((x-np.sqrt(3)/2)**2+(y-0.5)**2 - (np.pi/2)**2) )**2 + 5./np.cosh( 5.*((x+np.sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2


# ... test butterfly
#rho       = lambda x,y : 2.+np.sin(10.*np.pi*np.sqrt((x-0.6)**2+(y-0.6)**2)) 


#rho = lambda x,y : (1. + 7./cosh( 20.*((x-sqrt(4)/2)**2+(1.2*y-0.4)**2 - 2.)**2 ) + 7./cosh( 20.*((x+sqrt(3)/2)**2+(1.2*y-0.4)**2 - 2.)**2 ) )
#rho = lambda x,y : 1.+ 5.*exp( -200.*((x-0.1)**2+(y*sin(pi*x)**2-0.5)**2 - 0.3)**2)
#rho = lambda x,y : 1.+ 10.*exp( -10.*(4.*(x-0.1)**2+2.*(y-0.35)**2 - 1.4)**2)
#rho = lambda x,y : (1.+5./cosh(40.*(2./(y**2-4.*x*(x-1)**2+1.)-2.))**2+5./cosh(10.*(2./(y**2-4.*x*(x-1)**2+1.)-2.))**2 +(30./cosh(100.*(2./(y**2-4.*x*(x-1)**2+1.)-2.))**2+30./cosh(100.*(2./(y**2-4.*x*(x-1)**2+1.)-2.))**2)*(x>0.9)*(x<1.1))

rho = ["np.exp( -1000.*np.abs(1./(np.abs(y**2-4.*x*(x-1)**2)+1.)-1.))"]


#---Compute a solution
functions = [
        {"name": "density", "expression": rho[0]},
]

paraview_nurbsAdMeshMultipatch(nbpts, [Vhmp], [xmp], [ymp], [x11uh], [x12uh], adspace =[[Vh01],[Vh10]], functions = functions, filename = 'figs/admesh_2dexample')
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/admesh_2dexample.vtm"])

print("End")

# Create a grid
x = np.linspace(-0.5, 2.5, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Define the functions
F0 = Y**2 - X*(X-1)**2
# F2 = Y**2 - X*(X-1)**2 - 2
# C0 = (X-0.5)**2 + Y**2 - (np.sqrt(2)/10)**2
# C1 = (X-2)**2 + Y**2 - 0.25

# Plot
plt.figure(figsize=(8,8))

plt.plot(x, np.sqrt(x*(1-x)**2), color = 'r', marker = '+')
plt.plot(x,-np.sqrt(x*(1-x)**2), color = 'r', marker = '+')
# Contour for F0 = 0
plt.contour(X, Y, F0, levels=[0], colors='blue', linewidths=2, linestyles='solid', label='F0')
# Contour for F2 = 0
# plt.contour(X, Y, F2, levels=[0], colors='cyan', linewidths=2, linestyles='dashed', label='F2')
# # Contour for C0 = 0
# plt.contour(X, Y, C0, levels=[0], colors='red', linewidths=2, linestyles='solid', label='C0')
# # Contour for C1 = 0
# plt.contour(X, Y, C1, levels=[0], colors='green', linewidths=2, linestyles='dashed', label='C1')

# Line for x = 2
#plt.axvline(x=2, color='magenta', linestyle='dotted', linewidth=2, label='X2')

plt.xlim(-0.5, 1.5)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal')
# plt.title('Iso-curves and Boundaries')
plt.savefig('figs/isocurves.png')
plt.grid(True)
plt.show()
