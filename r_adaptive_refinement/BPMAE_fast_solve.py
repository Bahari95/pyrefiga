from   pyrefiga                    import compile_kernel
from   pyrefiga                    import SplineSpace
from   pyrefiga                    import TensorSpace
from   pyrefiga                    import StencilMatrix
from   pyrefiga                    import StencilVector
from   pyrefiga                    import pyccel_sol_field_2d
from   pyrefiga                    import sol_field_NURBS_2d
from   pyrefiga                    import quadratures_in_admesh
#.. Prologation by knots insertion matrix
from   pyrefiga                    import prolongation_matrix
# ... Using Kronecker algebra accelerated with Pyccel
from   pyrefiga                    import Poisson
# ...   load a geometry from xml file 
from   pyrefiga                    import getGeometryMap
#... import matrix assembly kernels
from   pyrefiga                   import assemble_stiffness1D
from   pyrefiga                   import assemble_mass1D
#..
from   gallery_section_08      import assemble_vector_ex01
from   gallery_section_08      import assemble_vector_ex02
from   gallery_section_08      import assemble_vector_ex04
from   gallery_section_08      import assemble_vectorbasis_ex02
from   gallery_section_08      import assemble_Quality_ex01

#..
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_dirs        = compile_kernel(assemble_vector_ex02, arity=1)
assemble_comp        = compile_kernel(assemble_vector_ex04, arity=1)
assemble_Quality     = compile_kernel(assemble_Quality_ex01, arity=1)
#==============================================================================
#.. Assembling basis on the interface
assemble_basis         = compile_kernel(assemble_vectorbasis_ex02, arity=1)

#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray, meshgrid, linspace
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
def bmae_solve(V1, V2, V, u11_mpH, u12_mpH, x_2 = None, tol = None, niter = None, quad_degree = None):
       
      #...  find corners of the domain
      corners    = np.asarray([V1.knots[0], V1.knots[-1], V2.knots[0], V2.knots[-1]])
      if niter is None :
         # ... Number of iterations
         niter   = 30   #
      if tol is None :
         tol     = 1e-8  # 
      # .. computes basis and sopans in adapted quadrature
      Quad_adm   = quadratures_in_admesh(V, reparameterization = True)
      #----------------------------------------------------------------------------------------------       
      #..Stiffness and Mass matrix in 1D in the first deriction
      K1         = assemble_stiffness1D(V1)
      K1         = K1.tosparse()
      K1         = csr_matrix(K1)
      #___
      M1         = assemble_mass1D(V1)
      M1         = M1.tosparse()
      M1         = csr_matrix(M1)

      #..Stiffness and Mass matrix in 1D in the second deriction
      K2         = assemble_stiffness1D(V2)
      K2         = K2.tosparse()
      K2         = csr_matrix(K2)
      #___
      M2         = assemble_mass1D(V2)
      M2         = M2.tosparse()
      M2         = csr_matrix(M2)

      #...step 0.1
      mats_1     = [M1, K1]
      mats_2     = [M2, K2]

      # ...Fast Solver
      poisson    = Poisson(mats_1, mats_2, tau=1e-8) 

      # ... for assembling residual
      M_res      = kron(M1, M2)
      # ... for Two or Multi grids
      if x_2 is None :    
         u11     = StencilVector(V.vector_space)
         u12     = StencilVector(V.vector_space)
         u_sol   = StencilVector(V.vector_space)
         x11     = np.zeros(V.nbasis) # dx/ appr.solution
         x12     = np.zeros(V.nbasis) # dy/ appr.solution
         # ...
         u11.from_array(V, x11)
         u12.from_array(V, x12)
         # ...Assembles Neumann boundary conditions
         x11[0,:]   = corners[0]
         x11[-1,:]  = corners[1]
         x12[:, 0]  = corners[2]
         x12[:,-1]  = corners[3]
         # .../
         x_2     = zeros(V1.nbasis*V2.nbasis)
      else           :     
         u11          = StencilVector(V.vector_space)
         u12          = StencilVector(V.vector_space)
         u_sol        = StencilVector(V.vector_space)
         x11          = np.zeros(V.nbasis) # dx/ appr.solution
         x12          = np.zeros(V.nbasis) # dy/ appr.solution
         # ...Assembles Neumann (Dirichlet) boundary conditions
         x11[0,:]   = corners[0]
         x11[-1,:]  = corners[1]
         x12[:, 0]  = corners[2]
         x12[:,-1]  = corners[3]
         # ...
         u_sol.from_array(V, x_2.reshape(V.nbasis))
         rhs           = StencilVector(V.vector_space)
         rhs           = assemble_dirs(V, fields = [u_sol], value = [1], out = rhs)
         b             = rhs.toarray().reshape(V1.nbasis*V2.nbasis)
         #...
         x11[1:-1,:]   =  poisson.project(b).reshape(V.nbasis)[1:-1,:]
         u11.from_array(V, x11)
         #___
         rhs           = StencilVector(V.vector_space)
         rhs           = assemble_dirs(V, fields = [u_sol], value = [0], out = rhs)
         b             = rhs.toarray()
         x12[:,1:-1]  =  poisson.project(b).reshape(V.nbasis)[:,1:-1]
         u12.from_array(V, x12)      
      #___ 
      for i in range(niter):
           
         # ... computes spans and basis in adapted quadrature 
         spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11, u12)
         #---Assembles a right hand side of Poisson equation
         rhs          = StencilVector(V.vector_space)
         rhs          = assemble_rhs(V, fields = [u11, u12, u11_mpH, u12_mpH], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2, corners], out= rhs)
         b            = rhs.toarray()
         # ... Solve first system
         x2           = poisson.solve(b)
         x2           = x2 -sum(x2)/len(x2)
         #___
         u_sol.from_array(V, x_2.reshape(V.nbasis))
         rhs           = StencilVector(V.vector_space)
         rhs           = assemble_dirs(V, fields = [u_sol], value = [1], out = rhs)
         b             = rhs.toarray()
         x11[1:-1,:]  =  poisson.project(b).reshape(V.nbasis)[1:-1,:]
         u11.from_array(V, x11)
         #___
         rhs           = StencilVector(V.vector_space)
         rhs           = assemble_dirs(V, fields = [u_sol], value = [0], out = rhs)
         b             = rhs.toarray()
         x12[:,1:-1]  =  poisson.project(b).reshape(V.nbasis)[:,1:-1]
         u12.from_array(V, x12)

         #..Residual 
         dx           = x2[:]-x_2[:]
         x_2[:]       = x2[:]
         
         #... Compute residual for L2
         l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
         if l2_residual < tol and i>15:
            break
      #... End of iterations working on projecting the composition back to one mapping (reparameterization)
      # ... computes spans and basis in adapted quadrature 
      # spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11, u12)
      # Create spline spaces for each direction
      grids = np.linspace(0, 1, V1.nelements*1+1)
      Vs1   = SplineSpace(degree=V.degree[0], grid = V.grid[0], nderiv = 1, omega = V.omega[0], sharing_grid = grids, quad_degree = quad_degree)
      Vs2   = SplineSpace(degree=V.degree[1], grid = V.grid[1], nderiv = 1, omega = V.omega[1], sharing_grid = grids, quad_degree = quad_degree)
      Vh            = TensorSpace(Vs1, Vs2)
      #... Return solution
      v11           = StencilVector(V.vector_space)
      v12           = StencilVector(V.vector_space)
      #... Project back to the fine mesh
      # rhs           = assemble_comp(V, fields = [u11_mpH], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2])
      rhs           = assemble_comp(Vh, fields = [u11, u12, u11_mpH], knots= True, value = [V1.omega, V2.omega])
      vx11          = poisson.project(rhs.toarray()).reshape(V.nbasis)
      from pyrefiga import build_dirichlet
      f_exact       = ['x+0.*y']
      x_d = build_dirichlet(V, f_exact, map = (u11_mpH.toarray().reshape(V.nbasis), u12_mpH.toarray().reshape(V.nbasis)), admap=( x11, x12, V, V) )[0]
      vx11[0,:]   = x_d[0,:]
      vx11[-1,:]  = x_d[-1,:]
      vx11[:, 0]  = x_d[:, 0]
      vx11[:,-1]  = x_d[:,-1]
      v11.from_array(V, vx11.T)
      #___
      # rhs           = assemble_comp(V, fields = [u12_mpH], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2])
      rhs            = assemble_comp(Vh, fields = [u11, u12, u12_mpH], knots= True, value = [V1.omega, V2.omega])
      vx12           = poisson.project(rhs.toarray()).reshape(V.nbasis)
      f_exact        = ['0.*x+y']
      x_d = build_dirichlet(V, f_exact, map = (u11_mpH.toarray().reshape(V.nbasis), u12_mpH.toarray().reshape(V.nbasis)), admap=( x11, x12, V, V) )[0]
      vx12[0,:]   = x_d[0,:]
      vx12[-1,:]  = x_d[-1,:]
      vx12[:, 0]  = x_d[:, 0]
      vx12[:,-1]  = x_d[:,-1]
      v12.from_array(V, vx12.T)
      # ..          
      u11.from_array(V, x11.T)
      u12.from_array(V, x12.T)
      return v11, v12, vx11, vx12, u11, u12, x11, x12

# # .................................................................
# ....................Using Two or Multi grid method for soving MAE
# #..................................................................
def  Bahari_solver(nb_ne, geometry = '../fields/teapot.xml', times = None, check = False) :
      
   if times is None :
      times       = 0.
   #..... Initialisation and computing optimal mapping for 16*16
   MG_time        = 0.
   #...=====================
   # ... Assembling mapping
   mp             = getGeometryMap(geometry,0)
   degree         = mp.degree # Use same degree as geometry
   quad_degree    = max(degree[0],degree[1])*4+1 # Quadrature degree
   mp.nurbs_check = True # Activate NURBS if geometry uses NURBS
   if mp.nelements[0]*nb_ne < 16 and mp.nelements[1]*nb_ne <16 :
      print("nelements = ", mp.nelements[0]*nb_ne, mp.nelements[1]*nb_ne)
      raise ValueError('please for the reason of sufficient mesh choose nelemsnts strictly greater than 4')
   # ... Assembling mapping
   weight, xmp, ymp = mp.RefineGeometryMap(numElevate=nb_ne)
   # ... Assembling mapping
   wm1, wm2         = weight[:,0], weight[0,:]

   # Create spline spaces for each direction
   V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,None, numElevate=nb_ne), nderiv = 1, omega = wm1, quad_degree = quad_degree)
   V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,None, numElevate=nb_ne), nderiv = 1, omega = wm2, quad_degree = quad_degree)
   Vh              = TensorSpace(V1, V2)
   #...=====================
   #.. second patch
   #...=====================
   nb_patches      = 1
   MultipatchVh    = []
   Multipatchmpx   = []
   Multipatchmpy   = []
   Multipatchadx   = []
   Multipatchady   = []
   l2_Quality      = 0.
   l2_displacement = 0.
   for i in range(0,nb_patches):
      # ... Assembling mapping
      mp              = getGeometryMap(geometry,i)
      mp.nurbs_check  = True # Activate NURBS if geometry uses NURBS
      # ... Assembling mapping
      weight, xmp, ymp = mp.RefineGeometryMap(numElevate=nb_ne)
      # ... Assembling mapping
      # wm1, wm2         = weight[:,0], weight[0,:]   
      #.....
      Multipatchmpx.append(xmp)
      Multipatchmpy.append(ymp)
      MultipatchVh.append(Vh)
      # ...
      u11_mph         = StencilVector(Vh.vector_space)
      u12_mph         = StencilVector(Vh.vector_space)
      u11_mph.from_array(Vh, xmp)
      u12_mph.from_array(Vh, ymp)

      start            = time.time()
      v11_H, v12_H, vx11uh, vx12uh, u11_pH, u12_pH, x11uh, x12uh     = bmae_solve(V1, V2, Vh, u11_mph, u12_mph, quad_degree = quad_degree)
      MG_time         += time.time()- start
      from pyrefiga import save_geometry_to_xml
      # print('weights =', Vh.omega)
      Gmap  = np.zeros((V1.nbasis*V2.nbasis,2))
      Gmap[:,0] = u12_pH.toarray()[:]
      Gmap[:,1] = u11_pH.toarray()[:]
      save_geometry_to_xml(Vh, Gmap, locname = "figs/admapping_psi{}".format(V1.nbasis))
      Gmap[:,0] = v11_H.toarray()[:]
      Gmap[:,1] = v12_H.toarray()[:]
      save_geometry_to_xml(Vh, Gmap, locname = "figs/admapping_patch{}".format(V1.nbasis))
      #...
      Multipatchadx.append(vx11uh)
      Multipatchady.append(vx12uh)
      if check :
         print("		Mapping for patch", i, "computed in", round(MG_time, 3), "seconds")
      # ...
      # Create spline spaces for each direction
      # grids = np.linspace(0, 1, V1.nelements*4+1)
      # V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,None, numElevate=nb_ne), nderiv = 1, omega = wm1, quad_degree = quad_degree)
      # V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,None, numElevate=nb_ne), nderiv = 1, omega = wm2, quad_degree = quad_degree)
      # Vh              = TensorSpace(V1, V2)
      # print( " nelements for patch", i, "=", V1.spans)
      # .. computes basis and spans in adapted quadrature
      Quad_adm         = quadratures_in_admesh(Vh, True, nders = 1)
      spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11_pH, u12_pH)
      Quality          = StencilVector(Vh.vector_space)
      Quality          = assemble_Quality(Vh, fields=[u11_pH, u12_pH, u11_mph, u12_mph, v11_H, v12_H],knots = True, value = [times, spans_ad1, spans_ad2, basis_ad1, basis_ad2],  out = Quality)
      norm             = Quality.toarray()
      l2_Quality      += norm[0]**2
      l2_displacement += norm[1]**2
      print(" The Volume for patch", i, "comp =", norm[2], "appr=", norm[3], "bdr=", norm[4] )

   l2_Quality       = sqrt(l2_Quality     )
   l2_displacement  = sqrt(l2_displacement)
   return Vh.nelements, l2_Quality, MG_time, l2_displacement, Multipatchadx, Multipatchady, Multipatchmpx, Multipatchmpy, MultipatchVh,norm[2], norm[3], norm[4]

# # ........................................................
# ....................For generating tables
# #.........................................................
if True :
   # ... unit-square
   #geometry = '../fields/unit_square.xml'

   geometry = '../fields/circle.xml'

   # ... quarter annulus
   #geometry = '../fields/quart_annulus.xml'

   # ... 
   #geometry = '../fields/lake.xml'

   # geometry = '../fields/elasticity.xml'

   #geometry = '../fields/nice_geo.xml'   
   # ... new discretization for plot
   
   nbpts    = args.nbpts
   table    = np.zeros((4,4))
   print("	\subcaption{geometry =",geometry,"}")
   print("	\\begin{tabular}{r c c c c}")
   print("		\hline")
   print("		$\#$cells & CPU-time (s) & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$ \\\\")
   print("		\hline")
   for ne in range(5,6):

      nb_ne = 2**ne
      nelements, l2_Quality, MG_time, l2_displacement, MPadx, MPady, MPmpx, MPmpy, MPVh,norm2, norm3, norm4 = Bahari_solver(nb_ne, geometry= geometry)
      table[0,ne-4] = (MPVh[0].degree[0]+MPVh[0].nelements[0])*(MPVh[0].degree[0]+MPVh[0].nelements[1])
      table[1,ne-4] = norm2
      table[2,ne-4] = norm3
      table[3,ne-4] = norm4
      # #---Compute a mapping
      uxx = np.zeros((len(MPVh)*nbpts, nbpts))
      uyy = np.zeros((len(MPVh)*nbpts, nbpts))
      uyx = np.zeros((len(MPVh)*nbpts, nbpts))
      uxy = np.zeros((len(MPVh)*nbpts, nbpts))
      for ii in range(len(MPVh)):
        jj = ii + 1
        #---Compute a solution
        uxx[ii*nbpts:jj*nbpts,:], uxy[ii*nbpts:jj*nbpts,:] = sol_field_NURBS_2d((nbpts,nbpts),  MPadx[ii], MPVh[ii].omega, MPVh[ii].knots, MPVh[ii].degree)[1:3]
        uyx[ii*nbpts:jj*nbpts,:], uyy[ii*nbpts:jj*nbpts,:] = sol_field_NURBS_2d((nbpts,nbpts),  MPady[ii], MPVh[ii].omega, MPVh[ii].knots, MPVh[ii].degree)[1:3]
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
      print("		",nelements[0],"$\\times$",nelements[1],"&",  MG_time, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
   print("		\hline")
   print("	\end{tabular}")
   print('\n')
   np.savetxt('tableerror.txt', table, fmt='%.20e')

#~~~~~~~~~~~~~~~~~~~~~~~
# for i in range(nbpts):
#   for j in range(nbpts):
#      if det[i,j] < 0.:
#          print('Npoints =',nbpts,'min_Jac-F in the entire domain = ', det[i,j] ,'index =', i, j)

# print('..../!\...: min~max value of the Jacobian function =', np.min(det),'~', np.max(det) )

#         -++++++++++++++++++++++++++++++++++++++++++++++++++++ End of sharing part of any geometry-----------------------------------------------------------

#.. Analytic Density function 
#rho = '1.+ 9./(1.+(10.*sqrt((x-0.-0.25*0.)**2+(y-0.)**2)*cos(arctan2(y-0.,x-0.-0.25*0.) -10.*((x-0.-0.25*0.)**2+(y-0.)**2)))**2)'
#rho = '(1.+5./(1.+exp(100.*((x-0.)**2+(y-0.)**2-0.9))))'
#rho  = '5./(2.+np.cos(4.*np.pi*np.sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2)))'
#rho  = '9./(2.+np.cos(10.*np.pi*np.sqrt((x)**2+(y+2.)**2)))'
#rho   = '1.+9.*np.exp(-10.*np.abs((x-0.5-0.0*np.cos(2.*np.pi*0.))**2-(y-0.5-0.5 *np.sin(2.*np.pi*0.))**2- 0.09))'

# rho   = '1.+5.*np.exp(-0.25*np.abs((x-0.)**2+(y-0.)**2-1.05**2))'
rho   = '1.+12./np.cosh( 80.*((x + y) )**2 )'

#rho = '1+5*np.exp(-100*np.abs((x-0.45)**2+(y-0.4)**2-0.09))+5.*np.exp(-100.*np.abs(x**2+y**2-0.2))+5.*np.exp(-100*np.abs((x+0.45)**2 +(y-0.4)**2-0.1))+7.*np.exp(-100.*np.abs(x**2+(y+1.25)**2-0.4))'

#rho = '1+5.*np.exp(-50.*np.abs(x**2+y**2-0.5))'
#rho = '1.+5./np.cosh(40.*(2./(y**2-x*(x-1)**2+1.)-2.))**2+5./np.cosh(10.*(2./(y**2-x*(x-1)**2+1.)-2.))**2' #+100*exp( -1000.*abs(1./(abs(y**2-x*(x-1)**2)+1.)-1.))*(x>0.95)*(x<1.05)'

#.. Test 1  circle
#rho = '5./np.cosh( 5.*((x-np.sqrt(3)/2)**2+(y-0.5)**2 - (np.pi/2)**2) )**2 + 5./np.cosh( 5.*((x+np.sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2'


# ... test butterfly
#rho       = lambda x,y : 2.+np.sin(10.*np.pi*np.sqrt((x-0.6)**2+(y-0.6)**2)) 
from   pyrefiga                    import paraview_nurbsAdMeshMultipatch, paraview_nurbsSolutionMultipatch
functions = [
    {"name": "density", "expression": rho},
]
#paraview_nurbsAdMeshMultipatch(nbpts, MPVh, MPmpx, MPmpy, MPadx, MPady, functions = functions)
paraview_nurbsSolutionMultipatch(nbpts, MPVh, MPadx, MPady, functions = functions, filename = "figs/admultipatch_multiblock")
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/admultipatch_multiblock.vtm"])