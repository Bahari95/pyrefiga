from   pyrefiga                    import compile_kernel
from   pyrefiga                    import SplineSpace
from   pyrefiga                    import TensorSpace
from   pyrefiga                    import StencilMatrix
from   pyrefiga                    import StencilVector
from   pyrefiga                    import pyccel_sol_field_2d
from   pyrefiga                    import quadratures_in_admesh
#.. Prologation by knots insertion matrix
from   pyrefiga                    import prolongation_matrix
# ... Using Kronecker algebra accelerated with Pyccel
from   pyrefiga                    import Poisson
#... import matrix assembly kernels
from   pyrefiga                   import assemble_stiffness1D
from   pyrefiga                   import assemble_mass1D
#..
from MPrIGA.gallery_section_00      import assemble_vector_ex01
from MPrIGA.gallery_section_00      import assemble_vector_ex02
from MPrIGA.gallery_section_00      import assemble_vectorbasis_ex02


#..
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_dirs        = compile_kernel(assemble_vector_ex02, arity=1)
#==============================================================================
#.. Assembling basis on the interface
assemble_basis         = compile_kernel(assemble_vectorbasis_ex02, arity=1)

#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   numpy                        import zeros
from   numpy                        import sqrt
import numpy                        as     np

#==============================================================================
#.......Poisson ALGORITHM
def bmae_solve(V1, V2, V, u11_mpH, u12_mpH, x_2 = None, tol = None, niter = None):
       
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
       return u11, u12, x11, x12
