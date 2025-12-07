import numpy as np
from functools import partial

from pyccel.epyccel import epyccel

from .linalg import StencilMatrix
from .linalg import StencilVector
from .spaces import TensorSpace

__all__ = ['assemble_matrix', 'assemble_vector', 'assemble_scalar', 'compile_kernel', 'apply_dirichlet', 'apply_periodic']

#==============================================================================
def assemble_matrix(core, V, fields=None, knots = None, value = None, out=None):
    if out is None:
        out = StencilMatrix(V.vector_space, V.vector_space)
        
    # ...
    args = []
    if knots is None :
       if isinstance(V, TensorSpace):
           # default for 2D: for integration we use same mesh and quadrature in each direction
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3

           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)   
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points]
    # ...
    else :
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3
           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])
           args += list(V.knots)

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points,
                   V.knots]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [u._data for u in fields]
    # ...
        
    if not(value is None):
        for x_value in value:
               args += [x_value]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_vector(core, V, fields=None, knots = None, value = None, out=None):
    if out is None:
        out = StencilVector(V.vector_space)

    # ...
    args = []
    if knots is None :
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3
           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)   
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points]
    # ...
    else :
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3
           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])
           args += list(V.knots)

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points,
                   V.knots]
                
    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]

    if not(value is None):
        for x_value in value:
               args += [x_value]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_scalar(core, V, fields=None, knots = None, value = None):
    # ...
    args = []
    if knots is None :
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3
           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)   
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points]
    # ...
    else :
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.omega[3] is not None)) else 3
           args += list(V.nelements[:int_par])
           args += list(V.degree)
           args += list(V.spans)
           args += list(V.basis)
           args += list(V.weights[:int_par])
           args += list(V.points[:int_par])
           args += list(V.knots)

       else:
           args = [V.nelements,
                   V.degree,
                   V.spans,
                   V.basis,
                   V.weights,
                   V.points,
                   V.knots]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]
    
    if not(value is None):
        for x_value in value:
               args += [x_value]
    return core( *args )

#==============================================================================
def compile_kernel(core, arity, pyccel=True):
    assert(arity in [0,1,2])

    if pyccel:
        core = epyccel(core) #, accelerators = '--openmp' libs = ["/usr/lib/gcc/x86_64-linux-gnu/9", "-gfortran", "-lm"] )#, language = 'c')

    if arity == 2:
        return partial(assemble_matrix, core)

    elif arity == 1:
        return partial(assemble_vector, core)

    elif arity == 0:
        return partial(assemble_scalar, core)

#==============================================================================
def apply_nitsch(V, stifness, u11_mph, u12_mph, interface):
    if isinstance(stifness, StencilMatrix):
        pass
    else:
        raise NotImplementedError("apply_nitsch: 'stifness' must be a StencilMatrix (other types not implemented)")
    # ... nitsche assembly tools
    from gallery import assemble_matrix_nitsche_ex00
    assemble_stiffness_nitsche  = compile_kernel(assemble_matrix_nitsche_ex00, arity=2)
    from gallery import assemble_matrix_nitsche_ex02
    assemble_stiffness2_nitsche = compile_kernel(assemble_matrix_nitsche_ex02, arity=1)
    

#==============================================================================
def apply_dirichlet(V, x, dirichlet = True, dirichlet_patch2 = None, update = None):
    """
    Applies Dirichlet boundary conditions to a matrix or vector by elimination.

    dirichlet can take different forms depending on how boundary conditions are specified:
    A single boolean (True or False) meaning the same condition applies to all boundaries.
    A list of booleans, e.g. [True, False], specifying the condition for each direction in 1D or 2D.
    A nested list of booleans, specifying Dirichlet conditions in multiple dimensions:
        2D example: [[True, False], [True, True]]
        3D example: [[True, False], [True, True], [True, True]]

    Parameters
    ----------
    V : TensorSpace
        The function space containing basis and dimension information.
    x : StencilMatrix or StencilVector
        The matrix or vector to which Dirichlet conditions are applied.
    dirichlet : list or tuple, optional
        Specifies which boundaries have Dirichlet conditions (default: None).
    dirichlet_patch2 : list or tuple, optional
        Specifies a second patch for Dirichlet elimination (default: None).
    update: StencilVector
        Updates the boundary values of the solution using the exact Dirichlet data.
    Returns
    -------
    ndarray
        The matrix or vector with Dirichlet boundaries applied and reshaped accordingly.
    """
    if update is None :
        if V.dim ==2:
            if dirichlet is True:
                dirichlet = [[True, True], [True, True]]
            n1, n2  = V.nbasis
            rhs     = False
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2

            if isinstance(x, StencilMatrix):
                x   = (x.tosparse()).toarray().reshape((n1,n2,n1,n2))
            elif isinstance(x, StencilVector): # TODO 
                if n1*n2 == x.toarray().shape[0]:
                    x   = x.toarray().reshape((n1,n2))
                    rhs = True
                else:
                    x   = x.toarray().reshape((n1,n2,n1,n2))
            else:
                raise NotImplementedError('Not available')
            #... apply Dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2,d3:d4].reshape((d2-d1)*(d4-d3))
                else:
                    x   = x[d1:d2,d3:d4,d1:d2,d3:d4].reshape(((d2-d1)*(d4-d3), (d2-d1)*(d4-d3)))
            else:
                #indeces for elimination
                pd1 = 1    if dirichlet_patch2[0][0] else 0 
                pd2 = n1-1 if dirichlet_patch2[0][1] else n1
                pd3 = 1    if dirichlet_patch2[1][0] else 0 
                pd4 = n2-1 if dirichlet_patch2[1][1] else n2
                x   = x[pd1:pd2,pd3:pd4,d1:d2,d3:d4].reshape(((pd2-pd1)*(pd4-pd3), (d2-d1)*(d4-d3)))
            return  x
        elif V.dim == 3:
            if dirichlet is True:
                dirichlet = [[True, True], [True, True], [True, True]]
            n1, n2, n3  = V.nbasis
            rhs         = False
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            d5 = 1    if dirichlet[2][0] else 0 
            d6 = n3-1 if dirichlet[2][1] else n3

            if isinstance(x, StencilMatrix):
                x   = (x.tosparse()).toarray().reshape((n1,n2,n3, n1,n2,n3))
            elif isinstance(x, StencilVector): # TODO special case for Nitsch's
                if n1*n2*n3 == x.toarray().shape[0]:
                    x   = x.toarray().reshape((n1,n2,n3))
                    rhs = True
                else:
                    x   = x.toarray().reshape((n1,n2,n3,n1,n2,n3))
            else:
                raise NotImplementedError('Not available')
            #... apply Dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2,d3:d4,d5:d6].reshape((d2-d1)*(d4-d3)*(d6-d5))
                else:
                    x   = x[d1:d2,d3:d4,d5:d6,d1:d2,d3:d4,d5:d6].reshape(((d2-d1)*(d4-d3)*(d6-d5), (d2-d1)*(d4-d3)*(d6-d5)))
            else:
                #indeces for elimination
                pd1 = 1    if dirichlet_patch2[0][0] else 0 
                pd2 = n1-1 if dirichlet_patch2[0][1] else n1
                pd3 = 1    if dirichlet_patch2[1][0] else 0 
                pd4 = n2-1 if dirichlet_patch2[1][1] else n2
                pd5 = 1    if dirichlet_patch2[2][0] else 0 
                pd6 = n3-1 if dirichlet_patch2[2][1] else n3
                x   = x[pd1:pd2,pd3:pd4,pd5:pd6,d1:d2,d3:d4,d5:d6].reshape(((pd2-pd1)*(pd4-pd3)*(pd6-pd5), (d2-d1)*(d4-d3)*(d6-d5)))
            return  x        
        elif V.dim == 1:
            if dirichlet is True:
                dirichlet = [True, True]
            n1      = V.nbasis
            rhs     = False
            #indeces for elimination
            d1 = 1    if dirichlet[0] else 0 
            d2 = n1-1 if dirichlet[1] else n1

            if isinstance(x, StencilMatrix):
                x   = (x.tosparse()).toarray().reshape((n1,n1))
            elif isinstance(x, StencilVector): # TODO 
                if n1 == x.toarray().shape[0]:
                    x   = x.toarray().reshape(n1)
                    rhs = True
                else:
                    x   = x.toarray().reshape((n1,n1))
            else:
                raise NotImplementedError('Not available')
            #... apply Dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2].reshape((d2-d1))
                else:
                    x   = x[d1:d2,d1:d2].reshape(((d2-d1), (d2-d1)))
            else:
                #indeces for elimination
                pd1 = 1    if dirichlet_patch2[0] else 0 
                pd2 = n1-1 if dirichlet_patch2[1] else n1
                x   = x[pd1:pd2,pd3:pd4,d1:d2,d3:d4].reshape(((pd2-pd1), (d2-d1)))
            return  x
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')
    else:
        if isinstance(update, StencilVector):
            pass
        else:
            raise NotImplementedError('Not available')
        u   = StencilVector(V.vector_space)
        # ...
        if V.dim ==2:
            if dirichlet is True:
                dirichlet = [[True, True], [True, True]]
            n1, n2  = V.nbasis
            rhs     = False
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            #... apply Dirichlet
            u[:,:]         = update[:,:]
            u[d1:d2,d3:d4] = x.reshape((d2-d1),(d4-d3))
            return  u
        elif V.dim == 3:
            if dirichlet is True:
                dirichlet = [[True, True], [True, True], [True, True]]
            n1, n2, n3  = V.nbasis
            rhs         = False
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            d5 = 1    if dirichlet[2][0] else 0 
            d6 = n3-1 if dirichlet[2][1] else n3
            #... apply Dirichlet
            u[:,:,:]             =   update[:,:,:] 
            u[d1:d2,d3:d4,d5:d6] = x.reshape((d2-d1),(d4-d3),(d6-d5))
            return  u
        elif V.dim == 1:
            if dirichlet is True:
                dirichlet = [True, True]
            n1      = V.nbasis
            rhs     = False
            #indeces for elimination
            d1 = 1    if dirichlet[0] else 0 
            d2 = n1-1 if dirichlet[1] else n1
            #... apply Dirichlet
            u[:]     =   update[:] 
            u[d1:d2] = x.reshape(d2-d1)
            return  u
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')
#==============================================================================
def apply_periodic(V, x, periodic = None, update = None):
       
  if update is None :
    if isinstance(x, StencilMatrix):
        x          = x.tosparse()
        x          = x.toarray()
        if V.dim == 1:
            p  = V.degree
            n1 = V.nbasis

            #... eliminate ghost regions
            li            = np.zeros((2*p, n1))
            li[:p,:]      = x[-p:,:]
            li[p:p+p,-p:] = x[-2*p:-p,-p:]

            x    = x[:-p,:-p] 
            for i in range(p):
                x[i,:]     += li[i,:-p]
                x[i,:p]    += li[i,-p:]       
                x[-1-i,:p] += li[2*p-1-i,-p:]

            return x

        elif V.dim == 2:
          x          = x.reshape((V.nbasis[0],V.nbasis[1], V.nbasis[0], V.nbasis[1]) )
          if True in periodic:
            p1,p2  = V.degree
            n1,n2  = V.nbasis
            #... eliminate ghost regions
            
            if periodic[0] == True:
               lix                       = np.zeros((2*p1, n2, n1, n2))
               lix[:p1,:, :, :]          = x[-p1:,:, :,:]
               lix[p1:p1+p1, :, -p1:, :] = x[-2*p1:-p1,:, -p1:, :]
            
               x    = x[:-p1,:,:-p1,:] 
               for i in range(p1):
                  x[i, :, :, :]      += lix[i, :, :-p1, :]
                  x[i, :, :p1, :]    += lix[i, :,-p1:, :]       
                  x[-1-i, :, :p1, :] += lix[2*p1-1-i, :, -p1:, :]
               n1 = n1 - p1
            if periodic[1] == True:
               liy                       = np.zeros((n1, 2*p2, n1, n2))
               liy[:,:p2,:,:]            = x[:,-p2:, :,:]
               liy[:,p2:p2+p2,:,-p2:]    = x[:, -2*p2:-p2, :, -p2:]
 
               x    = x[:,:-p2, :,:-p2] 
               for j in range(p2):
                  x[:, j, :, :]      += liy[:, j, :, :-p2]
                  x[:, j, :, :p2]    += liy[:, j, :, -p2:]       
                  x[:, -1-j, :, :p2] += liy[:, 2*p2-1-j, :, -p2:]
               n2 = n2 - p2
            x          = x.reshape(( n1*n2,n1*n2 ))                
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one dimension')

        elif V.dim == 3:
          x          = x.reshape((V.nbasis[0],V.nbasis[1],V.nbasis[2], V.nbasis[0],V.nbasis[1],V.nbasis[2]) )
          if True in periodic:
            p1,p2,p3   = V.degree
            n1,n2, n3  = V.nbasis
            #... eliminate ghost regions
            if periodic[0]==True :
               li                         = np.zeros((2*p1,n2,n3, n1,n2,n3))
               li[:p1,:,:, :,:,:]         = x[-p1:,:,:, :,:,:]
               li[p1:p1+p1,:,:, -p1:,:,:] = x[-2*p1:-p1,:,:, -p1:,:,:]
            
               x    = x[:-p1,:,:, :-p1,:,:] 
               for i in range(p1):
                  x[i,:,:, :,:,:]      += li[i,:,:, :-p1,:,:]
                  x[i,:,:, :p1,:,:]    += li[i,:,:, -p1:,:,:]       
                  x[-1-i,:,:,:p1,:,:]  += li[2*p1-1-i,:,:, -p1:,:,:]
               n1 = n1 - p1
            if periodic[1]==True :
               #...
               li                            = np.zeros((n1,2*p2,n3, n1,n2,n3))
               li[:,:p2,:, :,:,:]            = x[:,-p2:,:, :,:,:]
               li[:,p2:p2+p2,:, :,-p2:,:]    = x[:,-2*p2:-p2,:, :,-p2:,:]
 
               x    = x[:,:-p2,:, :,:-p2,:] 
               for j in range(p2):
                  x[:,j,:, :,:,:]      += li[:,j,:, :,:-p2,:]
                  x[:,j,:, :,:p2,:]    += li[:,j,:, :,-p2:,:]       
                  x[:,-1-j,:, :,:p2,:] += li[:,2*p2-1-j,:, :,-p2:,:]
               n2 = n2 - p2
            if periodic[2]==True :
               #...
               li                            = np.zeros((n1,n2,2*p3, n1,n2,n3))
               li[:,:,:p3, :,:,:]            = x[:,:,-p3:, :,:,:]
               li[:,:,p3:p3+p3, :,:,-p3:]    = x[:,:,-2*p3:-p3, :,:,-p3:]
 
               x    = x[:,:,:-p3, :,:,:-p3] 
               for k in range(p3):
                  x[:,:,k, :,:,:]      += li[:,:,k, :,:,:-p3]
                  x[:,:,k, :,:,:p3]    += li[:,:,k, :,:,-p3:]       
                  x[:,:,-1-k, :,:,:p3] += li[:,:,2*p3-1-k, :,:,-p3:]
               n3 = n3 - p3                
            x          = x.reshape(( n1*n2*n3, n1*n2*n3 ))                
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')
        else :
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    elif isinstance(x, StencilVector):
        x          = x.toarray()
        x          = x.reshape(V.nbasis)
        if V.dim == 1:
            #... eliminate ghost regions
            p    = V.degree

            a    = np.zeros(x.shape[0])
            a[:] = x[:]

            x  = x[:-p]
            for i in range(p):
                x[i]             += a[-p+i]
            return x

        elif V.dim == 2:
          if periodic == [True, True] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x      = x[:-p1,:-p2]
            for i in range(p1):
               for j in range(p2):            
                   x[i,j]            += a[i,-p2+j] + a[-p1+i,j] + a[-p1+i,-p2+j]
            for i in range(p1):
                x[i,p2:]             += a[-p1+i,p2:-p2]
            for j in range(p2):
                x[p1:,j]             += a[p1:-p1,-p2+j]
            x      = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2) ))
            return x

          elif periodic == [True, False] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x      = x[:-p1,:]
            for i in range(p1):
                x[i,:]             += a[-p1+i,:]
            x      = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]) ))
            return x
            
          elif  periodic == [False, True] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x     = x[:,:-p2]
            for j in range(p2):
                x[:,j]             += a[:,-p2+j]
            x     = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2) ))                                            
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                

        # ... 
        elif V.dim == 3:
          if periodic == [True, True, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:-p2,:-p3]
            for i in range(p1):
              for j in range(p2):
                for k in range(p3):
                    x[i,j,k]             += a[i,j,-p3+k] + a[i,-p2+j,k]  + a[i,-p2+j,-p3+k] + a[-p1+i,j,k]  + a[-p1+i,j,-p3+k] + a[-p1+i,-p2+j,k] + a[-p1+i,-p2+j,-p3+k]
            # ...
            for i in range(p1):
              for j in range(p2):
                    x[i,j,p3:]           += a[i,-p2+j,p3:-p3] + a[-p1+i,j,p3:-p3]  + a[-p1+i,-p2+j,p3:-p3]
            for i in range(p1):
                for k in range(p3):
                    x[i,p2:,k]           += a[i,p2:-p2,-p3+k] + a[-p1+i,p2:-p2,k]  + a[-p1+i,p2:-p2,-p3+k]
            for j in range(p2):
                for k in range(p3):
                    x[p1:,j,k]           += a[p1:-p1,j,-p3+k] + a[p1:-p1,-p2+j,k]  + a[p1:-p1,-p2+j,-p3+k]
            # ...
            for i in range(p1):
                    x[i,p2:,p3:]         += a[-p1+i,p2:-p2,p3:-p3]
            for j in range(p2):
                    x[p1:,j,p3:]         += a[p1:-p1,-p2+j,p3:-p3]
            for k in range(p3):
                    x[p1:,p2:,k]         += a[p1:-p1,p2:-p2,-p3+k]
                                                                                
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2)*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [True, True, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:-p2,:]
            for i in range(p1):
              for j in range(p2):
                    x[i,j,:]             += a[i,-p2+j,:] + a[-p1+i,j,:]  + a[-p1+i,-p2+j,:]
            # ...
            for i in range(p1):
                    x[i,p2:,:]           += a[-p1+i,p2:-p2,:]
            for j in range(p2):
                    x[p1:,j,:]           += a[p1:-p1,-p2+j,:]
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2)*(V.nbasis[2]) ))
            return x                

          elif periodic == [True, False, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:,:-p3]
            for i in range(p1):
                for k in range(p3):
                    x[i,:,k]             += a[i,:,-p3+k] + a[-p1+i,:,k]  + a[-p1+i,:,-p3+k]
            # ...
            for i in range(p1):
                    x[i,:,p3:]           += a[-p1+i,:,p3:-p3]
            for k in range(p3):
                    x[p1:,:,k]           += a[p1:-p1,:,-p3+k]
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1])*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, True, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:-p2,:-p3]
            for j in range(p2):
                for k in range(p3):
                    x[:,j,k]             += a[:,j,-p3+k] + a[:,-p2+j,k]  + a[:,-p2+j,-p3+k]   
            # ...
            for j in range(p2):
                    x[:,j,p3:]           += a[:,-p2+j,p3:-p3]
            for k in range(p3):
                    x[:,p2:,k]           += a[:,p2:-p2,-p3+k]
                    
            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2)*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, False, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:,:-p3]
            for k in range(p3):
                    x[:,:,k]             += a[:,:,-p3+k] 
            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1])*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, True, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:-p2,:]
            for j in range(p2):
                    x[:,j,:]             +=  a[:,-p2+j,:]                    

            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2)*(V.nbasis[2]) ))
            return x
          elif periodic == [True, False, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:,:]
            for i in range(p1):
                    x[i,:,:]             +=  a[-p1+i,:,:]
                    
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1])*(V.nbasis[2]) ))
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                
                             
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    else:
        raise TypeError('ERROR 1 ! Expecting StencilMatrix or StencilVector')

  else :
        if V.dim == 1:
            #... update the eliminated ghost regions
            p       = V.degree
            n1      = x.shape[0] + p
            
            a       = np.zeros(n1)
            for i in range(p):
                a[-p-i] = x[i]
            return a

        elif V.dim == 2:
          if periodic == [True, True] :
            #... update the eliminated ghost regions
            p1,p2   = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
                        
            a       = np.zeros((n1, n2) )           
            a[:-p1,:-p2]  = x[:,:]
            for i in range(p1):
               for j in range(p2):            
                   a[i,-p2+j]     = x[i,j]
                   a[-p1+i,j]     = x[i,j] 
                   a[-p1+i,-p2+j] = x[i,j]

            for i in range(p1):
                a[-p1+i,p2:-p2] = x[i,p2:]

            for j in range(p2):
                a[p1:-p1,-p2+j] = x[p1:,j]                  

            return a

          elif periodic == [True, False] :
            #... update the eliminated ghost regions
            p1,p2      = V.degree
            n1         = x.shape[0] + p1
            n2         = x.shape[1]
                        
            a          = np.zeros((n1, n2))
            a[:-p1,:]  = x[:,:]
            for i in range(p1):
                a[-p1+i,:]   = x[i,:]

            return a

          elif  periodic == [False, True] :
            #... update the eliminated ghost regions
            p1,p2      = V.degree
            n1         = x.shape[0]
            n2         = x.shape[1] + p2
                        
            a          = np.zeros((n1, n2))
            a[:,:-p2]  = x[:,:]
            for j in range(p2):
                a[:,-p2+j]   = x[:,j]                                            
            return a
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')      

        # ... 
        elif V.dim == 3:
          if periodic == [True, True, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:-p2,:-p3]  = x[:,:,:]            
            for i in range(p1):
              for j in range(p2):
                for k in range(p3):
                    a[i,j,-p3+k]          = x[i,j,k]
                    a[i,-p2+j,k]          = x[i,j,k]
                    a[i,-p2+j,-p3+k]      = x[i,j,k]
                    a[-p1+i,j,k]          = x[i,j,k]
                    a[-p1+i,j,-p3+k]      = x[i,j,k]
                    a[-p1+i,-p2+j,k]      = x[i,j,k]
                    a[-p1+i,-p2+j,-p3+k]  = x[i,j,k]
            # ...
            for i in range(p1):
              for j in range(p2):
                    a[i,-p2+j,p3:-p3]     = x[i,j,p3:]
                    a[-p1+i,j,p3:-p3]     = x[i,j,p3:]
                    a[-p1+i,-p2+j,p3:-p3] = x[i,j,p3:]
            for i in range(p1):
                for k in range(p3):
                    a[i,p2:-p2,-p3+k]     = x[i,p2:,k]
                    a[-p1+i,p2:-p2,k]     = x[i,p2:,k]
                    a[-p1+i,p2:-p2,-p3+k] = x[i,p2:,k]
            for j in range(p2):
                for k in range(p3):
                    a[p1:-p1,j,-p3+k]     = x[p1:,j,k]
                    a[p1:-p1,-p2+j,k]     = x[p1:,j,k]
                    a[p1:-p1,-p2+j,-p3+k] = x[p1:,j,k]
            # ...
            for i in range(p1):
                    a[-p1+i,p2:-p2,p3:-p3] = x[i,p2:,p3:]
            for j in range(p2):
                    a[p1:-p1,-p2+j,p3:-p3] = x[p1:,j,p3:]
            for k in range(p3):
                    a[p1:-p1,p2:-p2,-p3+k] = x[p1:,p2:,k]
            return a

          elif periodic == [True, True, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:-p2,:]  = x[:,:,:]            
            for i in range(p1):
              for j in range(p2):
                    a[i,-p2+i,:]     = x[i,j,:]
                    a[-p1+i,j,:]     = x[i,j,:]
                    a[-p1+i,-p2+j,:] = x[i,j,:]
            # ...
            for i in range(p1):
                    a[-p1+i,p2:-p2,:] = x[i,p2:,:]
            for j in range(p2):
                    a[p1:-p1,-p2+j,:] = x[p1:,j,:]
            return a

          elif periodic == [True, False, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] 
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:,:-p3]  = x[:,:,:]            
            for i in range(p1):
                for k in range(p3):
                    a[i,:,-p3+k]      = x[i,:,k]
                    a[-p1+i,:,k]      = x[i,:,k]
                    a[-p1+i,:,-p3+k]  = x[i,:,k]
            # ...
            for i in range(p1):
                    a[-p1+i,:,p3:-p3] = x[i,:,p3:]
            for k in range(p3):
                    a[p1:-p1,:,-p3+k] = x[p1:,:,k]
            return a 

          elif periodic == [False, True, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0]
            n2      = x.shape[1] + p2
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:-p2,:-p3]  = x[:,:,:]            
            for j in range(p2):
                for k in range(p3):
                    a[:,j,-p3+k]      = x[:,j,k]
                    a[:,-p2+j,k]      = x[:,j,k]
                    a[:,-p2+j,-p3+k]  = x[:,j,k]         
            # ...
            for j in range(p2):
                    a[:,-p2+j,p3:-p3] = x[:,j,p3:]
            for k in range(p3):
                    a[:,p2:-p2,-p3+k] = x[:,p2:,k]
            return a
            
          elif periodic == [False, False, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] 
            n2      = x.shape[1]
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:,:-p3]  = x[:,:,:]            
            for k in range(p3):
                    a[:,:,-p3+k] = x[:,:,k]
            return a

          elif periodic == [False, True, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] 
            n2      = x.shape[1] + p2
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:-p2,:]  = x[:,:,:]            
            for j in range(p2):
                    a[:,-p2+j,:] = x[:,j,:]
            return a
            
          elif periodic == [True, False, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] 
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:,:]  = x[:,:,:]            
            for i in range(p1):
                    a[-p1+i,:,:] = x[i,:,:]
            return a
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                

        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')