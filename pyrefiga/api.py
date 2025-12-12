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


from scipy.sparse import coo_matrix
from .            import nitsche_core as  core


def eliminated_rows_cols(n1, n2, 
                         pd1, pd2, pd3, pd4,    # row slice: i1 in [pd1:pd2], i2 in [pd3:pd4]
                         d1, d2, d3, d4):       # col slice: j1 in [d1:d2],  j2 in [d3:d4]
    """
    Given a 4D slice M[pd1:pd2, pd3:pd4, d1:d2, d3:d4], return the eliminated
    global rows and columns relative to (n1*n2 × n1*n2) global matrix.
    """

    # --- all possible row indices (i1,i2) ---
    all_i1 = np.arange(n1)
    all_i2 = np.arange(n2)

    # kept rows
    keep_i1 = np.arange(pd1, pd2)
    keep_i2 = np.arange(pd3, pd4)

    # eliminated rows = complement
    elim_rows = []
    for i2 in all_i2:
        for i1 in all_i1:
            if not (pd1 <= i1 < pd2 and pd3 <= i2 < pd4):
                elim_rows.append(i1 + n1 * i2)
    elim_rows = np.array(elim_rows, dtype=int)

    # --- all possible col indices (j1,j2) ---
    all_j1 = np.arange(n1)
    all_j2 = np.arange(n2)

    # kept columns: j1 in [d1:d2], j2 in [d3:d4]
    elim_cols = []
    for j2 in all_j2:
        for j1 in all_j1:
            if not (d1 <= j1 < d2 and d3 <= j2 < d4):
                elim_cols.append(j1 + n1 * j2)
    elim_cols = np.array(elim_cols, dtype=int)

    return elim_rows, elim_cols


class StencilNitsche(object):
    """
    Nitsche's Stencil Matrices in n-dimensional stencil format for multipatch IGA.

    Diagonal blocks: standard single-patch StencilMatrix.
    Off-diagonal blocks: Nitsche interface coupling (can be diagonal or sparse).
    """
    def __init__(self, V, W, dirichlet, interfaces):
        assert isinstance( V, TensorSpace)
        assert isinstance( W, TensorSpace)

        nmp            = len(dirichlet)
        # -------
        self._pads     = V.degree
        self._ndim     = V.dim
        self._domain   = V
        self._codomain = W
        self._nmp      = nmp  # number of patches
        self._type     = V.vector_space.dtype
        self._nbs      = V.vector_space.npts
        # ------
        elim_index     = np.zeros((nmp,self._ndim, 2), dtype = int) 
        for i in range(nmp):
            for j in range(self._ndim):
                elim_index[i,j,0] = 1             if dirichlet[i][j][0] else 0
                elim_index[i,j,1] = V.nbasis[j]-1 if dirichlet[i][j][1] else V.nbasis[0]
        # Build nbasis for each patch
        nbasis         = []
        for j in range(nmp):
            nb = (elim_index[j,0,1]-elim_index[j,0,0])
            for i in range(1, V.dim):
                nb = nb * (elim_index[j,i,1]-elim_index[j,i,0]) 
            nbasis.append(nb)
        self._Nitshedim = (sum(nbasis), sum(nbasis))
        self._nbasis    = nbasis
        self.elim_index = elim_index # [nmpatch, dim, 2] local matrix start from
        self.interfaces = interfaces
        self.dirichlet  = dirichlet
        #
        #... computes coeffs for Nitsche's method
        stab          = 4.*( V.degree[0] + V.dim ) * ( V.degree[0] + 1 )
        m_h           = (V.nbasis[0]*V.nbasis[1])
        self.Kappa    = 2.*stab*m_h
        # ...
        self.normS    = 0.5
        #------------------------------------------------------
        #Build a global multipatch sparse matrix in COO format.
        #Diagonal blocks: full patch stencil.
        #Off-diagonal blocks: Nitsche coupling (diagonal entries only).
        #------------------------------------------------------
        rows, cols, data = [], [], []
        self.stencilNitsche = coo_matrix(
            (data, (rows, cols)),
            shape = self._Nitshedim,
            dtype = self._type
        )
        self.stencilNitsche.eliminate_zeros()
        #-------------------------------
        # .. assemble Nitsche's matrices
        #-------------------------------
        self.assemble_nitsche2dDiag    = partial(assemble_matrix, core.assemble_matrix_diagnitsche)
        self.assemble_nitsche2doffDiag = partial(assemble_matrix, core.assemble_matrix_offdiagnitsche)        
    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._domain

    # ...
    @property
    def codomain( self ):
        return self._codomain

    #...
    def apply_NitscheDirichlet(self, x, nb_r, nb_c):
        #----------------------------------------------------------------------------------
        # .. apply dirichlet with respect to v2(id_r)*u1(id_c)
        #----------------------------------------------------------------------------------
        n1,n2  = self._domain.nbasis
        id_r   = nb_r-1
        id_c   = nb_c-1
        vecx   = np.arange(n1)
        vecy   = np.arange(n2)
        rows_D = [] 
        # boundary x = 0
        if self.dirichlet[id_r][0][0]:
            rows_D +=  list(0+n1*vecy)
            print(".x0")
        # boundary x = 1
        if self.dirichlet[id_r][0][1]:
            rows_D +=  list(n1-1+n1*vecy)
            print(".x1")
        # boundary y = 0
        if self.dirichlet[id_r][1][0]:
            rows_D +=  list(vecx)
            print(".y0")
        # boundary y = 1
        if self.dirichlet[id_r][1][1]:
            rows_D +=  list(vecx+n1*(n2-1))
            print(".y1")
        cols_D = [] 
        # boundary x = 0
        if self.dirichlet[id_c][0][0]:
            cols_D +=  list(0+n1*vecy)
            print("..x0")
        # boundary x = 1
        if self.dirichlet[id_c][0][1]:
            cols_D +=  list(n1-1+n1*vecy)
            print("..x1")
        # boundary y = 0
        if self.dirichlet[id_c][1][0]:
            cols_D +=  list(vecx)
            print("..y0")
        # boundary y = 1
        if self.dirichlet[id_c][1][1]:
            cols_D +=  list(vecx+n1*(n2-1))
            print("..y1")

        #indeces for elimination
        d1 = 1    if self.dirichlet[id_c][0][0] else 0 
        d2 = n1-1 if self.dirichlet[id_c][0][1] else n1
        d3 = 1    if self.dirichlet[id_c][1][0] else 0 
        d4 = n2-1 if self.dirichlet[id_c][1][1] else n2
        #indeces for elimination
        elim_index = 1    if self.dirichlet[id_r][0][0] else 0 
        pd2 = n1-1 if self.dirichlet[id_r][0][1] else n1
        pd3 = 1    if self.dirichlet[id_r][1][0] else 0 
        pd4 = n2-1 if self.dirichlet[id_r][1][1] else n2
        print("before doing anything---",x.toarray().reshape((n1*n2,n1*n2)) )
        M   = x.toarray().reshape((n1,n2,n1,n2))[elim_index:pd2,pd3:pd4,d1:d2,d3:d4].reshape(((pd2-elim_index)*(pd4-pd3), (d2-d1)*(d4-d3)))
        print("----before doing anything: matrix shape ",x.shape, "rows to eliminate", rows_D,"cols to eliminate", cols_D)
        Mr = eliminate_rows_cols(x, rows_D, cols_D)
        print("exact",M)
        print("shape after appling rows cols elimination",Mr.shape)
        print("after appling rows cols elimination",Mr)

        import matplotlib.pyplot as plt
        import scipy.sparse as sp
        plt.spy(M, markersize=2, color= 'g')
        plt.show()
        return eliminate_rows_cols(x, rows_D, cols_D)
    #...
    def correct_indexation(self, stiffnessoffdiag, nb_patch):
        '''
        Docstring pour correct_indexation 
        [i,j, i-p:i+p, j-p,j+p] to [i,j, n-i-p:n-i+p, j-p,j+p]
        
        :param stiffnessoffdiag: matrix off diagonal
        :param nb_patch patch number
        '''
        # Shortcuts
        nr = stiffnessoffdiag._codomain.npts
        nd = stiffnessoffdiag._ndim
        nc = stiffnessoffdiag._domain.npts
        ss = stiffnessoffdiag._codomain.starts
        pp = stiffnessoffdiag._codomain.pads

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
        if self.interfaces[nb_patch] == 1 or self.interfaces[nb_patch] == 2:
            for (index,value) in np.ndenumerate( stiffnessoffdiag._data[local] ):

                # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

                xx = index[:nd]  # x=i-s
                ll = index[nd:]  # l=p+k

                ii = [s+x for s,x in zip(ss,xx)]
                jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self._pads)]
                #...correct index ix -> nx-1-ix
                jj[0] = nc[0]-1-jj[0]

                I = ravel_multi_index( ii, dims=nr, order='C' )
                J = ravel_multi_index( jj, dims=nc, order='C' )

                rows.append( I )
                cols.append( J )
                data.append( value )
        else:
            for (index,value) in np.ndenumerate( stiffnessoffdiag._data[local] ):

                # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

                xx = index[:nd]  # x=i-s
                ll = index[nd:]  # l=p+k

                ii = [s+x for s,x in zip(ss,xx)]
                jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self._pads)]
                #...correct index iy -> ny-1-iy
                jj[-1] = nc[-1]-1-jj[-1]

                I = ravel_multi_index( ii, dims=nr, order='C' )
                J = ravel_multi_index( jj, dims=nc, order='C' )

                rows.append( I )
                cols.append( J )
                data.append( value )

        M = coo_matrix(
                (data,(rows,cols)),
                shape = [np.prod(nr),np.prod(nc)],
                dtype = self._type
        )
        M.eliminate_zeros()
        return M
    #...
    def addNitscheoffDiag(self, u11_mph, u12_mph, u21_mph, u22_mph):
        '''
        Docstring pour addNitscheoffDiag
        
        :param u11_mph: mapping component
        :param u12_mph: Description
        :param u21_mph: Description
        :param u22_mph: Description
        '''
        nb_patch   = 1
        nb_patch_n = 2

        stiffnessoffdiag = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
        self.assemble_nitsche2doffDiag(self._domain, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, value=[self._domain.omega[0],self._domain.omega[1], self.interfaces[nb_patch-1], self.Kappa, self.normS], out = stiffnessoffdiag)
        #... correct coo matrix        
        stiffnessoffdiag = self.correct_indexation(stiffnessoffdiag, nb_patch)

        import matplotlib.pyplot as plt
        import scipy.sparse as sp
        plt.spy(stiffnessoffdiag, markersize=2)
        plt.show()

        #... apply dirichlet
        stiffnessoffdiag = self.apply_NitscheDirichlet(stiffnessoffdiag, nb_patch_n, nb_patch) # v2*u1
        import matplotlib.pyplot as plt
        import scipy.sparse as sp
        plt.spy(stiffnessoffdiag, markersize=2)
        plt.show()
        return stiffnessoffdiag
        # self.appendBlock(stiffnessoffdiag, nb_patch_n, nb_patch)
        #..
    # ...
    def tosparse( self ):
        return self.stencilNitsche

    # #...
    def applyNitsche(self, stiffness, u11_mph, u12_mph, id_patch):
        '''
        Docstring pour applyNitsche for diagonal matrices
        
        :param self: Description
        :param stiffness: stifness matrix 
        :param u11_mph: mapping correspond to id_patch comp 1
        :param u12_mph: mapping correspond to id_patch comp 2
        :param id_patch: patch number start from 1
        '''
        if not (1 <= id_patch <= self._nmp):
            raise ValueError(f"id_patch={id_patch} out of range 1..{self._nmp}")

        self.assemble_nitsche2dDiag(self._domain, fields=[u11_mph, u12_mph], knots=True, value=[self._domain.omega[0],self._domain.omega[1], self.interfaces[id_patch-1], self.Kappa, self.normS], out = stiffness)
        #..

    #--------------------------------------
    # append block COO matrix
    #--------------------------------------
    def appendBlock(self, B, nb_patch, nb_patch_n = None):
        '''
        assemble block matrices B in global Nitsche's matrix
        
        :param B: stiffness matrix
        :param nb_patch: patch number
        :param nb_patch_n: patch number of neighbor patch
        '''
        if not (1 <= nb_patch <= self._nmp):
            raise ValueError(f"id_patch={nb_patch} out of range 1..{self._nmp}")

        if isinstance(B, StencilMatrix):
            B = B.tosparse()

        # compute position of block matrix in global Nitsche's matrix
        row = 0
        for i in range(nb_patch-1):
            row += self._nbasis[i]
        col = row
        if nb_patch_n is not None :
            col = 0
            for i in range(nb_patch_n-1):
                col += self._nbasis[i]
        # ...
        self.stencilNitsche += coo_matrix(
            (B.data.copy(), (row+B.row.copy(), col+B.col.copy())),
            shape = self._Nitshedim,
            dtype = self._type
        )

        self.stencilNitsche.eliminate_zeros()

#============================================================================== TODO STAY IN STENCIL FORMAT To delet
def apply_dirichlet_setdiag(V, x, dirichlet = True, dirichlet_patch2 = None, update = None):
    """
    Applies dirichlet boundary conditions to a matrix or vector by elimination.

    dirichlet can take different forms depending on how boundary conditions are specified:
    A single boolean (True or False) meaning the same condition applies to all boundaries.
    A list of booleans, e.g. [True, False], specifying the condition for each direction in 1D or 2D.
    A nested list of booleans, specifying dirichlet conditions in multiple dimensions:
        2D example: [[True, False], [True, True]]
        3D example: [[True, False], [True, True], [True, True]]

    Parameters
    ----------
    V : TensorSpace
        The function space containing basis and dimension information.
    x : StencilMatrix or StencilVector
        The matrix or vector to which dirichlet conditions are applied.
    dirichlet : list or tuple, optional
        Specifies which boundaries have dirichlet conditions (default: None).
    dirichlet_patch2 : list or tuple, optional
        Specifies a second patch for dirichlet elimination (default: None).
    update: StencilVector
        Updates the boundary values of the solution using the exact dirichlet data.
    Returns
    -------
    ndarray
        The matrix or vector with dirichlet boundaries applied and reshaped accordingly.
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
            #... apply dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2,d3:d4].reshape((d2-d1)*(d4-d3))
                else:
                    x   = x[d1:d2,d3:d4,d1:d2,d3:d4].reshape(((d2-d1)*(d4-d3), (d2-d1)*(d4-d3)))
            else:
                #indeces for elimination
                elim_index = 1    if dirichlet_patch2[0][0] else 0 
                pd2 = n1-1 if dirichlet_patch2[0][1] else n1
                pd3 = 1    if dirichlet_patch2[1][0] else 0 
                pd4 = n2-1 if dirichlet_patch2[1][1] else n2
                x   = x[elim_index:pd2,pd3:pd4,d1:d2,d3:d4].reshape(((pd2-elim_index)*(pd4-pd3), (d2-d1)*(d4-d3)))
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
            #... apply dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2,d3:d4,d5:d6].reshape((d2-d1)*(d4-d3)*(d6-d5))
                else:
                    x   = x[d1:d2,d3:d4,d5:d6,d1:d2,d3:d4,d5:d6].reshape(((d2-d1)*(d4-d3)*(d6-d5), (d2-d1)*(d4-d3)*(d6-d5)))
            else:
                #indeces for elimination
                elim_index = 1    if dirichlet_patch2[0][0] else 0 
                pd2 = n1-1 if dirichlet_patch2[0][1] else n1
                pd3 = 1    if dirichlet_patch2[1][0] else 0 
                pd4 = n2-1 if dirichlet_patch2[1][1] else n2
                pd5 = 1    if dirichlet_patch2[2][0] else 0 
                pd6 = n3-1 if dirichlet_patch2[2][1] else n3
                x   = x[elim_index:pd2,pd3:pd4,pd5:pd6,d1:d2,d3:d4,d5:d6].reshape(((pd2-elim_index)*(pd4-pd3)*(pd6-pd5), (d2-d1)*(d4-d3)*(d6-d5)))
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
            #... apply dirichlet
            if dirichlet_patch2 is None:
                if rhs:
                    x   = x[d1:d2].reshape((d2-d1))
                else:
                    x   = x[d1:d2,d1:d2].reshape(((d2-d1), (d2-d1)))
            else:
                #indeces for elimination
                elim_index = 1    if dirichlet_patch2[0] else 0 
                pd2 = n1-1 if dirichlet_patch2[1] else n1
                x   = x[elim_index:pd2,pd3:pd4,d1:d2,d3:d4].reshape(((pd2-elim_index), (d2-d1)))
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
            #... apply dirichlet
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
            #... apply dirichlet
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
            #... apply dirichlet
            u[:]     =   update[:] 
            u[d1:d2] = x.reshape(d2-d1)
            return  u
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')
#==============================================================================
from scipy.sparse import coo_matrix
import numpy as np
import numpy as np
from scipy.sparse import coo_matrix

def eliminate_rows_cols(A: coo_matrix, rows_D, cols_D):
    """
    Remove specified rows and columns from a COO matrix
    and shift indices accordingly.

    Parameters
    ----------
    A : coo_matrix
        Input sparse matrix.
    rows_D : list[int]
        List of row indices to eliminate.
    cols_D : list[int]
        List of column indices to eliminate.

    Returns
    -------
    A_new : coo_matrix
        New compacted COO matrix with eliminated rows/cols removed
        and indices shifted.
    """

    A = A.tocoo()

    # Convert to numpy arrays
    rows_D = np.array(rows_D, dtype=int)
    cols_D = np.array(cols_D, dtype=int)

    # --- Build KEEP sets ---
    all_rows = np.arange(A.shape[0])
    all_cols = np.arange(A.shape[1])

    keep_rows = np.setdiff1d(all_rows, rows_D)
    keep_cols = np.setdiff1d(all_cols, cols_D)

    # --- Build mapping from old → new ---
    row_map = -np.ones(A.shape[0], dtype=int)
    col_map = -np.ones(A.shape[1], dtype=int)

    row_map[keep_rows] = np.arange(len(keep_rows))
    col_map[keep_cols] = np.arange(len(keep_cols))

    # --- Filter out eliminated rows/cols ---
    mask = (row_map[A.row] != -1) & (col_map[A.col] != -1)

    new_rows = row_map[A.row[mask]]
    new_cols = col_map[A.col[mask]]
    new_data = A.data[mask]

    # --- Build new matrix ---
    A_new = coo_matrix(
        (new_data, (new_rows, new_cols)),
        shape=(len(keep_rows), len(keep_cols))
    )

    return A_new

def apply_dirichlet(V, x, dirichlet = True, update = None):
    """
    Applies dirichlet boundary conditions to a matrix or vector by elimination.

    dirichlet can take different forms depending on how boundary conditions are specified:
    A single boolean (True or False) meaning the same condition applies to all boundaries.
    A list of booleans, e.g. [True, False], specifying the condition for each direction in 1D or 2D.
    A nested list of booleans, specifying dirichlet conditions in multiple dimensions:
        2D example: [[True, False], [True, True]]
        3D example: [[True, False], [True, True], [True, True]]

    Parameters
    ----------
    V : TensorSpace
        The function space containing basis and dimension information.
    x : StencilMatrix or StencilVector
        The matrix or vector to which dirichlet conditions are applied.
    dirichlet : list or tuple, optional
        Specifies which boundaries have dirichlet conditions (default: True).
    dirichlet_patch2 : list or tuple, optional
        Specifies a second patch for dirichlet elimination (default: False).
    update: StencilVector
        Updates the boundary values of the solution using the exact dirichlet data.
    Returns
    -------
    ndarray
        The matrix or vector with dirichlet boundaries applied and reshaped accordingly.
    """
    if dirichlet is True :
        if V.dim == 1:
            dirichlet = [dirichlet, dirichlet]
        elif V.dim == 2:
            dirichlet = [[dirichlet, dirichlet],[dirichlet, dirichlet]]
        elif V.dim == 3:
            dirichlet = [[dirichlet, dirichlet],[dirichlet, dirichlet],[dirichlet, dirichlet]]
    elif dirichlet[0] is True and V.dim >1:
        if V.dim == 2:
            dirichlet = [[dirichlet[0], dirichlet[0]],[dirichlet[1], dirichlet[1]]]
        elif V.dim == 3:
            dirichlet = [[dirichlet[0], dirichlet[0]],[dirichlet[1], dirichlet[1]],[dirichlet[2], dirichlet[2]]]

    if update is None :
        #--------------------------------------------------------------------------
        if isinstance(x, StencilMatrix):
            if V.dim == 1:
                n1 = V.nbasis
                rows_D = [0, n1-1] 

                return eliminate_rows_cols(x.tosparse(), rows_D, rows_D)

            elif V.dim == 2:
                n1,n2  = V.nbasis
                vecx   = np.arange(n1)
                vecy   = np.arange(n2)
                rows_D = [] 
                # boundary x = 0
                if dirichlet[0][0]:
                    rows_D +=  list(0+n1*vecy)
                # boundary x = 1
                if dirichlet[0][1]:
                    rows_D +=  list(n1-1+n1*vecy)
                # boundary y = 0
                if dirichlet[1][0]:
                    rows_D +=  list(vecx)
                # boundary y = 1
                if dirichlet[1][1]:
                    rows_D +=  list(vecx+n1*(n2-1))
                return eliminate_rows_cols(x.tosparse(), rows_D, rows_D)

            elif V.dim == 3:
                n1,n2,n3 = V.nbasis
                vecx = np.arange(n1)
                vecy = np.arange(n2)
                vecz = np.arange(n3)
                rows_D = [] 
                # ... resetting bnd dof to 0, x = 0
                if dirichlet[0][0]:
                    rows_D +=  list(n1*vecy+n1*n2*vecz)
                # ... resetting bnd dof to 0, x = 1
                if dirichlet[0][1]:
                    rows_D +=  list(n1-1+n1*vecy+n1*n2*vecz)
                # ... resetting bnd dof to 0, y = 0
                if dirichlet[1][0]:
                    rows_D +=  list(vecx+n1*n2*vecz)
                # ... resetting bnd dof to 0, y = 1
                if dirichlet[1][1]:
                    rows_D +=  list(vecx+n1*(n2-1)+n1*n2*vecz)
                # ... resetting bnd dof to 0, z = 0
                if dirichlet[2][0]:
                    rows_D +=  list(vecx+n1*vecy)
                # ... resetting bnd dof to 0, z = 1
                if dirichlet[2][1]:
                    rows_D +=  list(vecx+n1*vecy+n1*n2*(n3-1))
                return eliminate_rows_cols(x.tosparse(), rows_D, rows_D)
            else :
                raise NotImplementedError('Only 1d, 2d and 3d are available')

        elif isinstance(x, StencilVector):
            if V.dim == 1:
                n1 = V.nbasis

                #indeces for elimination
                d1 = 1    if dirichlet[0] else 0 
                d2 = n1-1 if dirichlet[1] else n1

                x   = x.toarray().reshape(n1)
                x   = x[d1:d2].reshape((d2-d1))
                return x

            elif V.dim == 2:
                n1,n2 = V.nbasis
                #indeces for elimination
                d1 = 1    if dirichlet[0][0] else 0 
                d2 = n1-1 if dirichlet[0][1] else n1
                d3 = 1    if dirichlet[1][0] else 0 
                d4 = n2-1 if dirichlet[1][1] else n2

                x   = x.toarray().reshape((n1,n2))
                #... apply dirichlet
                x   = x[d1:d2,d3:d4].reshape((d2-d1)*(d4-d3))
                return x

            elif V.dim == 3:
                n1, n2, n3 = V.nbasis

                #indeces for elimination
                d1 = 1    if dirichlet[0][0] else 0 
                d2 = n1-1 if dirichlet[0][1] else n1
                d3 = 1    if dirichlet[1][0] else 0 
                d4 = n2-1 if dirichlet[1][1] else n2
                d5 = 1    if dirichlet[2][0] else 0 
                d6 = n3-1 if dirichlet[2][1] else n3

                x   = x.toarray().reshape((n1,n2,n3))
                x   = x[d1:d2,d3:d4,d5:d6].reshape((d2-d1)*(d4-d3)*(d6-d5))
                return x

            else:
                raise NotImplementedError('Only 1d, 2d and 3d are available')

        else:
            raise TypeError('Expecting StencilMatrix or StencilVector')
    
    else:
        if isinstance(update, StencilVector):
            pass
        else:
            raise NotImplementedError('Not available only StencilVector')
        u   = StencilVector(V.vector_space)
        # ...
        if V.dim == 1:
            n1      = V.nbasis
            #indeces for elimination
            d1 = 1    if dirichlet[0] else 0 
            d2 = n1-1 if dirichlet[1] else n1
            #... apply dirichlet
            u[:]     =   update[:] 
            u[d1:d2] = x.reshape(d2-d1)
            return  u
        elif V.dim ==2:
            n1, n2  = V.nbasis
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            #... apply dirichlet
            u[:,:]         = update[:,:]
            u[d1:d2,d3:d4] = x.reshape((d2-d1),(d4-d3))
            return  u
        elif V.dim == 3:
            n1, n2, n3  = V.nbasis
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            d5 = 1    if dirichlet[2][0] else 0 
            d6 = n3-1 if dirichlet[2][1] else n3
            #... apply dirichlet
            u[:,:,:]             =   update[:,:,:] 
            u[d1:d2,d3:d4,d5:d6] = x.reshape((d2-d1),(d4-d3),(d6-d5))
            return  u
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')


#============================================================================== TODO SHOULD STAY IN STENCIL FORMAT
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