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
from .utilities   import pyrefMultpatch

#==============================================================================
class StencilNitsche(object):
    """
    Nitsche's Stencil Matrices in n-dimensional stencil format for multipatch IGA.

    Diagonal blocks: standard single-patch StencilMatrix.
    Off-diagonal blocks: Nitsche interface coupling (can be diagonal or sparse).
    Parameters
    ----------
    V : TensorSpace
        The function space containing basis and dimension information.
    W : TensorSpace
        The function space containing basis and dimension information.
    pyrefMP : pyrefMultpatch
        The multipatch geometry information.
    u_d: list of StensilVector 
        Containes Dirichlet projection in V
    Isoparametric : bool, optional
        If True, use the same space for geometry and solution (default is True).
    Attributes
    ----------
    domain : TensorSpace
        The domain function space.
    codomain : TensorSpace  
        The codomain function space.
    Methods 
    """
    def __init__(self, V, W, pyrefMP, u_d = None, Isoparametric = True):
        assert isinstance( V, TensorSpace)
        assert isinstance( W, TensorSpace)
        assert isinstance( pyrefMP, pyrefMultpatch)

        nb_patches     = pyrefMP.nb_patches
        # -------
        if u_d is None:
            u_d = [StencilVector(V.vector_space) for _ in range(nb_patches)]
        self.u_d         = u_d
        self._pads       = V.degree
        self._ndim       = V.dim
        self._domain     = V
        self._codomain   = W
        Vmp1, Vmp2       = pyrefMP.UnifSplineSpace(mesh=V.mesh, quad_degree = (V.weights[0].shape[1]-1,V.weights[1].shape[1]-1))
        self._mpdomain   = TensorSpace(Vmp1, Vmp2)
        self._Tdomain    = TensorSpace(V.spaces[0],V.spaces[1], Vmp1, Vmp2)
        self._nb_patches = nb_patches  # number of patches
        self._type       = V.vector_space.dtype
        self._nbs        = V.vector_space.npts
        # ------
        elim_index     = np.zeros((nb_patches,self._ndim, 2), dtype = int) 
        for i in range(nb_patches):
            for j in range(self._ndim):
                elim_index[i,j,0]    = 1             if pyrefMP.getDirPatch(i+1)[j][0] else 0
                elim_index[i,j,1]    = V.nbasis[j]-1 if pyrefMP.getDirPatch(i+1)[j][1] else V.nbasis[j]
        # Build nbasis for each patch
        nbasis         = []
        block_index    = []
        block_index.append(0)
        for j in range(nb_patches):
            nb = (elim_index[j,0,1]-elim_index[j,0,0])
            for i in range(1, V.dim):
                nb = nb * (elim_index[j,i,1]-elim_index[j,i,0]) 
            nbasis.append(nb)
            block_index.append(sum(nbasis))
        self._Nitshedim   = (sum(nbasis), sum(nbasis))
        self._nbasis      = nbasis
        self._block_index = block_index # position of each block
        self.elim_index   = elim_index # [nb_patches, dim, 2] local matrix start from
        self.mp           = pyrefMP
        #
        #... computes coeffs for Nitsche's method
        stab               = 4.*( V.degree[0] + V.dim ) *( V.degree[1] + V.dim ) * ( V.degree[0] + 1 )
        m_h                = (V.nbasis[0]*V.nbasis[1])
        self.Kappa         = 2.*stab*m_h
        # ...
        self.normS         = 0.5
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
        # -----------------
        #... Dirichlet part
        #------------------
        self.b_dir  = np.zeros(self._Nitshedim[0])
        #-------------------------------
        # .. assemble Nitsche's matrices
        #-------------------------------
        if Isoparametric:
            #... using same spaces for FE analysis V and  Multipatch W
            self.assemble_nitsche2dDiag        = partial(assemble_matrix, core.assemble_matrix_diagnitsche)
            self.assemble_nitsche2dUnderDiag   = partial(assemble_matrix, core.assemble_matrix_offdiagnitsche)        
        else:
            #... using different spaces for FE analysis V and  Multipatch W
            self.assemble_nitsche2dDiag        = partial(assemble_matrix, core.assemble_matrix_DiffSpacediagnitsche)
            self.assemble_nitsche2dUnderDiag   = partial(assemble_matrix, core.assemble_matrix_DiffSpaceoffdiagnitsche)
            self.assemble_nitsche2dDirichlet   = partial(assemble_vector, core.assemble_vector_Dirichlet)

        self._newdim    = (0,0)
        self.new_id     = {}
        self.old_id     = {}
    #---------------------------------------------
    # representative_dof -> list of equivalent dofs
    #---------------------------------------------
    def to_mergeDOFs(self):
        equiv      = {}     # rep -> set(dofs)
        dof_to_rep = {}     # dof -> rep
        for interface in self.mp.getInterfaces():
            # ... get interface and patch numbers
            patch_nb       = interface[0] 
            patch_nb_n     = interface[1]
            # ... get interface mappings
            interface_like = interface[2][1]
            #...rows
            pd1 = self.elim_index[patch_nb_n-1,0,0]# for x
            pd2 = self.elim_index[patch_nb_n-1,0,1]
            pd3 = self.elim_index[patch_nb_n-1,1,0]# for y
            pd4 = self.elim_index[patch_nb_n-1,1,1]
            #... cols
            d1 = self.elim_index[patch_nb-1,0,0]
            d2 = self.elim_index[patch_nb-1,0,1]
            d3 = self.elim_index[patch_nb-1,1,0]
            d4 = self.elim_index[patch_nb-1,1,1]
            pw = self._block_index[patch_nb-1]
            cpw= self._block_index[patch_nb_n-1]
            #... [patche1: 2, patche2 : 1]
            if interface_like == 1: # x = 0
                # add test if 
                assert(pd3==d3 and pd4==d4) # conform multipatch
                for j in range(0,pd4-pd3):
                    dof_p = cpw + 0*(pd4-pd3)      + j
                    dof_q = pw + (d2-d1-1)*(d4-d3) + j

                    rep = min(dof_p, dof_q)
                    equiv.setdefault(rep, set()).update({dof_p, dof_q})

                    dof_to_rep[dof_p] = rep
                    dof_to_rep[dof_q] = rep

            if interface_like == 2: # x = 1
                assert(pd3==d3 and pd4==d4) # conform multipatch
                for j in range(0,pd4-pd3):
                    dof_p = cpw + (pd2-pd1-1)*(pd4-pd3) + j
                    dof_q = pw +   0*(d4-d3)            + j

                    rep = min(dof_p, dof_q)
                    equiv.setdefault(rep, set()).update({dof_p, dof_q})

                    dof_to_rep[dof_p] = rep
                    dof_to_rep[dof_q] = rep
            if interface_like == 3: # y = 0
                assert(pd1==d1 and pd2==d2) # conform multipatch
                for i in range(0,pd2-pd1):
                    dof_p = cpw + i*(pd4-pd3) + 0
                    dof_q = pw +  i*(d4-d3)   + (d4-d3-1)

                    rep = min(dof_p, dof_q)
                    equiv.setdefault(rep, set()).update({dof_p, dof_q})

                    dof_to_rep[dof_p] = rep
                    dof_to_rep[dof_q] = rep
            if interface_like == 4: # y = 1
                assert(pd1==d1 and pd2==d2) # conform multipatch
                for i in range(0,pd2-pd1):
                    dof_p = cpw + i*(pd4-pd3) + (pd4-pd3-1)
                    dof_q = pw +  i*(d4-d3)   + 0

                    rep = min(dof_p, dof_q)
                    equiv.setdefault(rep, set()).update({dof_p, dof_q})

                    dof_to_rep[dof_p] = rep
                    dof_to_rep[dof_q] = rep
        new_id  = {}
        old_id  = {}
        current = 0
        # IMPORTANT: iterate over all DOFs appearing in matrix
        all_dofs = set(self.stencilNitsche.tocoo().row) | set(self.stencilNitsche.tocoo().col)
        for old_dof in sorted(all_dofs):
            # for old_dof in range(self._Nitshedim[0]):
            if old_dof in new_id:
                continue
            if old_dof in dof_to_rep:
                rep = dof_to_rep[old_dof]
                old_id[current] = set()
                for d in equiv[rep]:
                    new_id[d] = current
                    old_id[current].add(d)
            else:
                new_id[old_dof] = current
                old_id[current] = {old_dof}
            current += 1
        self._newdim    = (current, current)
        self.new_id     = new_id
        self.old_id     = old_id
        # ...
        return 
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
    # ...
    def tosparse( self ):
        return self.stencilNitsche
    #...
    def NitscheMerge(self):
        ''''
        NitscheMerge: merge DoFs of the same interface
        '''
        #... Compute relationship between old and new DoFs
        self.to_mergeDOFs()
        rows, cols, data = [], [], []

        for i, j, v in zip(self.stencilNitsche.tocoo().row,
                        self.stencilNitsche.tocoo().col,
                        self.stencilNitsche.data):
            rows.append(self.new_id[i])
            cols.append(self.new_id[j])
            data.append(v)

        stencilMergNitsche = coo_matrix(
            (data, (rows, cols)),
            shape=self._newdim,
            dtype=self._type
        )

        stencilMergNitsche.sum_duplicates()

        return stencilMergNitsche
    #... merg rhs
    def NitscheMergeRHS(self):
        '''
        NitscheMergeRHS:  merge rhs DoFs of the same interface
        '''
        rhsMerged = np.zeros(self._newdim[0], dtype=self._type)

        for old_dof, value in enumerate(self.b_dir):
            I = self.new_id[old_dof]
            rhsMerged[I] += value   # SUM contributions

        return rhsMerged
    #... Extract solution
    def NitscheextractSol(self, sol_Dof):
        '''
        NitscheextractSol:  Extract Solution DoFs from master/slave DoFs
        '''
        SolExtracted = np.zeros(self._Nitshedim[0], dtype=self._type)
        for new_dof, olds in self.old_id.items():
            for d in olds:
                SolExtracted[d] = sol_Dof[new_dof]   # Extract contributions
        return SolExtracted
    #...
    def addNitscheoffDiag(self):
        '''
        Docstring pour addNitscheoffDiag

        # :param Vh: FE space & multipatch space
        '''
        # if not isinstance(Vh, TensorSpace):
        #     raise TypeError("Vh must be a TensorSpace")
        for interface in self.mp.getInterfaces():
            # ... get interface and patch numbers
            patch_nb       = interface[0] 
            patch_nb_n     = interface[1]
            # ... get interface mappings
            interface_like = interface[2][0]
            # assemble mappings for patches
            u11_mph, u12_mph = self.mp.getStencilMapping(patch_nb)
            u21_mph, u22_mph = self.mp.getStencilMapping(patch_nb_n)
            #... assemble off diagonal matrix
            stiffnessoffdiag = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
            self.assemble_nitsche2dUnderDiag(self._Tdomain, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, 
                                             value=[self._mpdomain.omega[0],self._mpdomain.omega[1], interface_like, self.Kappa, self.normS], 
                                             out = stiffnessoffdiag)
            #... correct coo matrix
            stiffnessoffdiag = self.collect_offdiagStencilMatrix(stiffnessoffdiag, interface)
            self.appendBlock(stiffnessoffdiag, patch_nb_n, patch_nb)
            self.appendBlock(stiffnessoffdiag.T, patch_nb, patch_nb_n)
        #...

    # #...
    def applyNitsche(self, stiffness, patch_nb):
        '''
        Docstring pour applyNitsche for diagonal matrices
        
        :param self: Description
        :param stiffness: stifness matrix 
        :param patch_nb: patch number start from 1
        '''
        if not (1 <= patch_nb <= self._nb_patches):
            raise ValueError(f"patch_nb={patch_nb} out of range 1..{self._nb_patches}")
        # assemble mappings for patches
        u11_mph, u12_mph = self.mp.getStencilMapping(patch_nb)
        #... get interfaces for a given patch
        interfaces_like = self.mp.getInterfacePatch(patch_nb)
        # ... assemble diagonal matrix
        self.assemble_nitsche2dDiag(self._Tdomain, fields=[u11_mph, u12_mph], knots=True, value=[self._mpdomain.omega[0],self._mpdomain.omega[1], interfaces_like, self.Kappa, self.normS], out = stiffness)
        # ...
        return
    
    #--------------------------------------
    # ... assemble global Nitsche's matrix
    #--------------------------------------
    def assembleNitsche(self):
        '''
        Docstring pour getNitsche
        assemble global Nitsche matrix

        :param self: Description
        '''
        # ...
        for patch_nb in range(1,self.mp.nb_patches+1):
            # ... initial diag stiffness matrix
            stiffness  = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
            # ... assemble 
            self.applyNitsche(stiffness, patch_nb)
            # ... apply dirichlet
            stiffness  = apply_dirichlet(self._domain, stiffness, dirichlet = self.mp.getDirPatch(patch_nb))
            # ... assemble it into globel matrix
            self.appendBlock(stiffness, patch_nb)
            # ...
        self.addNitscheoffDiag()
        #... scaling
        # D = self.scalingNistcheVector()
        # A  = self.stencilNitsche.tocoo(copy=False)
        # self.stencilNitsche.data *= D[A.row] * D[A.col]
        #...
        return self.stencilNitsche
    #-------------------------------------------------
    # assemble Nitsche's Dirichlet contribution
    #-------------------------------------------------
    def assembleNitsche_Dirichlet(self, rhs, patch_nb):
        '''
        Docstring for assembleNitsche_Dirichlet: assemble rhs vector
        
        :param self: Description
        :param u_d: stencile vector
        :param patch_nb: patch number start from 1
        '''
        # assert isinstance(u_d, StencilVector)
        # for patch_nb in range(1, self.mp.nb_patches+1):
        # assemble mappings for patches
        u11_mph, u12_mph = self.mp.getStencilMapping(patch_nb)
        #... get interfaces for a given patch
        interfaces_like = self.mp.getInterfacePatch(patch_nb)
        # ...
        u_tmp = StencilVector(self._domain.vector_space)
        nS = 1.
        self.assemble_nitsche2dDirichlet(self._Tdomain, fields=[u11_mph, u12_mph, self.u_d[patch_nb-1]], knots=True, value=[self._mpdomain.omega[0],self._mpdomain.omega[1], interfaces_like, 0.*self.Kappa, 0.*self.normS, nS], out = u_tmp)
        u_tmp = apply_dirichlet(self._domain, u_tmp, dirichlet = self.mp.getDirPatch(patch_nb))
        self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] = u_tmp[:]
        # ...
        self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] += rhs[:]
        # if patch_nb == self.mp.nb_patches:
        #     self.scalingRhs()
        #     print('hereyy')
        # ...
        return
    #...
    def addNitscheoffDiag_SameSpace(self, u11_mph, u12_mph, u21_mph, u22_mph, interface):
        '''
        Docstring pour addNitscheoffDiag

        # :param Vh: FE space & multipatch space
        :param u11_mph: mapping component
        :param u12_mph: Description
        :param u21_mph: Description
        :param u22_mph: Description
        :param interface: (patch_nb, patch_nbnext, (patchBoundary,patchBoundary_next))
        '''
        # if not isinstance(Vh, TensorSpace):
        #     raise TypeError("Vh must be a TensorSpace")
        # ... get interface and patch numbers
        patch_nb       = interface[0] 
        patch_nb_n     = interface[1]
        # ... get interface mappings
        interface_like = interface[2][0]
        #... assemble off diagonal matrix
        stiffnessoffdiag = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
        self.assemble_nitsche2dUnderDiag(self._domain, fields=[u11_mph, u12_mph, u21_mph, u22_mph], knots=True, 
                                            value=[self._domain.omega[0],self._domain.omega[1], interface_like, self.Kappa, self.normS], 
                                            out = stiffnessoffdiag)
        #... correct coo matrix
        stiffnessoffdiag = self.collect_offdiagStencilMatrix(stiffnessoffdiag, interface)
        self.appendBlock(stiffnessoffdiag, patch_nb_n, patch_nb)
        self.appendBlock(stiffnessoffdiag.T, patch_nb, patch_nb_n)
    # #...
    def applyNitsche_SameSpace(self, stiffness, u11_mph, u12_mph, patch_nb):
        '''
        Docstring pour applyNitsche for diagonal matrices
        
        :param self: Description
        :param stiffness: stifness matrix 
        :param u11_mph: mapping correspond to patch_nb comp 1
        :param u12_mph: mapping correspond to patch_nb comp 2
        :param patch_nb: patch number start from 1
        '''
        if not (1 <= patch_nb <= self._nb_patches):
            raise ValueError(f"patch_nb={patch_nb} out of range 1..{self._nb_patches}")
        #... get interfaces for a given patch
        interfaces_like = self.mp.getInterfacePatch(patch_nb)
        #.. assemble diagonal matrix
        self.assemble_nitsche2dDiag(self._domain, fields=[u11_mph, u12_mph], knots=True, value=[self._domain.omega[0],self._domain.omega[1], interfaces_like, self.Kappa, self.normS], out = stiffness)
        #..
    #====================================
    #... collect off diag NItsch matrices
    #====================================
    def collect_offdiagStencilMatrix(self, stiffnessoffdiag, interface):
        '''
        Docstring for collect_offdiagStencilMatrix 
        [i,j, i-p:i+p, j-p,j+p] to [i,j, n-i-p:n-i+p, j-p,j+p]
        
        :param stiffnessoffdiag: matrix off diagonal
        :param patch_nb patch number
        '''
        # ... get interface and patch numbers
        patch_nb       = interface[0]
        patch_nb_n     = interface[1]
        # ... get interface mappings
        interface_like = interface[2][0]
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
        #...rows
        pd1 = self.elim_index[patch_nb_n-1,0,0]# for x
        pd2 = self.elim_index[patch_nb_n-1,0,1]
        pd3 = self.elim_index[patch_nb_n-1,1,0]# for y
        pd4 = self.elim_index[patch_nb_n-1,1,1]
        #... cols
        d1 = self.elim_index[patch_nb-1,0,0]
        d2 = self.elim_index[patch_nb-1,0,1]
        d3 = self.elim_index[patch_nb-1,1,0]
        d4 = self.elim_index[patch_nb-1,1,1]
        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
        index_coll = 0
        if interface_like == 1 or interface_like == 2:
            index_coll = 0
        else:
            index_coll = -1
        for (index,value) in np.ndenumerate( stiffnessoffdiag._data[local] ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # x=i-s
            ll = index[nd:]  # l=p+k

            ii = [s+x for s,x in zip(ss,xx)]
            jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self._pads)]
            #...correct index ix -> nx-1-ix jj[0] = nc[0]-1-jj[0] #...correct index iy -> ny-1-iy//jj[-1] = nc[-1]-1-jj[-1]
            jj[index_coll] = nc[index_coll]-1-jj[index_coll]
            if ( pd1 <= ii[0] < pd2) and (pd3 <= ii[1] < pd4) and ( d1 <= jj[0] < d2) and (d3 <= jj[1] < d4):
                # correct index
                ii[0] = ii[0]-pd1
                ii[1] = ii[1]-pd3
                jj[0] = jj[0]-d1
                jj[1] = jj[1]-d3
                #...
                I = ravel_multi_index( ii, dims=(pd2-pd1, pd4-pd3), order='C' )
                J = ravel_multi_index( jj, dims=(d2-d1, d4-d3), order='C' )

                rows.append( I )
                cols.append( J )
                data.append( value )

        M = coo_matrix(
                (data,(rows,cols)),
                shape = [self._nbasis[patch_nb_n-1],self._nbasis[patch_nb-1]],
                dtype = self._type
        )
        M.eliminate_zeros()
        return M
    #--------------------------------------
    # append block COO matrix
    #--------------------------------------
    def appendBlock(self, B, patch_nb, patch_nb_n = None):
        '''
        assemble block matrices B in global Nitsche's matrix
        
        :param B: Stencile matrix
        :param patch_nb: patch number
        :param patch_nb_n: patch number of neighbor patch
        '''
        if not (1 <= patch_nb <= self._nb_patches):
            raise ValueError(f"patch_nb={patch_nb} out of range 1..{self._nb_patches}")

        if isinstance(B, StencilMatrix):
            B = B.tosparse()

        # compute position of block matrix in global Nitsche's matrix
        row = self._block_index[patch_nb-1]
        col = self._block_index[patch_nb-1]
        if patch_nb_n is not None :
            col = self._block_index[patch_nb_n-1]
        # ...
        self.stencilNitsche += coo_matrix(
            (B.data.copy(), (row+B.row.copy(), col+B.col.copy())),
            shape = self._Nitshedim,
            dtype = self._type
        )

        self.stencilNitsche.eliminate_zeros()

#==============================================================================
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
    if dirichlet is True or dirichlet is False:
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
                #indeces for elimination
                d1 = 1    if dirichlet[0] else 0 
                d3 = 1    if dirichlet[1] else 0 
                d2 = n1-d1
                d4 = n1-d3
                # Shortcuts
                nd = x._ndim
                nc = x._domain.npts
                ss = x._codomain.starts
                pp = x._codomain.pads

                ravel_multi_index = np.ravel_multi_index

                # COO storage
                rows = []
                cols = []
                data = []
                # Range of data owned by local process (no ghost regions)
                local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
                for (index,value) in np.ndenumerate( x._data[local] ):

                    # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

                    xx = index[:nd]  # x=i-s
                    ll = index[nd:]  # l=p+k

                    ii = [s+x for s,x in zip(ss,xx)]
                    jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,pp)]

                    if ( d1 <= ii[0] < d2) and ( d3 <= jj[0] < d4):
                        # correct index
                        ii[0] = ii[0]-d1
                        jj[0] = jj[0]-d3
                        #...
                        I = ravel_multi_index( ii, dims=(d2-d1), order='C' )
                        J = ravel_multi_index( jj, dims=(d4-d3), order='C' )

                        rows.append( I )
                        cols.append( J )
                        data.append( value )

                M = coo_matrix(
                        (data,(rows,cols)),
                        shape = [(d2-d1),(d4-d3)],
                        dtype = x._domain.dtype
                )
                M.eliminate_zeros()
                return M

            elif V.dim == 2:
                n1,n2  = V.nbasis
                #indeces for elimination
                d1 = 1    if dirichlet[0][0] else 0 
                d2 = n1-1 if dirichlet[0][1] else n1
                d3 = 1    if dirichlet[1][0] else 0 
                d4 = n2-1 if dirichlet[1][1] else n2
                # Shortcuts
                nd = x._ndim
                nc = x._domain.npts
                ss = x._codomain.starts
                pp = x._codomain.pads

                ravel_multi_index = np.ravel_multi_index

                # COO storage
                rows = []
                cols = []
                data = []
                # Range of data owned by local process (no ghost regions)
                local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
                for (index,value) in np.ndenumerate( x._data[local] ):

                    # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

                    xx = index[:nd]  # x=i-s
                    ll = index[nd:]  # l=p+k

                    ii = [s+x for s,x in zip(ss,xx)]
                    jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,pp)]

                    if ( d1 <= ii[0] < d2) and (d3 <= ii[1] < d4) and ( d1 <= jj[0] < d2) and (d3 <= jj[1] < d4):
                        # correct index
                        ii[0] = ii[0]-d1
                        ii[1] = ii[1]-d3
                        jj[0] = jj[0]-d1
                        jj[1] = jj[1]-d3
                        #...
                        I = ravel_multi_index( ii, dims=(d2-d1, d4-d3), order='C' )
                        J = ravel_multi_index( jj, dims=(d2-d1, d4-d3), order='C' )

                        rows.append( I )
                        cols.append( J )
                        data.append( value )

                M = coo_matrix(
                        (data,(rows,cols)),
                        shape = [(d2-d1)*(d4-d3),(d2-d1)*(d4-d3)],
                        dtype = x._domain.dtype
                )
                M.eliminate_zeros()
                return M

            elif V.dim == 3:
                n1,n2,n3 = V.nbasis
                #indeces for elimination
                d1 = 1    if dirichlet[0][0] else 0 
                d2 = n1-1 if dirichlet[0][1] else n1
                d3 = 1    if dirichlet[1][0] else 0 
                d4 = n2-1 if dirichlet[1][1] else n2
                d5 = 1    if dirichlet[2][0] else 0 
                d6 = n3-1 if dirichlet[2][1] else n3
                # Shortcuts
                nd = x._ndim
                nc = x._domain.npts
                ss = x._codomain.starts
                pp = x._codomain.pads

                ravel_multi_index = np.ravel_multi_index

                # COO storage
                rows = []
                cols = []
                data = []
                # Range of data owned by local process (no ghost regions)
                local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
                for (index,value) in np.ndenumerate( x._data[local] ):

                    # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

                    xx = index[:nd]  # x=i-s
                    ll = index[nd:]  # l=p+k

                    ii = [s+x for s,x in zip(ss,xx)]
                    jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,pp)]

                    if ( d1 <= ii[0] < d2) and (d3 <= ii[1] < d4) and (d5 <= ii[2] < d6) and ( d1 <= jj[0] < d2) and (d3 <= jj[1] < d4) and (d5 <= jj[2] < d6):
                        # correct index
                        ii[0] = ii[0]-d1
                        ii[1] = ii[1]-d3
                        ii[2] = ii[2]-d5
                        jj[0] = jj[0]-d1
                        jj[1] = jj[1]-d3
                        jj[2] = jj[2]-d5
                        #...
                        I = ravel_multi_index( ii, dims=(d2-d1, d4-d3, d6-d5), order='C' )
                        J = ravel_multi_index( jj, dims=(d2-d1, d4-d3, d6-d5), order='C' )

                        rows.append( I )
                        cols.append( J )
                        data.append( value )

                M = coo_matrix(
                        (data,(rows,cols)),
                        shape = [(d2-d1)*(d4-d3)*(d6-d5),(d2-d1)*(d4-d3)*(d6-d5)],
                        dtype = x._domain.dtype
                )
                M.eliminate_zeros()
                return M
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
        u   = update
        # ...
        if V.dim == 1:
            n1      = V.nbasis
            #indeces for elimination
            d1 = 1    if dirichlet[0] else 0 
            d2 = n1-1 if dirichlet[1] else n1
            #... apply dirichlet
            u[d1:d2]+= x.reshape(d2-d1)
            return  u
        elif V.dim ==2:
            n1, n2  = V.nbasis
            #indeces for elimination
            d1 = 1    if dirichlet[0][0] else 0 
            d2 = n1-1 if dirichlet[0][1] else n1
            d3 = 1    if dirichlet[1][0] else 0 
            d4 = n2-1 if dirichlet[1][1] else n2
            #... apply dirichlet
            u[d1:d2,d3:d4]+= x.reshape((d2-d1),(d4-d3))
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
            u[d1:d2,d3:d4,d5:d6]+= x.reshape((d2-d1),(d4-d3),(d6-d5))
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