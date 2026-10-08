import numpy as np
from functools import partial

from pyccel  import epyccel

from .linalg import StencilMatrix
from .linalg import StencilVector
from .spaces import TensorSpace

__all__ = ['assemble_matrix', 'assemble_vector', 'assemble_scalar', 'compile_kernel', 'StencilNitsche', 'apply_dirichlet', 'apply_zeros', 'apply_periodic']

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
def assemble_vector(core, V, fields=None, knots = False, value = None, out=None):
    if out is None:
        out = StencilVector(V.vector_space)

    # ...
    args = []
    if not knots:
       if isinstance(V, TensorSpace):
           int_par = 2 if not (V.dim % 3 == 0 and (V.dim != 6 or V.spaces[3].nurbs)) else 3
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
from .            import nitsche_core   as  n_core
from .            import adnitsche_core as  adcore
from .utilities   import pyref_multipatch

#==============================================================================
class StencilNitsche(object):
    """
    Nitsche's Stencil Matrices in n-dimensional stencil format for multipatch IGA.

    Diagonal blocks: standard single-patch StencilMatrix.
    Off-diagonal blocks: Nitsche interface coupling (can be diagonal or sparse).
    Fixed two-dimensional mappings support all edge pairs (1 through 4),
    including reversed traces. Merged trace bases must have matching degrees,
    normalized knots and rational weights. Adaptive mappings use the legacy
    assembly path.
    Parameters
    ----------
    V : TensorSpace
        The function space containing basis and dimension information.
    W : TensorSpace containes spaces for FE and mapping
        The function space containing basis and dimension information.
    pyrefMP : pyref_multipatch
        The multipatch geometry information.
    u_d: list of StensilVector 
        Containes Dirichlet projection in V
    ad_mapping  : class for adaptive mapping
        The adaptive multipatch mapping information
    Attributes
    ----------
    domain : TensorSpace
        The domain function space.
    codomain : TensorSpace  
        The codomain function space.
    Methods 
    """
    def __init__(self, V, W, pyrefMP, u_d = None, ad_mapping = None):
        assert isinstance( V, TensorSpace)
        assert isinstance( W, TensorSpace)
        assert isinstance( pyrefMP, pyref_multipatch)

        nb_patches     = pyrefMP.nb_patches
        # -------
        if u_d is None:
            u_d = [StencilVector(V.vector_space) for _ in range(nb_patches)]
        pyrefMP.propagate_dirichlet_corners(V, u_d)
        self.u_d         = u_d
        self._pads       = V.degree
        self._ndim       = V.dim
        self._domain     = V
        self._Alldomain  = W
        self._mpdomain   = pyrefMP.space
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
        # Edge elimination is rectangular. Isolated prescribed vertices are
        # removed later from the merged correction system, not from whole edges.
        self._corner_dofs = set()
        self._corner_indices = [[] for _ in range(nb_patches)]
        for group in pyrefMP.getDirichletCornerGroups():
            for patch_nb, u, v in group:
                i, j = u*(V.nbasis[0]-1), v*(V.nbasis[1]-1)
                lo, hi = elim_index[patch_nb-1, :, 0], elim_index[patch_nb-1, :, 1]
                if lo[0] <= i < hi[0] and lo[1] <= j < hi[1]:
                    self._corner_indices[patch_nb-1].append((i, j))
                    self._corner_dofs.add(block_index[patch_nb-1]
                                          + (i-lo[0])*(hi[1]-lo[1])+j-lo[1])
        self.mp           = pyrefMP
        self.admp         = ad_mapping
        #... computes coeffs for Nitsche's method
        stab              = 4.*( V.degree[0] + V.dim ) *( V.degree[1] + V.dim ) * ( V.degree[0] + 1 )
        m_h               = (V.nbasis[0]*V.nbasis[1])
        self.Kappa        = 2.*stab*m_h
        # ...
        self.normS        = 0.5
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
        if ad_mapping is not None:
            #... using different spaces for FE analysis V and  Multipatch W
            self.assemble_nitsche2dDiag        = partial(assemble_matrix, adcore.assemble_matrix_DiffSpacediagnitsche)
            self.assemble_nitsche2dUnderDiag   = partial(assemble_matrix, adcore.assemble_matrix_DiffSpaceoffdiagnitsche)
        self._newdim    = (0,0)
        self.new_id     = {}
        self.old_id     = {}
    #---------------------------------------------
    # representative_dof -> list of equivalent dofs
    #---------------------------------------------
    def to_merge_dofs(self):
        '''
        Compute representative DOF -> set(equivalent DOFs) and mappings.

        Builds:
          - self.new_id: mapping old_dof -> new_dof (-1 for prescribed corners)
          - self.old_id: mapping new_dof -> set(old_dofs)
          - self._newdim: new global dimension (n,n)
        '''
        equiv      = {}     # rep -> set(dofs)
        dof_to_rep = {}     # dof -> rep
        for interface in self.mp.getInterfaces():
            # ... get interface and patch numbers
            patch_nb       = interface[0] 
            patch_nb_n     = interface[1]
            source_edge = interface[2][0]
            neighbor_edge = interface[2][1]
            source_axis = (source_edge - 1) // 2
            neighbor_axis = (neighbor_edge - 1) // 2
            source_tangent = 1 - source_axis
            neighbor_tangent = 1 - neighbor_axis
            from .interfaces import validate_trace_space
            reversed_trace = self.mp.isInterfaceReversed(interface)
            validate_trace_space(self._domain, source_edge, neighbor_edge, reversed_trace)
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
            source_starts = (d1, d3)
            source_stops = (d2, d4)
            neighbor_starts = (pd1, pd3)
            neighbor_stops = (pd2, pd4)
            for tangent_index in range(self._nbs[source_tangent]):
                source_coordinates = [0, 0]
                neighbor_coordinates = [0, 0]
                source_coordinates[source_axis] = ((source_edge-1) % 2)*(self._nbs[source_axis]-1)
                neighbor_coordinates[neighbor_axis] = ((neighbor_edge-1) % 2)*(self._nbs[neighbor_axis]-1)
                source_coordinates[source_tangent] = tangent_index
                neighbor_index = (self._nbs[neighbor_tangent]-1-tangent_index
                                  if reversed_trace else tangent_index)
                neighbor_coordinates[neighbor_tangent] = neighbor_index
                source_free = all(source_starts[a] <= source_coordinates[a] < source_stops[a]
                                  for a in (0, 1))
                neighbor_free = all(neighbor_starts[a] <= neighbor_coordinates[a] < neighbor_stops[a]
                                    for a in (0, 1))
                dof_q = pw + (source_coordinates[0]-d1)*(d4-d3) + source_coordinates[1]-d3
                dof_p = cpw + (neighbor_coordinates[0]-pd1)*(pd4-pd3) + neighbor_coordinates[1]-pd3
                if not source_free or not neighbor_free:
                    if ((source_free and dof_q not in self._corner_dofs)
                            or (neighbor_free and dof_p not in self._corner_dofs)):
                        raise ValueError("Interface endpoints have incompatible Dirichlet constraints")
                    continue
                if dof_p in self._corner_dofs or dof_q in self._corner_dofs:
                    continue

                rep_p = dof_to_rep.get(dof_p, dof_p)
                rep_q = dof_to_rep.get(dof_q, dof_q)
                rep = min(rep_p, rep_q)

                set_p = equiv.get(rep_p, {rep_p})
                set_q = equiv.get(rep_q, {rep_q})
                merged = set_p | set_q | {dof_p, dof_q}
                equiv[rep] = merged

                for dof in merged:
                    dof_to_rep[dof] = rep

                if rep_p != rep:
                    equiv.pop(rep_p, None)
                if rep_q != rep:
                    equiv.pop(rep_q, None)
        new_id  = {dof: -1 for dof in self._corner_dofs}
        old_id  = {}
        current = 0
        # Include every free DOF, even if its assembled row is currently zero.
        all_dofs = range(self._Nitshedim[0])
        for old_dof in all_dofs:
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
    def Alldomain( self ):
        '''
        Tensor space containes all B-spline spaces: 
            FE X MAPPING || FE X AD_MAPPING ( MAPPING is excluded in this special case as it has diff structure)
        '''
        return self._Alldomain
    # ...
    def tosparse( self ):
        return self.stencilNitsche
    #...
    def nitsche_merge(self):
        ''''
        nitsche_merge: merge DoFs of the same interface
        '''
        #... Compute relationship between old and new DoFs
        self.to_merge_dofs()
        rows, cols, data = [], [], []

        for i, j, v in zip(self.stencilNitsche.tocoo().row,
                        self.stencilNitsche.tocoo().col,
                        self.stencilNitsche.data):
            if self.new_id[i] < 0 or self.new_id[j] < 0:
                continue
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
    def nitsche_merge_rhs(self):
        '''
        nitsche_merge_rhs:  merge rhs DoFs of the same interface
        '''
        rhsMerged = np.zeros(self._newdim[0], dtype=self._type)

        for old_dof, value in enumerate(self.b_dir):
            I = self.new_id[old_dof]
            if I < 0:
                continue
            rhsMerged[I] += value   # SUM contributions

        return rhsMerged
    #... Extract solution
    def extract_sol(self, sol_Dof, patch_nb = 0, u_last = None):
        '''
        extract_sol:  Extract Solution DoFs from master/slave DoFs

        :param patch_nb: is patch number start from 1
        :param u_last: last solution to be updated; not Dirichlet boudary
        '''
        SolExtracted = np.zeros(self._Nitshedim[0], dtype=self._type)
        for new_dof, olds in self.old_id.items():
            for d in olds:
                SolExtracted[d] = sol_Dof[new_dof]   # Extract contributions
        if patch_nb == 0:
            return SolExtracted
        if u_last is not None:
            x_tmp = SolExtracted[self._block_index[patch_nb-1]:self._block_index[patch_nb]]
            u_sol = apply_dirichlet(self._domain, x_tmp, dirichlet = self.mp.getDirPatch(patch_nb), update = u_last)
            for i, j in self._corner_indices[patch_nb-1]:
                u_sol[i, j] = self.u_d[patch_nb-1][i, j]
            return u_sol
        else:
            x_tmp = SolExtracted[self._block_index[patch_nb-1]:self._block_index[patch_nb]]
            u_sol = apply_dirichlet(self._domain, x_tmp, dirichlet = self.mp.getDirPatch(patch_nb), update= self.u_d[patch_nb-1])
            for i, j in self._corner_indices[patch_nb-1]:
                u_sol[i, j] = self.u_d[patch_nb-1][i, j]
            return u_sol
    #====================================
    #... collect off diag NItsch matrices
    #====================================
    def collect_offdiag_stencil_matrix(self, stiffnessoffdiag, interface):
        '''
        Docstring for collect_offdiag_stencil_matrix 
        [i,j, i-p:i+p, j-p,j+p] to [i,j, n-i-p:n-i+p, j-p,j+p]
        
        :param stiffnessoffdiag: matrix off diagonal
        :param patch_nb patch number
        '''
        # ... get interface and patch numbers
        patch_nb       = interface[0]
        patch_nb_n     = interface[1]
        # ... get interface mappings
        interface_like = interface[2][0]
        interface_likeL = interface[2][1]
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
        source_normal = (interface_like - 1) // 2
        source_side = (interface_like - 1) % 2
        neighbor_normal = (interface_likeL - 1) // 2
        neighbor_side = (interface_likeL - 1) % 2
        for (index,value) in np.ndenumerate( stiffnessoffdiag._data[local] ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # x=i-s
            ll = index[nd:]  # l=p+k

            ii = [s+x for s,x in zip(ss,xx)]
            iiO = [s+x for s,x in zip(ss,xx)]
            jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self._pads)]
            tangent_index = ii[1-source_normal]
            source_opposite_boundary = (1-source_side) * (nc[source_normal] - 1)
            row_depth = abs(ii[source_normal] - source_opposite_boundary)
            ii[neighbor_normal] = neighbor_side * (nc[neighbor_normal] - 1) + (1-2*neighbor_side) * row_depth
            ii[1-neighbor_normal] = tangent_index
            column_depth = abs(jj[source_normal] - source_opposite_boundary)
            jj[source_normal] = source_side * (nc[source_normal] - 1) + (1-2*source_side) * column_depth
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
    def append_block(self, B, patch_nb, patch_nb_n = None):
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
        
    #==========================================================================
    #... Special Nitsche's methods for Laplace operator
    #==========================================================================
    def add_nitsche_off_diag(self):
        '''
        Docstring pour add_nitsche_off_diag for Laplace operator

        # :param Vh: FE space & multipatch space
        '''
        # if not isinstance(Vh, TensorSpace):
        #     raise TypeError("Vh must be a TensorSpace")
        if self.admp is None:
            self._prepare_uniform_interfaces()
            for interface, block in self._uniform_cross:
                p, q = interface[:2]
                block = block[self._uniform_keep[q-1]][:, self._uniform_keep[p-1]].tocoo()
                self.append_block(block, q, p)
                self.append_block(block.T, p, q)
            return
        for interface in self.mp.getInterfaces():
            # ... get interface and patch numbers
            patch_nb       = interface[0]
            patch_nb_n     = interface[1]
            # ... get interface mappings
            interface_like = interface[2][0]
            # assemble mappings for patches
            u_mae                                  = self.admp.stencil_mapping(patch_nb)
            # asemble new basis and spans for geometry mapping
            spansx, spansy, basisx, basisy         = self.admp.getBoundary_basis(self._domain, patch_nb)
            # assemble mappings for patches _n
            u_maen                                 = self.admp.stencil_mapping(patch_nb_n)
            # asemble new basis and spans for geometry mapping
            spansx_n, spansy_n, basisx_n, basisy_n = self.admp.getBoundary_basis(self._domain, patch_nb_n)
            # assemble mappings for patches
            u11_mph, u12_mph = self.mp.stencil_mapping(patch_nb)
            u21_mph, u22_mph = self.mp.stencil_mapping(patch_nb_n)
            #... assemble off diagonal matrix
            stiffnessoffdiag = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
            self.assemble_nitsche2dUnderDiag(self._Alldomain, fields=[u_mae[0], u_mae[1], u11_mph, u12_mph , u_maen[0], u_maen[1], u21_mph, u22_mph], knots=True,
                                            value=[spansx, spansy, basisx, basisy, spansx_n, spansy_n, basisx_n, basisy_n, self._mpdomain.knots[0],self._mpdomain.knots[1], self._mpdomain.omega[0],self._mpdomain.omega[1], interface_like, self.Kappa, self.normS],
                                            out = stiffnessoffdiag)
            #... correct coo matrix
            stiffnessoffdiag = self.collect_offdiag_stencil_matrix(stiffnessoffdiag, interface)
            assert not np.isnan(stiffnessoffdiag.data).any(), "OFF diag Sparse matrix contains NaNs"
            self.append_block(stiffnessoffdiag, patch_nb_n, patch_nb)
            self.append_block(stiffnessoffdiag.T, patch_nb, patch_nb_n)
        #...

    # #...
    def apply_nitsche(self, stiffness, patch_nb):
        '''
        Docstring pour apply_nitsche for diagonal matrices for Laplace operator
        
        :param self: Description
        :param stiffness: stifness matrix 
        :param patch_nb: patch number start from 1
        '''
        if not (1 <= patch_nb <= self._nb_patches):
            raise ValueError(f"patch_nb={patch_nb} out of range 1..{self._nb_patches}")
        if self.admp is None:
            self._prepare_uniform_interfaces()
            block = self._uniform_diagonal[patch_nb-1].tocoo()
            for row, col, value in zip(block.row, block.col, block.data):
                i, j = np.unravel_index(row, self._nbs)
                k, l = np.unravel_index(col, self._nbs)
                p, q = self._pads
                stiffness._data[p+i, q+j, p+k-i, q+l-j] += value
        else:
            # assemble mappings for patches
            u_mae                          = self.admp.stencil_mapping(patch_nb)
            # asemble new basis and spans for geometry mapping
            spansx, spansy, basisx, basisy = self.admp.getBoundary_basis(self._domain, patch_nb)
            # assemble mappings for patches
            u11_mph, u12_mph     = self.mp.stencil_mapping(patch_nb)
            #... get interfaces for a given patch
            interfaces_like      = self.mp.getInterfacePatch(patch_nb)
            # ... assemble diagonal matrix
            self.assemble_nitsche2dDiag(self._Alldomain, fields=[u_mae[0], u_mae[1], u11_mph, u12_mph], knots=True, value=[spansx, spansy, basisx, basisy, self._mpdomain.knots[0],self._mpdomain.knots[1], self._mpdomain.omega[0],self._mpdomain.omega[1], interfaces_like, self.Kappa, self.normS], out = stiffness)            
        # ...
        return

    def _prepare_uniform_interfaces(self):
        """Cache physical interface blocks before boundary elimination."""
        if hasattr(self, '_uniform_diagonal'):
            return
        from .interfaces import assemble_interface
        from scipy.sparse import csr_matrix
        n = int(np.prod(self._nbs))
        diagonal = [csr_matrix((n, n)) for _ in range(self._nb_patches)]
        cross = []
        keep = []
        lift = [np.zeros(n) for _ in range(self._nb_patches)]
        for patch_nb in range(self._nb_patches):
            starts, stops = self.elim_index[patch_nb, :, 0], self.elim_index[patch_nb, :, 1]
            indices = np.arange(n).reshape(self._nbs)
            keep.append(indices[starts[0]:stops[0], starts[1]:stops[1]].ravel())
        for interface in self.mp.getInterfaces():
            p, q, edges = interface
            pp, qq, qp = assemble_interface(
                self._domain, self.mp.get_patch(p), self.mp.get_patch(q), edges,
                self.mp.isInterfaceReversed(interface), self.Kappa, self.normS,
                n_core.assemble_interface_quadrature,
            )
            diagonal[p-1] += pp
            diagonal[q-1] += qq
            cross.append((interface, qp))
            dp, dq = self.u_d[p-1].tensor.ravel(), self.u_d[q-1].tensor.ravel()
            lift[p-1] -= pp @ dp + qp.T @ dq
            lift[q-1] -= qp @ dp + qq @ dq
        # Publish the cache only after all interfaces have validated successfully.
        self._uniform_diagonal = diagonal
        self._uniform_cross = cross
        self._uniform_keep = keep
        self._uniform_lift = lift
        self._uniform_rhs_lift_applied = [False]*self._nb_patches

    #--------------------------------------
    # ... assemble global Nitsche's matrix
    #--------------------------------------
    def assemble_nitsche(self):
        '''
        Docstring pour assemble_nitsche for Laplace operator
        assemble global Nitsche matrix in uniform mesh

        :param self: Description
        '''
        # ...
        for patch_nb in range(1,self.mp.nb_patches+1):
            # ... initial diag stiffness matrix
            stiffness  = StencilMatrix(self._domain.vector_space, self._domain.vector_space)
            # ... assemble 
            self.apply_nitsche(stiffness, patch_nb)
            # ... apply dirichlet
            stiffness  = apply_dirichlet(self._domain, stiffness, dirichlet = self.mp.getDirPatch(patch_nb))
            assert not np.isnan(stiffness.data).any(), "Sparse matrix contains NaNs"
            # ... assemble it into globel matrix
            self.append_block(stiffness, patch_nb)
            # ...
        self.add_nitsche_off_diag()
        #...
        return self.stencilNitsche
    #-------------------------------------------------
    # assemble Nitsche's Dirichlet contribution
    #-------------------------------------------------
    def assemble_nitsche_rhs(self, rhs, patch_nb, accumulate = False):
        '''
        Docstring for assemble_nitsche_rhs: assemble rhs vector for Laplace operator
        
        :param self: Description
        :param rhs: array vector
        :param patch_nb: patch number start from 1
        ! param accumulate: whether to accumulate the result into the existing rhs
        '''
        # assert isinstance(u_d, StencilVector)
        if self.admp is None:
            self._prepare_uniform_interfaces()
            if not accumulate or not self._uniform_rhs_lift_applied[patch_nb-1]:
                rhs = rhs[:] + self._uniform_lift[patch_nb-1][self._uniform_keep[patch_nb-1]]
            self._uniform_rhs_lift_applied[patch_nb-1] = True
            if accumulate:
                self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] += rhs[:]
            else:
                self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] = rhs[:]
        else:
            if accumulate:
                self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] += rhs[:]
            else:
                self.b_dir[self._block_index[patch_nb-1]:self._block_index[patch_nb]] = rhs[:]
        # ...
        return

#==============================================================================
#.... application of Dirichlet boundary conditions
#==============================================================================
def apply_dirichlet(V, x, dirichlet = True, update = None, periodic = [False, False]):
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
    periodic: list or tuple, optional only 2d case
        Specifies periodic boundary conditions in each direction (default: [False, False]).
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
        
        u   = StencilVector(V.vector_space)
        u.from_array(V, update.tensor)
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
            # if periodic[0] == True: # in x direction
            #     u[-1,:] = u[0,:]
            # if periodic[1] == True: # in y direction
            #     u[:,0] = u[:,-1]
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


def apply_zeros(V, x, row_dirichlet=True, col_dirichlet=None):
    """Remove selected rows and columns from a stencil matrix."""
    if not isinstance(x, StencilMatrix):
        raise TypeError('Expecting a StencilMatrix')

    if col_dirichlet is None:
        col_dirichlet = row_dirichlet

    if V.dim == 1:
        npts = (V.nbasis,)
    elif V.dim == 2:
        npts = tuple(V.nbasis)
    elif V.dim == 3:
        npts = tuple(V.nbasis)
    else:
        raise NotImplementedError('Only 1d, 2d and 3d are available')

    def free_indices(dirichlet):
        if V.dim == 1:
            if isinstance(dirichlet, bool):
                dirichlet = [dirichlet, dirichlet]
            starts = [1 if dirichlet[0] else 0]
            stops = [npts[0] - 1 if dirichlet[1] else npts[0]]
        else:
            if isinstance(dirichlet, bool):
                dirichlet = [[dirichlet, dirichlet] for _ in range(V.dim)]
            starts = [1 if boundary[0] else 0 for boundary in dirichlet]
            stops = [n - 1 if boundary[1] else n for n, boundary in zip(npts, dirichlet)]

        grid = np.arange(np.prod(npts)).reshape(npts)
        return grid[tuple(slice(start, stop) for start, stop in zip(starts, stops))].ravel()

    matrix = x.tosparse().tocsr()
    rows = free_indices(row_dirichlet)
    cols = free_indices(col_dirichlet)
    return matrix[rows][:, cols].tocoo()


#==============================================================================
def apply_periodic(V, x, periodic=None, update=None):
    """Fold repeated spline coefficients while retaining stencil storage.

    Matrices return a reduced StencilMatrix representing P.T @ A @ P;
    vectors return a reduced StencilVector representing P.T @ b. P extends
    independent coefficients by copying the first degree entries at each
    periodic axis's end. No dense or sparse matrix conversion occurs here.

    With update=True, extend reduced coefficients to the original space.
    A StencilVector input returns a StencilVector; array inputs return arrays.
    If periodic is omitted, all parameter directions are periodic.
    """
    from .linalg import StencilVectorSpace

    ndim = V.dim
    if ndim not in (1, 2, 3):
        raise NotImplementedError('Only 1d, 2d and 3d are available')
    shape = (V.nbasis,) if ndim == 1 else tuple(V.nbasis)
    pads = (V.degree,) if ndim == 1 else tuple(V.degree)
    if periodic is None:
        periodic = (True,)*ndim
    elif isinstance(periodic, (bool, np.bool_)):
        periodic = (bool(periodic),)*ndim
    else:
        periodic = tuple(bool(flag) for flag in periodic)
    if len(periodic) != ndim:
        raise ValueError('periodic must contain one flag per parameter direction')
    reduced_shape = tuple(n-p if flag else n for n, p, flag in zip(shape, pads, periodic))
    if any(n <= 0 for n in reduced_shape):
        raise ValueError('Periodic space must have independent coefficients')
    reduced_space = StencilVectorSpace(reduced_shape, pads, periodic, dtype=V.vector_space.dtype)

    if update is not None and update is not False:
        if isinstance(x, StencilMatrix):
            raise TypeError('Periodic extension requires coefficient arrays or a StencilVector')
        values = x.tensor if isinstance(x, StencilVector) else np.asarray(x)
        if values.size != int(np.prod(reduced_shape)):
            raise ValueError('Coefficient size does not match the reduced periodic space')
        values = values.reshape(reduced_shape)
        indices = [np.arange(n) % m if flag else np.arange(n)
                   for n, m, flag in zip(shape, reduced_shape, periodic)]
        extended = values[np.ix_(*indices)]
        if isinstance(x, StencilVector):
            result = StencilVector(V.vector_space)
            result.from_array(V, extended)
            return result
        return extended.copy()

    owned = tuple(slice(p, p+n) for p, n in zip(pads, shape))
    if isinstance(x, StencilMatrix):
        if x.domain.npts != shape or x.codomain.npts != shape:
            raise ValueError('Matrix dimensions do not match the original spline space')
        result = StencilMatrix(reduced_space, reduced_space)
        local = x._data[owned]
        indices = np.nonzero(local)
        if not len(indices[0]):
            return result
        values = local[indices]
        rows, offsets = indices[:ndim], indices[ndim:]
        targets = []
        diagonals = []
        valid = np.ones(values.shape, dtype=bool)
        for axis, (n, m, p, flag) in enumerate(zip(shape, reduced_shape, pads, periodic)):
            column = rows[axis]+offsets[axis]-p
            # Match the input stencil's column wrapping convention.
            column = column % n
            row = rows[axis] % m if flag else rows[axis]
            column = column % m if flag else column
            delta = column-row
            if flag:
                delta = (delta+m//2) % m-m//2
            valid &= np.abs(delta) <= p
            targets.append(row+p)
            diagonals.append(delta+p)
        if not valid.all():
            raise ValueError('Periodic folding produced a coupling outside the stencil bandwidth')
        np.add.at(result._data, tuple(targets+diagonals), values)
        return result
    if isinstance(x, StencilVector):
        if x.space.npts != shape:
            raise ValueError('Vector dimensions do not match the original spline space')
        result = StencilVector(reduced_space)
        source_indices = np.indices(shape)
        target_indices = tuple(source_indices[a] % reduced_shape[a] if periodic[a]
                               else source_indices[a] for a in range(ndim))
        folded = np.zeros(reduced_shape, dtype=x._data.dtype)
        np.add.at(folded, target_indices, x._data[owned])
        target_owned = tuple(slice(p, p+n) for p, n in zip(pads, reduced_shape))
        result._data[target_owned] = folded
        return result
    raise TypeError('Periodic reduction requires a StencilMatrix or StencilVector')
