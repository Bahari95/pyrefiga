"""
Module-level docstring
---------------------
Provides a fast diagonalization based solver for Poisson / Laplace-type
problems in 2D and 3D using separable tensor-product operators.
Main class
----------
Poisson
    Encapsulates the data and routines required to solve linear systems
    arising from separable discretizations of the Laplace/Poisson operator
    with optional shift (tau). The solver performs a generalized eigenvalue
    decomposition of each 1D operator pair (K, M) and uses the resulting
    eigenbasis to reduce multi-dimensional problems to independent
    (small) Sylvester-like systems solved by optimized routines.
Constructor
-----------
Poisson(mats_1, mats_2, mats_3=None, tau=0.0)
    Build a Poisson solver instance for a tensor-product domain.
    Parameters
    - mats_1, mats_2, mats_3: sequence of two 1D discrete operators
      Each mats_i must be a sequence/tuple/list of two square sparse
      matrices: (M_i, K_i) where M_i is the mass matrix and K_i is the
      stiffness (or similar) matrix along direction i. mats_3 is optional;
      if omitted the instance represents a 2D problem, otherwise 3D.
      The constructor asserts len(mats_i) == 2 for provided mats.
    - tau: float (default 0.0)
      Scalar shift added to the operator when solving the unit Sylvester
      systems (affects the right-hand side handled inside the core solvers).
    Behavior
    - For each provided (M, K) pair the constructor converts sparse inputs
      to dense arrays and computes the generalized eigen-decomposition
      K U = M U diag(d) using scipy.linalg.eigh. Eigenvectors are cast
      to ensure a C-contiguous ordering compatible with downstream
      (pyccel / Fortran interaction) code.
    - Stores per-direction eigenvalues (`ds`) and eigenvectors (`Us`),
      the problem dimensionality (`rdim`), tau, and the per-direction
      numbers of degrees of freedom (`nDoFs`).
Public properties
-----------------
- rdim: int
  Problem spatial dimension (2 or 3).
- nDoFs: tuple[int,...]
  Number of degrees of freedom in each direction (length rdim).
- mats_1, mats_2, mats_3:
  Original operator tuples provided at construction (mats_3 may be None).
- ds: list[numpy.ndarray]
  Per-direction eigenvalue arrays from the generalized eigenproblems.
- Us: list[numpy.ndarray]
  Per-direction eigenvector matrices (columns are eigenvectors).
- tau: float
  Shift parameter passed to Sylvester solvers.
Solving and projection API
--------------------------
- solve(b, ij_dir=None) -> numpy.ndarray
  Solve the full Poisson problem for a flattened right-hand side vector b.
  For 2D problems this dispatches to an internal 2D solver, for 3D to
  the 3D solver. The caller must provide b as a 1D array of length equal
  to the product of the per-direction sizes; the returned solution is
  also a 1D array with the same ordering.
- project(b) -> numpy.ndarray
  Perform an orthogonal change-of-basis using the stored eigenvectors
  and then project back (useful for testing/verification of transforms).
  Acts as an identity up to numerical round-off when applied successively
  with the same eigenvectors.
Internal / auxiliary methods (descriptions)
------------------------------------------
- _solve_2d(self, b)
  Solve a 2D separable system by:
    * reshaping b into (n1, n2),
    * applying the left/right eigenvector transforms U0^T, U1^T,
    * calling core.solve_unit_sylvester_system_2d to solve the diagonalized
      small systems (in-place),
    * transforming back with U0, U1^T and returning the flattened result.
- solve_23d(self, b, i_dir, j_dir)
  Solve a 2D problem embedded in the 3D context for the two directions
  indicated by indices i_dir and j_dir. Useful to avoid building full
  Kronecker products when solving in a fixed pair of dimensions.
- _solve_3d(self, b)
  Solve a full 3D separable problem by applying a sequence of basis
  transforms (implemented with efficient reshapes and einsum calls),
  calling core.solve_unit_sylvester_system_3d on the transformed data
  and transforming back. Input and output are 1D flattened arrays using
  the same ordering convention as nDoFs.
- _project_2d / _project_3d
  Perform forward/backward transforms across directions only (no linear
  solve). Useful to project vectors into the eigenbasis and reassemble.
Notes and implementation details
-------------------------------
- The module relies on an external compiled core (fast_diag_core) that
  provides efficient routines:
    * solve_unit_sylvester_system_2d(...)
    * solve_unit_sylvester_system_3d(...)
  These routines perform the actual diagonal-block solves and are expected
  to operate in-place on passed arrays.
- Input sparse matrices are converted to dense arrays via toarray()
  before eigen-decomposition; for large 1D operators this can be
  memory-expensive. The design favors performance of the separable
  diagonalization approach at the cost of storing 1D dense eigenvectors.
- Eigenvectors are explicitly made C-contiguous via a CSR -> dense round
  trip to avoid Fortran/C ordering issues when interfacing with
  pyccel/Fortran-compiled code.
- The class assumes consistent ordering of degrees of freedom and that
  the provided operators are symmetric definite so that scipy.linalg.eigh
  is applicable. There are assertions checking the expected tuple lengths.
- The solve methods expect and return flattened 1D numpy arrays. The
  mapping between flattened order and multi-dimensional layout follows
  the reshaping conventions used internally (row-major / C-order).
Example (conceptual)
--------------------
Construct a 2D solver with operator pairs (M1, K1) and (M2, K2), then
solve for a flattened right-hand side b:
    P = Poisson((M1, K1), (M2, K2), tau=0.0)
    x = P.solve(b)
Limitations and warnings
------------------------
- This implementation is tailored to separable tensor-product discretizations.
  It is not suitable for general unstructured sparse systems.
- Converting large sparse 1D operators to dense arrays may not be feasible
  for very large meshes.
- The quality/stability of the solver depends on the eigen-decomposition.
  Ensure M is positive definite (or at least definite in the subspace of
  interest) when calling eigh(K, b=M).
fast_diag.py
----------------
@author : M. BAHARI
"""

import numpy         as np
from scipy.linalg    import eigh
from scipy.sparse    import csr_matrix, coo_matrix

from .               import fast_diag_core as core
from scipy.sparse    import kron
from scipy.sparse    import csr_matrix
from pyccel.epyccel  import epyccel

# =========================================================================
class Poisson(object):
    def __init__(self, mats_1, mats_2, mats_3=None, tau=0.):
        # ...
        assert(len(mats_1) == 2)
        assert(len(mats_2) == 2)


        rdim  = None
        if mats_3 is None:
            rdim = 2
        else:
            assert(len(mats_3) == 2)
            rdim = 3
        # ...

        # ...
        if rdim == 2:
            Ms = [mats_1[0], mats_2[0]]
            Ks = [mats_1[1], mats_2[1]]
        else:
            Ms = [mats_1[0], mats_2[0], mats_3[0]]
            Ks = [mats_1[1], mats_2[1], mats_3[1]]
        # ...

        # ... generalized eigenvalue decomposition
        nDoFs = []
        ds    = []
        Us    = []
        for M, K in zip(Ms, Ks):
            M = M.toarray()
            K = K.toarray()

            d, U = eigh(K, b=M)

            # trick to avoid F/C ordering with pyccel
            U   = csr_matrix(U).toarray()


            ds.append(d)
            Us.append(U)
            nDoFs.append(len(d))

        # ...
        self._mats_1 = mats_1
        self._mats_2 = mats_2
        self._mats_3 = mats_3

        self._ds    = ds
        self._Us    = Us
        self._rdim  = rdim
        self._tau   = tau
        self._nDoFs = nDoFs
        # ...

    @property
    def rdim(self):
        return self._rdim
    
    @property
    def nDoFs(self):
        return self._nDoFs

    @property
    def mats_1(self):
        return self._mats_1

    @property
    def mats_2(self):
        return self._mats_2

    @property
    def mats_3(self):
        return self._mats_3

    @property
    def ds(self):
        return self._ds

    @property
    def Us(self):
        return self._Us

    @property
    def tau(self):
        return self._tau

    def _solve_2d(self, b):

        n1, n2  = self.nDoFs
        s_tilde = b.reshape((n1,n2))
        s_tilde = self.Us[0].T @ s_tilde @ self.Us[1]
        core.solve_unit_sylvester_system_2d(*self.ds, s_tilde, float(self.tau), s_tilde)
        s_tilde = self.Us[0] @ s_tilde @ self.Us[1].T
        s_tilde = s_tilde.reshape(n1*n2)
        return s_tilde

    def solve_23d(self, b, i_dir, j_dir):

        # # ... Avoidding kron product : i_dir j_dir, if in 3D one wants to solve the Poisson equation in a fixed direction
        n1, n2  = self.nDoFs[i_dir], self.nDoFs[j_dir]
        s_tilde = b.reshape((n1,n2))
        s_tilde = self.Us[i_dir].T @ s_tilde @ self.Us[j_dir]
        core.solve_unit_sylvester_system_2d(self.ds[i_dir], self.ds[j_dir], s_tilde, float(self.tau), s_tilde)
        s_tilde = self.Us[i_dir] @ s_tilde @ self.Us[j_dir].T
        s_tilde = s_tilde.reshape(n1*n2)
        return s_tilde
    
    def _solve_3d(self, b):
        # ... Avoidding kron product
        n1, n2, n3     = self.nDoFs
        s_tilde        = b.reshape(n1,n2*n3)
        s_tilde        = s_tilde.T @ self.Us[0]
        # matrix becomes (n2*n3, n1)
        r_tilde = np.zeros((n1,n2,n3))
        r_tilde[:,:,:] = s_tilde.T.reshape(n1, n2, n3)[:,:,:]
        r_tilde[:,:,:] = np.einsum('ij,njk->nik', self.Us[1].T, r_tilde)[:,:,:]      # transform along axis=1
        r_tilde[:,:,:] = np.einsum('nij,jk->nik', r_tilde, self.Us[2])[:,:,:]        # transform along axis=2
        core.solve_unit_sylvester_system_3d(self.ds[0], self.ds[1],self.ds[2], r_tilde, float(self.tau), r_tilde)

        r_tilde[:,:,:] = np.einsum('ij,njk->nik', self.Us[1], r_tilde)[:,:,:]        # transform along axis=1
        r_tilde[:,:,:] = np.einsum('nij,jk->nik', r_tilde, self.Us[2].T)[:,:,:]      # transform along axis=2
        # reshape and transpose back to (n2*n3, n1)
        s_tilde        = r_tilde.reshape(n1, n2*n3)
        #...
        s_tilde = self.Us[0] @ s_tilde
        s_tilde = s_tilde.reshape(n1*n2*n3)
        # ...
        return s_tilde

    def _project_2d(self, b):
        # # ... Avoidding kron product
        n1, n2  = self.nDoFs
        s_tilde = b.reshape((n1,n2))
        s_tilde = self.Us[0].T @ s_tilde @ self.Us[1]
        s_tilde = self.Us[0] @ s_tilde @ self.Us[1].T
        s_tilde = s_tilde.reshape(n1*n2)
        return s_tilde

    def _project_3d(self, b):
        # ... Avoidding kron product
        n1, n2, n3 = self.nDoFs
        s_tilde    = b.reshape(n1,n2*n3)
        s_tilde    = s_tilde.T @ self.Us[0]
        # matrix becomes (n2*n3, n1)
        r_tilde = np.zeros((n1,n2,n3))
        r_tilde[:,:,:] = s_tilde.T.reshape(n1, n2, n3)[:,:,:]
        r_tilde[:,:,:] = np.einsum('ij,njk->nik', self.Us[1].T, r_tilde)[:,:,:]      # transform along axis=1
        r_tilde[:,:,:] = np.einsum('nij,jk->nik', r_tilde, self.Us[2])[:,:,:]        # transform along axis=2
        # ...
        r_tilde[:,:,:] = np.einsum('ij,njk->nik', self.Us[1], r_tilde)[:,:,:]        # transform along axis=1
        r_tilde[:,:,:] = np.einsum('nij,jk->nik', r_tilde, self.Us[2].T)[:,:,:]      # transform along axis=2
        # reshape and transpose back to (n2*n3, n1)
        s_tilde        = r_tilde.reshape(n1, n2*n3)
        # ...
        s_tilde = self.Us[0] @ s_tilde
        s_tilde = s_tilde.reshape(n1*n2*n3)
        # ...
        return s_tilde

    
    def solve(self, b, ij_dir = None):
        if self.rdim == 2:
            return self._solve_2d(b)
        else:
            return self._solve_3d(b)

    def project(self, b):
        if self.rdim == 2:
            return self._project_2d(b)
        else:
            return self._project_3d(b)
            
            
            
            
            
            
            
            
            
