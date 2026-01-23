"""
pyrefiga.spaces
Utilities for constructing and working with one-dimensional B-spline and tensor-product spline spaces used
for assembly and numerical integration in isogeometric and spline-based discretizations.
This module provides:
- SplineSpace: Represents a 1D B-spline/NURBS space, prepares knot vectors, local quadrature points,
    basis values and derivatives on element quadrature grids and encapsulates vector-space metadata.
- TensorSpace: A small container to combine multiple SplineSpace instances into a tensor-product space,
    exposing combined vector-space metadata used by the library linear-algebra utilities.
Exports:
- SplineSpace, TensorSpace
Notes:
- The module relies on helper routines in bsplines, quadratures and nurbs_core for knot construction,
    quadrature grid construction and basis evaluation. The SplineSpace may be constructed with or without
    NURBS weights (omega) and supports periodic and non-periodic spaces.
SplineSpace(degree, nelements=None, grid=None, nderiv=1, periodic=False, normalization=False,
                        omega=None, mesh=None, quad_degree=None)
Container describing a one-dimensional B-spline or NURBS function space together with
precomputed data required for numerical integration and assembly.
Parameters
- degree (int): Polynomial degree of the B-spline basis.
- nelements (int, optional): Number of (parametric) elements on the integral grid. Either nelements or grid must be provided. nelements = len(mesh)
- grid (array-like, optional): Increasing sequence of grid (knot-span) breakpoints. If provided, nelements is inferred.
- nderiv (int, optional): Highest derivative order to evaluate for basis functions (0 => values only). Default: 1.
- periodic (bool, optional): If True, constructs a periodic spline space by appropriate knot handling. Default: False.
- normalization (bool, optional): If True, apply normalization to basis/derivative values where appropriate.
- omega (array-like, optional): If provided, treated as NURBS weights and the basis evaluations become rational (NURBS).
- mesh (array-like, optional): Integration/assembly mesh used to build element-wise quadrature points. If None, the 'grid' is used.
- quad_degree (int, optional): Degree (number of Gauss points) used for element quadrature. Defaults to 'degree' when None.
Attributes (selected)
- knots (ndarray): Full knot vector used to define the B-spline basis.
- spans (ndarray): Array of span indices for each element/or quadrature point (shape depends on shared mesh flag).
- grid (ndarray): The breakpoints (coarse grid) used to define element boundaries (not necessarily the integration mesh).
- mesh (ndarray): The integration mesh used to create quadrature points (may differ from grid when adaptive integration is needed in this case nelements = len(mesh) !=len(grid)).
- degree (int): Polynomial degree.
- nelements (int): Number of integration elements associated with mesh.
- nbasis (int): Number of basis functions (size of the discrete function space).
- points (ndarray): Quadrature evaluation points per element (local coordinates).
- weights (ndarray): Quadrature weights per element.
- basis (ndarray): Precomputed nonzero basis values and derivatives on the quadrature points.
- omega (ndarray or None): NURBS weights if provided (None for standard B-splines).
- vector_space (StencilVectorSpace): Lightweight descriptor describing stencil connectivity for DOF indexing.
Behavior and implementation notes
- When omega is not provided, basis evaluations are computed by Python helpers from the bsplines submodule.
- When omega is provided (NURBS), evaluation is delegated to optimized nurbs_core routines which populate
    preallocated arrays for basis and derivative values.
- The object stores both geometric grid information and per-element quadrature data to simplify assembly loops.
TensorSpace(*spaces)
Container for forming a tensor-product space from one or more SplineSpace instances.
Parameters
- *spaces: One or more SplineSpace objects. All arguments must be instances of SplineSpace.
Attributes (selected)
- spaces (tuple): Ordered tuple of the component SplineSpace instances.
- vector_space (StencilVectorSpace): Combined vector-space descriptor for the tensor-product DOF layout.
- knots, spans, grid, mesh, degree, nelements, nbasis, points, weights, basis, omega:
        List-valued properties exposing the corresponding attribute from each component SplineSpace, in the same order.
Notes
- TensorSpace does not assemble combined quadrature or evaluate tensor-product bases itself; it is a convenience
    descriptor that packages multiple 1D spaces and creates the appropriate combined StencilVectorSpace used by
    assembly/solver utilities.
- The dimension property reports the sum of the component dimensions (1 per SplineSpace here), consistent
    with how this project treats geometric dimension versus tensor-product directions.
Small smoke-test helper that constructs a single 1D SplineSpace and creates placeholder linear-algebra objects
(StencilMatrix/StencilVector) to validate basic integration with the linalg utilities.
This function is intended for quick local checks and examples only; it does not perform assertions or return values.
Small smoke-test helper that constructs two SplineSpace instances, forms a TensorSpace from them and
creates placeholder linear-algebra objects (StencilMatrix/StencilVector) to validate interaction between
tensor-space descriptors and the linalg utilities.
This function is intended for quick local checks and examples only; it does not perform assertions or return values.
"""

from .bsplines    import elements_spans  # computes the span for each element
from .bsplines    import make_knots      # create a knot sequence from a grid
from .bsplines    import quadrature_grid # create a quadrature rule over the whole 1d grid
from .bsplines    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from .quadratures import gauss_legendre
from .linalg      import StencilVectorSpace
from .            import nurbs_core as  core

from numpy import linspace, zeros, unique, ones

__all__ = ['SplineSpace', 'TensorSpace']

# =================================================================================================
class SplineSpace(object):
    def __init__(self, degree, nelements=None, grid=None, nderiv=1, multiplicity = 1,
                 periodic=False, normalization=False, omega = None, mesh = None, quad_degree = None):
        
        '''
        class SplineSpace:
        ------------------------------------------
        Description:
        This class creates a one-dimensional B-spline space defined over a specified grid or number of
        elements. It supports periodic and non-periodic B-splines, allows for the computation of B-spline
        basis functions and their derivatives at quadrature points, and can handle NURBS weights for rational B-splines.
        ------------------------------------------
        Parameters:
        - degree (int): Degree of the B-spline basis functions.
        - nelements (int, optional): Number of elements in the grid. If not provided, 'grid' must be specified. Important: nelements = len(mesh)>= len(grid)
        - grid (array-like, optional): Grid points defining the B-spline space. If not provided, 'nelements' must be specified.
        - nderiv (int, optional): Number of derivatives to compute for the B-spline basis functions. Default is 1.
        - periodic (bool, optional): If True, creates a periodic B-spline space. Default is False.
        - normalization (bool, optional): If True, normalizes the B-spline basis functions. Default is False.
        - omega (array-like, optional): Weights for NURBS basis functions. If None, standard B-splines are used.
        - mesh (array-like, optional): Custom mesh for quadrature points. If None, the grid is used then span has same ne as shape otherewise (ne,quad).
        - quad_degree (int, optional): Degree of the quadrature rule. If None, defaults to 'degree'.
        ------------------------------------------
        Returns:
        - An instance of the SplineSpace class with properties and methods to access B-spline characteristics.
        ------------------------------------------  
        '''
        if (nelements is None) and (grid is None):
            raise ValueError('Either nelements or grid must be provided')

        if grid is None:
            grid  = linspace(0., 1., nelements+1)

        if mesh is not None:
            assert len(mesh) >= len(unique(grid)), "mesh must have at least as many points as grid"

        knots     = make_knots(grid, degree, periodic=periodic, multiplicity = multiplicity)

        nbasis    = len(knots) - degree - 1
        
        # .. for assembling integrals
        # create the gauss-legendre rule, on [-1, 1]
        if quad_degree is None :
            quad_degree = degree
        u, w = gauss_legendre( quad_degree)

        if omega is None:
            if mesh is None :

                mesh            = unique(grid)
                nelements       = len(mesh)-1
                # for each element on the grid, we create a local quadrature grid
                points, weights = quadrature_grid( mesh, u, w )

                # for each element and a quadrature points,
                # we compute the non-vanishing B-Splines
                basis, spans = basis_ders_on_quad_grid( knots, degree, points, nderiv,
                                            normalization=normalization, mesh = True)
                assert all(spans[:, 0] == spans[:, -1]), "Span indices mismatch"
                spans        = spans[:,0]
            else :
                nelements    = len(mesh)-1 # corresponds to integration discretization, which may differ from the knot grid
                # for each element on the grid, we create a local quadrature grid
                points, weights = quadrature_grid( mesh, u, w )
                basis, spans = basis_ders_on_quad_grid( knots, degree, points, nderiv,
                                            normalization=normalization, mesh = True)
            omega = ones(nbasis)
            nurbs = False
        else:
            if mesh is None :
                mesh            = unique(grid)
                nelements       = len(mesh)-1
                # for each element on the grid, we create a local quadrature grid
                points, weights = quadrature_grid( mesh, u, w )
                # for each element and a quadrature points,
                # we compute the non-vanishing B-Splines
                basis = zeros((nelements, degree+1, nderiv+1, points.shape[1]))
                spans = zeros(nelements, dtype=int )
                core.nurbs_ders_on_quad_grid(nelements, degree, spans, basis, weights, points, knots, omega, nderiv)
            else :
                nelements    = len(mesh)-1 # corresponds to integration discretization, which may differ from the knot grid
                # for each element on the grid, we create a local quadrature grid
                points, weights = quadrature_grid( mesh, u, w )
                # for each element and a quadrature points,
                # we compute the non-vanishing B-Splines
                basis = zeros((nelements, degree+1, nderiv+1, points.shape[1]))
                spans = zeros((nelements,points.shape[1]), dtype=int )
                core.nurbs_ders_on_shared_quad_grid(nelements, degree, spans, basis, weights, points, knots, omega, nderiv)
            nurbs = True
        self._periodic  = periodic
        self._knots     = knots
        self._spans     = spans
        self._grid      = grid
        self._mesh      = mesh
        self._degree    = degree
        self._nelements = nelements
        self._nbasis    = nbasis
        self._points    = points
        self._weights   = weights
        self._basis     = basis
        self._omega     = omega
        self._nderiv    = nderiv
        self._nurbs     = nurbs

        self._vector_space = StencilVectorSpace([nbasis], [degree], [periodic])

    @property
    def vector_space(self):
        return self._vector_space

    @property
    def periodic(self):
        return self._periodic

    @property
    def knots(self):
        return self._knots

    @property
    def spans(self):
        return self._spans

    @property
    def grid(self):
        return self._grid

    @property
    def mesh(self):
        return self._mesh

    @property
    def degree(self):
        return self._degree

    @property
    def nelements(self):
        return self._nelements

    @property
    def nbasis(self):
        return self._nbasis

    @property
    def points(self):
        return self._points

    @property
    def weights(self):
        return self._weights

    @property
    def basis(self):
        return self._basis
    @property
    def omega(self):
        return self._omega
    
    @property
    def nderiv(self):
        return self._nderiv
    @property
    def dim(self):
        return 1
    @property
    def nurbs(self):
        return self._nurbs
# =================================================================================================
class TensorSpace(object):
    def __init__( self, *args ):
        """."""
        assert all( isinstance( s, SplineSpace ) for s in args )
        self._spaces = tuple(args)

        self._nurbs = bool(sum(not V.nurbs for V in self.spaces))
        # ...
        nbasis   = [V.nbasis   for V in self.spaces]
        degree   = [V.degree   for V in self.spaces]
        periodic = [V.periodic for V in self.spaces]

        self._vector_space = StencilVectorSpace(nbasis, degree, periodic)

    @property
    def vector_space(self):
        return self._vector_space

    @property
    def spaces(self):
        return self._spaces

    @property
    def knots(self):
        return [V.knots for V in self.spaces]

    @property
    def spans(self):
        return [V.spans for V in self.spaces]

    @property
    def grid(self):
        return [V.grid for V in self.spaces]

    @property
    def mesh(self):
        return [V.mesh for V in self.spaces]

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def nelements(self):
        return [V.nelements for V in self.spaces]

    @property
    def nbasis(self):
        return [V.nbasis for V in self.spaces]

    @property
    def points(self):
        return [V.points for V in self.spaces]

    @property
    def weights(self):
        return [V.weights for V in self.spaces]

    @property
    def basis(self):
        return [V.basis for V in self.spaces]

    @property
    def omega(self):
        return [V.omega for V in self.spaces]

    @property
    def nderiv(self):
        return [V.nderiv for V in self.spaces]
    @property
    def dim(self):
        return sum([V.dim for V in self.spaces])
    @property
    def nurbs(self):
        return self._nurbs
# =================================================================================================
def test_1d():
    V = SplineSpace(degree=3, nelements=16)

    M = StencilMatrix(V.vector_space, V.vector_space)
    u = StencilVector(V.vector_space)

def test_2d():
    V1 = SplineSpace(degree=3, nelements=16)
    V2 = SplineSpace(degree=2, nelements=8)
    V = TensorSpace(V1, V2)

    M = StencilMatrix(V.vector_space, V.vector_space)
    u = StencilVector(V.vector_space)

##################################
if __name__ == '__main__':
    from linalg import StencilVector, StencilMatrix

    test_1d()
    test_2d()
