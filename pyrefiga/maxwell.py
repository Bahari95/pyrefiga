"""Compatible periodic 2D curl-curl waves on a Cartesian rectangle.

The component spaces are S(p-1,p) and S(p,p-1), not two copies of
S(p,p). Materials epsilon and mu are positive constants. Geometry maps,
NURBS weights and multipatch coupling are not included here.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigvalsh
from scipy.sparse import block_diag, coo_matrix, eye, hstack, kron, vstack
from scipy.sparse.linalg import splu

from .spaces import SplineSpace, TensorSpace
from .linalg import StencilVector
from .api import apply_periodic
from .ad_mesh_tools import assemble_mass1D


def _periodic_basis(space, points):
    """Evaluation matrix after identifying repeated periodic coefficients."""
    full = BSpline.design_matrix(np.asarray(points), space.knots, space.degree).tocoo()
    n = space.nbasis-space.degree
    return coo_matrix((full.data, (full.row, full.col % n)),
                      shape=(len(points), n)).tocsr()


def _derivative(high, low):
    # Trimming the first and last knot gives the derivative spline space.
    if not np.array_equal(high.knots[1:-1], low.knots):
        raise ValueError('Derivative space must use the trimmed knot vector')
    n, p = high.nbasis-high.degree, high.degree
    j = np.arange(n)
    scale = p/(high.knots[j+p+1]-high.knots[j+1])
    return coo_matrix((np.r_[-scale, scale],
                       (np.r_[j, j], np.r_[j, (j+1) % n])), shape=(n, n)).tocsr()


class PeriodicCurlCurl2D:
    """Assemble M_epsilon u_tt + C.T M_(1/mu) C u = F.

    Periodic identifications use the same Cartesian tangent on paired edges:
    u_y(0,y)=u_y(1,y), u_x(x,0)=u_x(x,1) on the unit square.
    Both components are represented on the periodic rectangle (a torus).
    Degree may be an integer or (px,py), each >= 2; nelements may likewise
    be an integer or a pair. Supply grids=(x_breaks,y_breaks) for a nonuniform
    mesh or a rectangle other than [0,1]^2. Coefficients use C-order flattening.
    """

    def __init__(self, degree=2, nelements=16, epsilon=1.0, mu=1.0, grids=None):
        degrees = (degree, degree) if np.isscalar(degree) else tuple(degree)
        if len(degrees) != 2 or any(not isinstance(p, (int, np.integer)) or p < 2 for p in degrees):
            raise ValueError('degree must contain two integers >= 2')
        self.epsilon, self.mu = float(epsilon), float(mu)
        if not all(np.isfinite(c) and c > 0 for c in (self.epsilon, self.mu)):
            raise ValueError('epsilon and mu must be finite positive constants')
        if grids is None:
            counts = (nelements, nelements) if np.isscalar(nelements) else tuple(nelements)
            if len(counts) != 2 or any(not isinstance(n, (int, np.integer)) or n <= p
                                       for n, p in zip(counts, degrees)):
                raise ValueError('nelements must exceed degree in each direction')
            grids = tuple(np.linspace(0, 1, n+1) for n in counts)
        if len(grids) != 2:
            raise ValueError('Provide one grid per direction')
        grids = tuple(np.asarray(g, dtype=float) for g in grids)
        if any(g.ndim != 1 or len(g) <= p+1 or not np.isfinite(g).all()
               or not np.all(np.diff(g) > 0) for g, p in zip(grids, degrees)):
            raise ValueError('Grids must be finite, increasing, with more elements than degree')
        self.grids = grids
        high = tuple(SplineSpace(int(p), grid=g, periodic=True, quad_degree=int(p)+1)
                     for p, g in zip(degrees, grids))
        low = tuple(SplineSpace(int(p)-1, grid=g, periodic=True, quad_degree=int(p)+1)
                    for p, g in zip(degrees, grids))
        self.scalar_space = TensorSpace(*high)
        self.component_spaces = (TensorSpace(low[0], high[1]), TensorSpace(high[0], low[1]))
        self.curl_space = TensorSpace(*low)
        self.shape = tuple(len(g)-1 for g in grids)
        self.component_size = int(np.prod(self.shape))

        # Keep univariate mass matrices in stencil format through periodic folding.
        self.mass_high_stencils = tuple(apply_periodic(v, assemble_mass1D(v)) for v in high)
        self.mass_low_stencils = tuple(apply_periodic(v, assemble_mass1D(v)) for v in low)
        mh = tuple(m.tosparse().tocsr() for m in self.mass_high_stencils)
        ml = tuple(m.tosparse().tocsr() for m in self.mass_low_stencils)
        dx, dy = (_derivative(h, l) for h, l in zip(high, low))
        ix, iy = (eye(n, format='csr') for n in self.shape)
        self.gradient = vstack((kron(dx, iy), kron(ix, dy)), format='csr')
        self.curl = hstack((-kron(ix, dy), kron(dx, iy)), format='csr')
        self.mass = self.epsilon*block_diag((kron(ml[0], mh[1]), kron(mh[0], ml[1])), format='csr')
        self.curl_mass = kron(ml[0], ml[1], format='csr')/self.mu
        self.stiffness = (self.curl.T @ self.curl_mass @ self.curl).tocsr()

        # Tensor mass inverses avoid factoring the full two-dimensional matrix.
        self._mass_factors = ((splu(ml[0].tocsc()), splu(mh[1].tocsc())),
                              (splu(mh[0].tocsc()), splu(ml[1].tocsc())))
        # Nonzero curl-curl eigenvalues are sums of the 1D generalized eigenvalues.
        maxima = [eigvalsh((d.T @ mlow @ d).toarray(), mhigh.toarray(),
                           subset_by_index=(n-1, n-1))[0]
                  for d, mlow, mhigh, n in zip((dx, dy), ml, mh, self.shape)]
        self.lambda_max = float(sum(maxima)/(self.epsilon*self.mu))
        self.dt_limit = 2/np.sqrt(self.lambda_max)
        self._points = tuple(v.points.ravel() for v in high)
        self._weights = np.outer(high[0].weights.ravel(), high[1].weights.ravel())
        self._load_bases = tuple(tuple(_periodic_basis(v, q) for v, q in zip(space.spaces, self._points))
                                 for space in self.component_spaces)

    def _coefficients(self, values):
        a = np.asarray(values, dtype=float)
        if a.shape != (2*self.component_size,) or not np.isfinite(a).all():
            raise ValueError('Expected a finite, flattened two-component coefficient vector')
        return a

    def solve_mass(self, rhs):
        """Apply the consistent mass inverse; no mass lumping is used."""
        rhs = self._coefficients(rhs)
        result = []
        for i, (fx, fy) in enumerate(self._mass_factors):
            block = rhs[i*self.component_size:(i+1)*self.component_size].reshape(self.shape)
            result.append(fy.solve(fx.solve(block).T).T.ravel()/self.epsilon)
        return np.concatenate(result)

    def load(self, field, time=0.0):
        """Integrate J_e(x,y,t) against the component bases.

        field(x,y,t) returns (Jx,Jy), broadcastable to the quadrature grid.
        This returns the load vector F, rather than source coefficients.
        """
        if field is None:
            return np.zeros(2*self.component_size)
        x, y = self._points[0][:, None], self._points[1][None, :]
        components = field(x, y, time)
        if len(components) != 2:
            raise ValueError('field must return two components')
        result = []
        for value, (bx, by) in zip(components, self._load_bases):
            weighted = np.broadcast_to(np.asarray(value, dtype=float), self._weights.shape)*self._weights
            result.append(np.asarray(by.T @ (bx.T @ weighted).T).T.ravel())
        return self._coefficients(np.concatenate(result))

    def project(self, field, time=0.0):
        """Componentwise L2 projection of field(x,y,t) onto H(curl)."""
        return self.solve_mass(self.epsilon*self.load(field, time))

    def acceleration(self, u, time=0.0, source=None):
        return self.solve_mass(self.load(source, time)-self.stiffness @ self._coefficients(u))

    def expand(self, u):
        """Return full periodic StencilVectors, one per component space."""
        u = self._coefficients(u)
        result = []
        for i, space in enumerate(self.component_spaces):
            values = apply_periodic(space, u[i*self.component_size:(i+1)*self.component_size], update=True)
            vector = StencilVector(space.vector_space)
            vector.from_array(space, values)
            result.append(vector)
        return tuple(result)

    def evaluate(self, u, x, y):
        """Sample the vector field on the Cartesian product of x and y."""
        u = self._coefficients(u)
        return tuple(np.asarray(_periodic_basis(space.spaces[1], y) @
                                (_periodic_basis(space.spaces[0], x) @
                                 u[i*self.component_size:(i+1)*self.component_size].reshape(self.shape)).T).T
                     for i, space in enumerate(self.component_spaces))

    def l2_error(self, u, exact, time=0.0):
        values = self.evaluate(u, *self._points)
        reference = exact(self._points[0][:, None], self._points[1][None, :], time)
        return float(np.sqrt(sum(np.sum(self._weights*(v-r)**2) for v, r in zip(values, reference))))


class CurlCurlLeapfrog:
    """Stagger u at integer times and velocity at half times.

    v^(1/2)=v^0+dt/2 M^-1(F^0-Au^0),
    u^(n+1)=u^n+dt v^(n+1/2),
    v^(n+3/2)=v^(n+1/2)+dt M^-1(F^(n+1)-Au^(n+1)).
    """

    def __init__(self, operator, u0, velocity0=None, dt=None, source=None, time=0.0):
        self.operator = operator
        self.dt = 0.9*operator.dt_limit if dt is None else float(dt)
        if not np.isfinite(self.dt) or not 0 < self.dt < operator.dt_limit:
            raise ValueError(f'Leapfrog requires 0 < dt < {operator.dt_limit:.8g}')
        self.time = float(time)
        if not np.isfinite(self.time):
            raise ValueError('Initial time must be finite')
        self._initial_time = self.time
        self.steps = 0
        self.source = source
        self.u = operator._coefficients(u0).copy()
        v0 = np.zeros_like(self.u) if velocity0 is None else operator._coefficients(velocity0)
        self.velocity_half = v0+0.5*self.dt*operator.acceleration(self.u, self.time, source)

    def step(self):
        self.u += self.dt*self.velocity_half
        self.steps += 1
        self.time = self._initial_time+self.steps*self.dt
        self.velocity_half += self.dt*self.operator.acceleration(self.u, self.time, self.source)
        return self.u

    def energy(self):
        """Conserved half-time discrete energy for J_e=0 (up to roundoff)."""
        op, v = self.operator, self.velocity_half
        return float(0.5*(v @ (op.mass @ v)+self.u @ (op.stiffness @ (self.u+self.dt*v))))
