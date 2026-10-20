"""Single-patch NURBS elasticity using RR-8111, equations (3), (8) and (14).

Full bivariate rational weights are retained, including nonseparable nets.
Both displacement components use the geometry's scalar NURBS basis.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import bmat, coo_matrix, diags, kron
from scipy.sparse.linalg import spsolve

from .bsplines import basis_funs_all_ders, find_span, greville, hrefinement_matrix
from .spaces import SplineSpace


def kirsch_stress(x, y, tension=10., radius=1.):
    """Return (sigma_xx, sigma_yy, sigma_xy) for remote x-directed tension."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    r2 = x*x+y*y
    if np.any(r2 == 0):
        raise ValueError('Kirsch stress is undefined at the hole center')
    q = radius*radius/r2
    c2, s2 = (x*x-y*y)/r2, 2*x*y/r2
    rr = .5*tension*(1-q+(1-4*q+3*q*q)*c2)
    tt = .5*tension*(1+q-(1+3*q*q)*c2)
    rt = -.5*tension*(1+2*q-3*q*q)*s2
    return np.array([x*x/r2*rr+y*y/r2*tt-s2*rt,
                     y*y/r2*rr+x*x/r2*tt+s2*rt,
                     .5*s2*(rr-tt)+c2*rt])


def principal_stress(stress):
    """Largest in-plane principal stress, distinct from Cartesian sigma_xx."""
    xx, yy, xy = stress
    return .5*(xx+yy)+np.hypot(.5*(xx-yy), xy)


def _basis(knots, degree, points):
    """Sparse univariate B-spline value and first-derivative matrices."""
    points = np.asarray(points, dtype=float)
    n = len(knots)-degree-1
    rows, columns, values, derivatives = [], [], [], []
    for row, point in enumerate(points):
        span = find_span(knots, degree, point)
        local = basis_funs_all_ders(knots, degree, point, span, 1)
        rows.extend([row]*(degree+1))
        columns.extend(range(span-degree, span+1))
        values.extend(local[0])
        derivatives.extend(local[1])
    return tuple(coo_matrix((data, (rows, columns)), shape=(len(points), n)).tocsr()
                 for data in (values, derivatives))


def _transform_axis(data, matrix, axis):
    moved = np.moveaxis(data, axis, 0)
    result = matrix @ moved.reshape(moved.shape[0], -1)
    return np.moveaxis(np.asarray(result).reshape((-1, *moved.shape[1:])), 0, axis)


class NurbsElasticity2D:
    """Assemble integral 2 mu eps(u):eps(v) + lambda div(u) div(v).

    Coefficients are ordered [all u_x, all u_y], each in tensor C-order.
    Degree elevation and knot insertion act on homogeneous control points,
    preserving the geometry and its continuity. Plane stress is the default.
    The constructor uses an XML geometry, not the separable-weight patch
    evaluator: the plate benchmark has a genuinely bivariate denominator.
    """

    def __init__(self, filename, patch_id=0, degree=None, refinement=0,
                 young=1e5, poisson=.3, plane='stress', quad_degree=None):
        if not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not -1 < poisson < .5:
            raise ValueError('Use finite E > 0 and -1 < nu < 0.5')
        if plane not in ('stress', 'strain'):
            raise ValueError('plane must be stress or strain')
        if not isinstance(refinement, int) or refinement < 0:
            raise ValueError('refinement must be a nonnegative integer')
        geometry = ET.parse(filename).getroot().find(f".//Geometry[@id='{patch_id}']")
        if geometry is None:
            raise ValueError(f'No geometry with id {patch_id}')
        bases = sorted(geometry.findall(".//Basis[@type='BSplineBasis']"),
                       key=lambda basis: int(basis.get('index')))
        if len(bases) != 2 or int(geometry.find('.//coefs').get('geoDim')) != 2:
            raise ValueError('Provide a planar tensor-product NURBS patch')
        knots = [np.fromstring(b.find('KnotVector').text, sep=' ') for b in bases]
        degrees = [int(b.find('KnotVector').get('degree')) for b in bases]
        shape = tuple(len(k)-p-1 for k, p in zip(knots, degrees))
        control = np.fromstring(geometry.find('.//coefs').text, sep=' ').reshape((*shape, 2))
        weights_element = geometry.find('.//weights')
        weights = (np.ones(shape) if weights_element is None else
                   np.fromstring(weights_element.text, sep=' ').reshape(shape))
        if (not np.isfinite(control).all() or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError('NURBS control points must be finite and weights positive')
        homogeneous = np.concatenate((control*weights[..., None], weights[..., None]), axis=-1)
        targets = degrees if degree is None else ([degree]*2 if np.isscalar(degree) else list(degree))
        if len(targets) != 2 or any(not isinstance(p, (int, np.integer)) or p < old
                                    for p, old in zip(targets, degrees)):
            raise ValueError('Analysis degrees must be integers at least the geometry degrees')
        for axis, target in enumerate(targets):
            if target != degrees[axis]:
                breaks, counts = np.unique(knots[axis], return_counts=True)
                elevated = np.repeat(breaks, counts+target-degrees[axis])
                # The old space is nested in the elevated one. Collocation
                # gives its exact change of basis, including repeated knots.
                points = greville(elevated, int(target), False)
                transform = np.linalg.solve(_basis(elevated, target, points)[0].toarray(),
                                            _basis(knots[axis], degrees[axis], points)[0].toarray())
                homogeneous = _transform_axis(homogeneous, transform, axis)
                knots[axis], degrees[axis] = elevated, int(target)
            if refinement:
                breaks = np.unique(knots[axis])
                inserted = np.concatenate([np.linspace(a, b, 2**refinement+1)[1:-1]
                                           for a, b in zip(breaks[:-1], breaks[1:])])
                transform = hrefinement_matrix(inserted, degrees[axis], knots[axis])
                homogeneous = _transform_axis(homogeneous, transform, axis)
                knots[axis] = np.sort(np.concatenate((knots[axis], inserted)))
        self.knots, self.degree = tuple(knots), tuple(degrees)
        self.shape = homogeneous.shape[:2]
        self.size = int(np.prod(self.shape))
        self.rational_weights = homogeneous[..., 2].ravel()
        self.control = (homogeneous[..., :2]/homogeneous[..., 2, None]).reshape(-1, 2)
        self.young, self.poisson, self.plane = young, poisson, plane
        self.mu = young/(2*(1+poisson))
        self.lam = (young*poisson/(1-poisson**2) if plane == 'stress' else
                    young*poisson/((1+poisson)*(1-2*poisson)))
        self.constitutive = np.array([[self.lam+2*self.mu, self.lam, 0],
                                      [self.lam, self.lam+2*self.mu, 0], [0, 0, self.mu]])
        order = max(degrees)+3 if quad_degree is None else quad_degree
        if not isinstance(order, int) or order < max(degrees):
            raise ValueError('quad_degree must be an integer at least the largest spline degree')
        self.spaces = tuple(SplineSpace(p, grid=k[p:-p], nderiv=1, quad_degree=order)
                            for k, p in zip(knots, degrees))
        self.nelements = tuple(v.nelements for v in self.spaces)
        self.points = tuple(v.points.ravel() for v in self.spaces)
        self.basis, du, dv = self.reference_basis(*self.points)
        self.coordinates = self.basis @ self.control
        self.dx, self.dy, determinant = self.physical_gradients(du, dv)
        if np.any(determinant <= 0) or not np.isfinite(determinant).all():
            raise ValueError('Geometry must have a positive Jacobian at integration points')
        self.weights = np.outer(*(v.weights.ravel() for v in self.spaces)).ravel()*determinant
        weight = diags(self.weights)
        xx, yy = self.dx.T @ weight @ self.dx, self.dy.T @ weight @ self.dy
        xy = self.dx.T @ weight @ self.dy
        a11 = (self.lam+2*self.mu)*xx+self.mu*yy
        a22 = self.mu*xx+(self.lam+2*self.mu)*yy
        a12 = self.lam*xy+self.mu*xy.T
        self.stiffness = bmat([[a11, a12], [a12.T, a22]], format='csr')

    def reference_basis(self, xi, eta):
        """Full bivariate rational values and derivatives on a tensor grid."""
        points = (np.asarray(xi, dtype=float), np.asarray(eta, dtype=float))
        for values, knots, p in zip(points, self.knots, self.degree):
            if (values.ndim != 1 or not np.isfinite(values).all()
                    or np.any(values < knots[p]) or np.any(values > knots[-p-1])):
                raise ValueError('Parameter points must be finite 1D arrays inside the knot domain')
        (a, da), (b, db) = [_basis(k, p, x) for k, p, x in zip(self.knots, self.degree, points)]
        matrices = [kron(a, b, format='csr'), kron(da, b, format='csr'), kron(a, db, format='csr')]
        weighted = [matrix @ diags(self.rational_weights) for matrix in matrices]
        denominator, wu, wv = [np.asarray(matrix.sum(axis=1)).ravel() for matrix in weighted]
        rational = diags(1/denominator) @ weighted[0]
        return (rational.tocsr(),
                (diags(1/denominator) @ weighted[1]-diags(wu/denominator) @ rational).tocsr(),
                (diags(1/denominator) @ weighted[2]-diags(wv/denominator) @ rational).tocsr())

    def physical_gradients(self, du, dv):
        a, c = (du @ self.control).T
        b, d = (dv @ self.control).T
        determinant = a*d-b*c
        # The square plate's corner is a singular parameter point. Stress
        # sampled exactly there is undefined; retain NaN rather than invent it.
        inverse = np.divide(1., determinant, out=np.full_like(determinant, np.nan),
                            where=np.abs(determinant) > 1e-13)
        return ((diags(d*inverse) @ du-diags(c*inverse) @ dv).tocsr(),
                (diags(a*inverse) @ dv-diags(b*inverse) @ du).tocsr(), determinant)

    def edge_data(self, axis, side):
        """Boundary basis, physical points, outward normals and line weights."""
        if axis not in (0, 1) or side not in (0, 1):
            raise ValueError('Edge axis and side must be 0 or 1')
        points = list(self.points)
        k, p = self.knots[axis], self.degree[axis]
        points[axis] = np.array([k[p] if side == 0 else k[-p-1]])
        basis, du, dv = self.reference_basis(*points)
        tangent = (dv if axis == 0 else du) @ self.control
        sign = (2*side-1) if axis == 0 else (1-2*side)
        normal = sign*np.column_stack((tangent[:, 1], -tangent[:, 0]))
        length = np.linalg.norm(normal, axis=1)
        if np.any(length <= 0):
            raise ValueError('Boundary Jacobian must be positive at integration points')
        return basis, basis @ self.control, normal/length[:, None], self.spaces[1-axis].weights.ravel()*length

    def assemble_load(self, traction, edges=((0, 1),)):
        result = np.zeros((2, self.size))
        for axis, side in edges:
            basis, coordinates, normals, weights = self.edge_data(axis, side)
            values = np.broadcast_to(np.asarray(traction(*coordinates.T, *normals.T), dtype=float),
                                     (2, len(weights)))
            for i in range(2):
                result[i] += basis.T @ (weights*values[i])
        if not np.isfinite(result).all():
            raise ValueError('Traction must be finite')
        return result.ravel()

    def solve(self, traction, edges=((0, 1),), prescribed=None):
        """Solve with exact elimination; NaN in prescribed denotes a free DOF.

        By default impose u_y=0 at eta=0 and u_x=0 at eta=1,
        the symmetry conditions of the first-quadrant plate benchmark.
        """
        if prescribed is None:
            prescribed = np.full((2, *self.shape), np.nan)
            prescribed[0, :, -1] = 0
            prescribed[1, :, 0] = 0
        prescribed = np.asarray(prescribed, dtype=float).ravel()
        if prescribed.size != 2*self.size or np.isinf(prescribed).any():
            raise ValueError('Prescribed values need one entry per vector DOF, with NaN for free DOFs')
        self.free = np.flatnonzero(np.isnan(prescribed))
        self.load = self.assemble_load(traction, edges)
        result = np.nan_to_num(prescribed)
        rhs = (self.load-self.stiffness @ result)[self.free]
        matrix = self.stiffness[self.free][:, self.free].tocsc()
        result[self.free] = spsolve(matrix, rhs)
        residual = (self.stiffness @ result-self.load)[self.free]
        self.relative_residual = np.linalg.norm(residual)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
        if not np.isfinite(result).all() or self.relative_residual > 1e-8:
            raise RuntimeError(f'Elasticity solve failed: relative residual {self.relative_residual:.3e}')
        return result

    def stress(self, coefficients, dx=None, dy=None):
        u = np.asarray(coefficients, dtype=float).reshape(2, self.size)
        dx, dy = (self.dx, self.dy) if dx is None else (dx, dy)
        strain = np.array([dx @ u[0], dy @ u[1], dy @ u[0]+dx @ u[1]])
        return self.constitutive @ strain

    def errors(self, coefficients, exact):
        difference = self.stress(coefficients)-exact(*self.coordinates.T)
        principal = principal_stress(self.stress(coefficients))-principal_stress(exact(*self.coordinates.T))
        energy_density = np.sum(difference*np.linalg.solve(self.constitutive, difference), axis=0)
        return dict(L2_sigma_xx=float(np.sqrt(self.weights @ difference[0]**2)),
                    L2_principal=float(np.sqrt(self.weights @ principal**2)),
                    energy_error=float(np.sqrt(self.weights @ energy_density)))

    def sample(self, coefficients, xi, eta):
        basis, du, dv = self.reference_basis(xi, eta)
        dx, dy, determinant = self.physical_gradients(du, dv)
        shape = (len(xi), len(eta))
        u = np.asarray(coefficients).reshape(2, self.size)
        return dict(coordinates=(basis @ self.control).T.reshape((2, *shape)),
                    displacement=np.array([basis @ component for component in u]).reshape((2, *shape)),
                    stress=self.stress(coefficients, dx, dy).reshape((3, *shape)),
                    determinant=determinant.reshape(shape))
