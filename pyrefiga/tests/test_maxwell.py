"""Checks of the periodic de Rham complex and curl-curl time integration."""

import unittest
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

from pyrefiga.maxwell import PeriodicCurlCurl2D, CurlCurlLeapfrog


class MaxwellTests(unittest.TestCase):
    def test_nonuniform_derivative_and_complex(self):
        op = PeriodicCurlCurl2D(degree=(3, 2), grids=(
            np.array([0, .07, .2, .4, .65, .8, 1]), np.array([0, .1, .3, .55, .7, 1])))
        rng = np.random.default_rng(5)
        phi = rng.normal(size=op.component_size)
        gradient = op.gradient @ phi
        np.testing.assert_allclose(op.curl @ gradient, 0, atol=3e-13)
        np.testing.assert_allclose(op.stiffness @ gradient, 0, atol=3e-11)
        x, y = np.linspace(0, 1, 17), np.linspace(0, 1, 19)
        scalar = op.scalar_space
        from pyrefiga import apply_periodic
        full = apply_periodic(scalar, phi, update=True)
        bx = BSpline(scalar.spaces[0].knots, np.eye(scalar.nbasis[0]), scalar.degree[0])
        by = BSpline(scalar.spaces[1].knots, np.eye(scalar.nbasis[1]), scalar.degree[1])
        ux, uy = op.evaluate(gradient, x, y)
        np.testing.assert_allclose(ux, bx(x, nu=1) @ full @ by(y).T, atol=2e-13)
        np.testing.assert_allclose(uy, bx(x) @ full @ by(y, nu=1).T, atol=2e-13)

    def test_periodic_traces_and_curl_quadrature(self):
        op = PeriodicCurlCurl2D(degree=3, nelements=(5, 6), epsilon=2, mu=3)
        u = np.random.default_rng(2).normal(size=2*op.component_size)
        ux, uy = op.evaluate(u, np.linspace(0, 1, 13), np.linspace(0, 1, 15))
        np.testing.assert_allclose(uy[0], uy[-1], atol=1e-13)
        np.testing.assert_allclose(ux[:, 0], ux[:, -1], atol=1e-13)
        # Compare the assembled quadratic form against derivatives of full splines.
        x, y = op._points
        derivatives = []
        for i, (space, field) in enumerate(zip(op.component_spaces, op.expand(u))):
            bx = BSpline(space.spaces[0].knots, np.eye(space.nbasis[0]), space.degree[0])
            by = BSpline(space.spaces[1].knots, np.eye(space.nbasis[1]), space.degree[1])
            derivatives.append(bx(x, nu=int(i == 1)) @ field.tensor @ by(y, nu=int(i == 0)).T)
        scalar_curl = derivatives[1]-derivatives[0]
        self.assertAlmostEqual(u @ (op.stiffness @ u), np.sum(op._weights*scalar_curl**2)/op.mu, places=10)
        sampled = op.evaluate(u, x, y)
        self.assertAlmostEqual(u @ (op.mass @ u), op.epsilon*sum(np.sum(op._weights*v**2) for v in sampled), places=12)
        np.testing.assert_allclose(op.solve_mass(op.mass @ u), u, atol=2e-13)

    def test_spectrum_energy_and_time_accuracy(self):
        op = PeriodicCurlCurl2D(nelements=5, epsilon=1.7, mu=2.3)
        eigenvalues, eigenvectors = eigh(op.stiffness.toarray(), op.mass.toarray())
        self.assertAlmostEqual(op.lambda_max, eigenvalues[-1], places=10)
        k = np.flatnonzero(eigenvalues > 1e-8)[0]
        mode, omega = eigenvectors[:, k], np.sqrt(eigenvalues[k])
        errors = []
        for steps in (20, 40, 80):
            state = CurlCurlLeapfrog(op, mode, dt=0.4/steps)
            initial_energy = state.energy()
            for _ in range(steps):
                state.step()
            error = state.u-np.cos(omega*state.time)*mode
            errors.append(np.sqrt(error @ (op.mass @ error)))
            self.assertAlmostEqual(state.energy()/initial_energy, 1, places=12)
            np.testing.assert_allclose(op.gradient.T @ (op.mass @ state.u), 0, atol=2e-13)
        self.assertTrue(3.9 < errors[0]/errors[1] < 4.1)
        self.assertTrue(3.9 < errors[1]/errors[2] < 4.1)
        with self.assertRaisesRegex(ValueError, 'Leapfrog requires'):
            CurlCurlLeapfrog(op, mode, dt=op.dt_limit*1.01)

    def test_source_and_initial_velocity(self):
        op = PeriodicCurlCurl2D(nelements=5, epsilon=2.5)
        constant = op.project(lambda x, y, t: (1., -2.))
        # u(t)=(t+t^2)*(1,-2); curl u=0 and J=epsilon*(2,-4).
        state = CurlCurlLeapfrog(op, np.zeros_like(constant), velocity0=constant, dt=.01,
                                source=lambda x, y, t: (2*op.epsilon, -4*op.epsilon))
        for _ in range(20):
            state.step()
        np.testing.assert_allclose(state.u, (state.time+state.time**2)*constant, atol=2e-13)

        # Exercise a time-dependent source in a mode with nonzero curl.
        eigenvalues, eigenvectors = eigh(op.stiffness.toarray(), op.mass.toarray())
        mode, eigenvalue = eigenvectors[:, -1], eigenvalues[-1]

        def source(x, y, t):
            values = op.evaluate(mode, x.ravel(), y.ravel())
            return tuple(op.epsilon*(2+eigenvalue*t*t)*v for v in values)

        t0 = .2
        state = CurlCurlLeapfrog(op, t0*t0*mode, velocity0=2*t0*mode, dt=.01, source=source, time=t0)
        for _ in range(20):
            state.step()
        np.testing.assert_allclose(state.u, state.time**2*mode, atol=2e-12)

    def test_spatial_convergence(self):
        errors = []
        for n in (6, 12, 24):
            op = PeriodicCurlCurl2D(nelements=n)
            exact = lambda x, y, t: (np.sin(2*np.pi*y), np.sin(2*np.pi*x))
            errors.append(op.l2_error(op.project(exact), exact))
        self.assertTrue(all(a/b > 7 for a, b in zip(errors, errors[1:])))


if __name__ == '__main__':
    unittest.main()
