"""G-alpha Jacobian, rejected trials and accepted adaptive time steps."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from pyrefiga import SplineSpace, TensorSpace, StencilVector, apply_periodic


examples = Path(__file__).resolve().parents[2]/'docs'/'examples'
sys.path.insert(0, str(examples))
spec = importlib.util.spec_from_file_location('cahn_hilliard_galpha', examples/'cahn_Hilliard2d_Galpha_example.py')
ga = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ga)
finally:
    sys.path.remove(str(examples))


class GeneralizedAlphaTests(unittest.TestCase):
    def space(self, n=6):
        v1 = SplineSpace(2, nelements=n, periodic=True, nderiv=2)
        v2 = SplineSpace(2, nelements=n, periodic=True, nderiv=2)
        return v1, v2, TensorSpace(v1, v2)

    def test_gallery_jacobian_matches_residual_derivative(self):
        v1, v2, v = self.space(4)
        rng = np.random.default_rng(4)
        x = apply_periodic(v, .6+.02*rng.normal(size=(4, 4)), update=True)
        tx = apply_periodic(v, rng.normal(size=(4, 4)), update=True)
        direction = rng.normal(size=(4, 4))
        extended = apply_periodic(v, direction, update=True)
        dt, alpha, rho = .001, 30, .2
        am, af, gamma = ga.generalized_alpha_parameters(rho)

        def vector(coefficients):
            u = StencilVector(v.vector_space)
            u.from_array(v, coefficients)
            return u

        def residual(delta):
            uf = vector(x+delta*af*gamma*dt*extended)
            um = vector(tx+delta*am*extended)
            return apply_periodic(v, ga.assemble2_rhs(v, fields=[uf, um], value=[alpha])).toarray()

        matrix = apply_periodic(v, ga.assemble2_stiffness(v, fields=[vector(x)], value=[dt, alpha, rho]))
        h = 1e-4
        derivative = (residual(h)-residual(-h))/(2*h)
        np.testing.assert_allclose(matrix.tosparse() @ direction.ravel(), derivative, rtol=2e-7, atol=1e-8)

    def test_adaptive_rejection_preserves_inputs_and_mass(self):
        v1, v2, v = self.space()
        u, x, ut, tx, mass, initial_energy = ga.Proj_solve(v1, v2, v, 6000, np.random.default_rng(7))
        copies = [a.copy() for a in (x, tx, u._data, ut._data)]
        result, used_dt, next_dt, error, rejected = ga.adaptive_step(
            v1, v2, v, u, ut, x, tx, mass, 1e-6, 6000, atol=1e-9, rtol=1e-7)
        self.assertGreater(rejected, 0)
        self.assertLess(used_dt, 1e-6)
        self.assertLessEqual(error, 1)
        self.assertTrue(1e-12 <= next_dt <= 1e-4)
        for a, b in zip((x, tx, u._data, ut._data), copies):
            np.testing.assert_array_equal(a, b)
        ones = np.ones(mass.shape[0])
        before = ones @ (mass @ x[:6, :6].ravel())
        after = ones @ (mass @ result[1][:6, :6].ravel())
        self.assertAlmostEqual(before, after, places=12)
        self.assertLess(result[-1], initial_energy)
        np.testing.assert_array_equal(result[1][-2:, :], result[1][:2, :])
        np.testing.assert_array_equal(result[1][:, -2:], result[1][:, :2])

    def test_failed_trial_preserves_inputs(self):
        v1, v2, v = self.space()
        u, x, ut, tx, mass, energy = ga.Proj_solve(v1, v2, v, 6000, np.random.default_rng(7))
        copies = [a.copy() for a in (x, tx, u._data, ut._data)]
        with self.assertRaises(ga.NonlinearConvergenceError):
            ga.Cahn_Hliard_solve(v1, v2, v, u, ut, x, tx, 1e-4, 6000, N_iter=1)
        for a, b in zip((x, tx, u._data, ut._data), copies):
            np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(RuntimeError, 'Adaptive G-alpha step failed'):
            ga.adaptive_step(v1, v2, v, u, ut, x, tx, mass, 1e-4, 6000,
                             dt_min=1e-4, dt_max=1e-4, N_iter=1)


if __name__ == '__main__':
    unittest.main()
