"""Checks of the full-weight NURBS plate-with-hole elasticity example."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from pyrefiga import load_xml, point_on_nurbs_surface
from pyrefiga.elasticity import NurbsElasticity2D, kirsch_stress, principal_stress


def exact_traction(x, y, nx, ny):
    xx, yy, xy = kirsch_stress(x, y)
    return xx*nx+xy*ny, xy*nx+yy*ny


class ElasticityTests(unittest.TestCase):
    def test_kirsch_free_hole_and_equilibrium(self):
        angle = np.linspace(0, np.pi/2, 31)
        x, y = np.cos(angle), np.sin(angle)
        xx, yy, xy = kirsch_stress(x, y)
        np.testing.assert_allclose(xx*x+xy*y, 0, atol=2e-14)
        np.testing.assert_allclose(xy*x+yy*y, 0, atol=2e-14)
        self.assertAlmostEqual(kirsch_stress(0., 1.)[0], 30.)
        np.testing.assert_allclose(kirsch_stress(1e8, 0), [10, 0, 0], atol=1e-13)
        x, y, step = 1.5, 1.7, 1e-5
        dx = (kirsch_stress(x+step, y)-kirsch_stress(x-step, y))/(2*step)
        dy = (kirsch_stress(x, y+step)-kirsch_stress(x, y-step))/(2*step)
        np.testing.assert_allclose([dx[0]+dy[2], dx[2]+dy[1]], 0, atol=2e-9)
        self.assertEqual(principal_stress([0., 0., 2.]), 2.)

    def test_full_nurbs_geometry_degree_elevation_and_refinement(self):
        filename = load_xml('elasticity.xml')
        original = ET.parse(filename).getroot().find(".//Geometry[@id='0']")
        control = np.fromstring(original.find('.//coefs').text, sep=' ').reshape(3, 4, 2)
        weights = np.fromstring(original.find('.//weights').text, sep=' ').reshape(3, 4)
        self.assertGreater(np.linalg.matrix_rank(weights), 1)
        knots = [np.fromstring(k.text, sep=' ') for k in original.findall('.//KnotVector')]
        for degree in (2, 3, 4):
            operator = NurbsElasticity2D(filename, degree=degree, refinement=1)
            for xi, eta in ((0., .23), (.41, .37), (1., .75)):
                basis, du, dv = operator.reference_basis([xi], [eta])
                expected = point_on_nurbs_surface(*knots, control, weights, xi, eta)
                np.testing.assert_allclose((basis @ operator.control)[0], expected, atol=3e-14)
                np.testing.assert_allclose(basis.sum(axis=1), 1, atol=3e-14)
                np.testing.assert_allclose(du.sum(axis=1), 0, atol=3e-14)
                np.testing.assert_allclose(dv.sum(axis=1), 0, atol=3e-14)
            inner = operator.edge_data(0, 0)[1]
            np.testing.assert_allclose(np.sum(inner*inner, axis=1), 1, atol=2e-14)
            self.assertAlmostEqual(operator.weights.sum(), 16-np.pi/4, places=10)

    def test_rigid_modes_and_affine_patch(self):
        for plane in ('stress', 'strain'):
            operator = NurbsElasticity2D(load_xml('elasticity.xml'), degree=3,
                                         refinement=1, young=17, plane=plane)
            x, y = operator.control.T
            for mode in (np.r_[np.ones_like(x), np.zeros_like(x)],
                         np.r_[np.zeros_like(x), np.ones_like(x)], np.r_[-y, x]):
                np.testing.assert_allclose(operator.stiffness @ mode, 0, atol=1e-12)
            # A physical affine displacement is represented exactly by the
            # rational geometry basis, even when its weights are nonseparable.
            expected = np.r_[.7*x-.2*y+.1, .4*x+.3*y-.2]
            prescribed = np.full((2, *operator.shape), np.nan)
            boundary = np.zeros(operator.shape, dtype=bool)
            boundary[[0, -1], :] = True
            boundary[:, [0, -1]] = True
            prescribed[:, boundary] = expected.reshape((2, *operator.shape))[:, boundary]
            solution = operator.solve(lambda x, y, nx, ny: (0., 0.), edges=(), prescribed=prescribed)
            np.testing.assert_allclose(solution, expected, atol=2e-13)
            stress = operator.constitutive @ np.array([.7, .3, .2])
            np.testing.assert_allclose(operator.stress(solution),
                                       np.broadcast_to(stress[:, None], (3, len(operator.weights))), atol=2e-12)
            density = np.array([.7, .3, .2]) @ stress
            self.assertAlmostEqual(solution @ (operator.stiffness @ solution),
                                   density*operator.weights.sum(), places=10)
            self.assertLess(operator.relative_residual, 1e-12)

    def test_traction_normals_and_consistent_boundary_load(self):
        operator = NurbsElasticity2D(load_xml('elasticity.xml'), refinement=1)
        outer, _, normals, weights = operator.edge_data(0, 1)
        # Right and top sides of the square carry unit outward normals.
        np.testing.assert_allclose(np.sum(normals, axis=1), 1, atol=2e-14)
        self.assertAlmostEqual(weights.sum(), 8, places=12)
        load = operator.assemble_load(lambda x, y, nx, ny: (2*nx+ny, nx+3*ny))
        np.testing.assert_allclose(load.reshape(2, -1).sum(axis=1), [12, 16], atol=2e-12)
        np.testing.assert_allclose(outer.sum(axis=1), 1, atol=2e-14)

    def test_stress_convergence_energy_and_symmetry(self):
        final_errors, orders = [], []
        for degree in (2, 3, 4):
            errors, energies = [], []
            for refinement in (1, 2, 3):
                operator = NurbsElasticity2D(load_xml('elasticity.xml'), degree=degree,
                                             refinement=refinement)
                solution = operator.solve(exact_traction)
                errors.append(operator.errors(solution, kirsch_stress)['L2_sigma_xx'])
                energies.append(solution @ (operator.stiffness @ solution))
                self.assertAlmostEqual(energies[-1], solution @ operator.load, places=12)
                np.testing.assert_allclose(solution.reshape(2, *operator.shape)[0, :, -1], 0)
                np.testing.assert_allclose(solution.reshape(2, *operator.shape)[1, :, 0], 0)
            self.assertTrue(all(b > a for a, b in zip(energies, energies[1:])))
            self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))
            final_errors.append(errors[-1])
            orders.append(np.log2(errors[-2]/errors[-1]))
        self.assertTrue(final_errors[2] < final_errors[1] < final_errors[0])
        self.assertTrue(orders[2] > orders[1] > orders[0] > 1.4)

    def test_example_export_and_importable_main(self):
        import pyvista as pv
        from docs.examples.elasticity2d_example import main

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            prefix = str(Path(directory)/'plate')
            operator, solution, rows = main(['--refinement', '1', '--levels', '1',
                                             '--export', '--nbpts', '10', '--output', prefix])
            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(prefix+'_convergence.csv').is_file())
            grid = pv.read(prefix+'.vtm')[0]
            points = np.linspace(0, 1, 10)
            sampled = operator.sample(solution, points, points)
            np.testing.assert_allclose(grid['sigma_xx'], sampled['stress'][0].ravel(order='F'))
            np.testing.assert_allclose(grid.points[:, 0], sampled['coordinates'][0].ravel(order='F'))
            self.assertTrue(np.isfinite(grid['principal_stress']).all())

    def test_invalid_material_and_geometry(self):
        with self.assertRaisesRegex(ValueError, 'E > 0'):
            NurbsElasticity2D(load_xml('elasticity.xml'), young=-1)
        with self.assertRaisesRegex(ValueError, 'geometry degrees'):
            NurbsElasticity2D(load_xml('elasticity.xml'), degree=1)
        from docs.examples.elasticity2d_example import main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            main(['--geometry', 'unitSquare.xml', '--levels', '1', '--output', str(Path(directory)/'bad')])


if __name__ == '__main__':
    unittest.main()
