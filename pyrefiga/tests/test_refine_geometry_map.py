"""Geometry refinement preserves polynomial and rational tensor-product maps."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import BSpline

from pyrefiga.utilities import pyref_patch, pyref_multipatch


def make_patch(degrees, grids, geo_dim, nurbs):
    """Build coefficient data without the XML reader's orientation correction."""
    patch = pyref_patch.__new__(pyref_patch)
    patch._degree = list(degrees)
    patch._dim = len(degrees)
    patch._geo_dim = geo_dim
    patch._nurbs = nurbs
    patch.knots_data = [np.r_[[grid[0]] * p, grid, [grid[-1]] * p]
                        for p, grid in zip(degrees, grids)]
    patch._grids = [knots[p:-p] for p, knots in zip(degrees, patch.knots_data)]
    patch._nbasis = [len(knots)-p-1 for p, knots in zip(degrees, patch.knots_data)]
    patch._nelements = [len(np.unique(grid))-1 for grid in grids]
    rng = np.random.default_rng(42)
    patch._coefs = rng.normal(size=(geo_dim, *patch.nbasis))
    # General NURBS weights need not factor into one vector per direction.
    patch._weights = rng.uniform(0.5, 2.0, size=np.prod(patch.nbasis))
    return patch


def evaluate(knots, degrees, coefs, weights):
    values = np.stack(coefs, axis=-1) * weights[..., None]
    values = np.concatenate((weights[..., None], values), axis=-1)
    for axis, (t, p) in enumerate(zip(knots, degrees)):
        points = np.linspace(t[p], t[-p-1], 13)
        basis = BSpline(t, np.eye(len(t)-p-1), p)(points)
        values = np.moveaxis(np.tensordot(basis, values, axes=(1, axis)), 0, axis)
    return values[..., 1:] / values[..., :1]


class RefineGeometryMapTests(unittest.TestCase):
    def check_geometry(self, patch, levels):
        original_coefs = patch._coefs.copy()
        original_weights = patch._weights.copy()
        original_knots = [knots.copy() for knots in patch.knots]
        result = patch.RefineGeometryMap(levels)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), patch.geo_dim + int(patch.nurbs))
        if patch.nurbs:
            weights, *coefs = result
        else:
            coefs = result
            weights = np.ones(coefs[0].shape)
        if levels == 0:
            np.testing.assert_array_equal(coefs, original_coefs)
            if patch.nurbs:
                np.testing.assert_array_equal(weights, original_weights.reshape(patch.nbasis))

        refined_knots = []
        for axis, knots in enumerate(original_knots):
            grid = patch.Refinegrid(axis, levels)
            inserted = np.setdiff1d(grid, knots)
            refined_knots.append(np.sort(np.r_[knots, inserted]))
            spans = len(np.unique(knots))-1
            self.assertEqual(coefs[0].shape[axis], patch.nbasis[axis] + spans*(2**levels-1))
        original_w = (original_weights.reshape(patch.nbasis) if patch.nurbs
                      else np.ones(patch.nbasis))
        expected = evaluate(original_knots, patch.degree, original_coefs, original_w)
        actual = evaluate(refined_knots, patch.degree, coefs, weights)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(patch._coefs, original_coefs)
        np.testing.assert_array_equal(patch._weights, original_weights)
        for before, after in zip(original_knots, patch.knots):
            np.testing.assert_array_equal(before, after)
        for array in result:
            self.assertFalse(np.shares_memory(array, patch._coefs))
            self.assertFalse(np.shares_memory(array, patch._weights))

    def test_polynomial_and_rational_maps(self):
        cases = [((2, 1), ([0, .2, .2, .7, 1], [-2, 0, 3]), 2),
                 ((1, 2), ([0, .4, 1], [0, .6, 1]), 3),
                 ((1, 2, 1), ([0, .3, 1], [-1, 0, 2], [2, 5]), 3)]
        for degrees, grids, geo_dim in cases:
            for nurbs in (False, True):
                for levels in (0, 1, 2):
                    with self.subTest(degrees=degrees, nurbs=nurbs, levels=levels):
                        self.check_geometry(make_patch(degrees, grids, geo_dim, nurbs), levels)

    def test_discontinuous_and_degree_zero_maps(self):
        for degrees, grids in [((2, 1), ([0, .5, .5, .5, 1], [0, 1])),
                               ((0, 1), ([0, .3, 1], [0, 1]))]:
            for nurbs in (False, True):
                with self.subTest(degrees=degrees, nurbs=nurbs):
                    self.check_geometry(make_patch(degrees, grids, 2, nurbs), 2)

    def test_invalid_refinement_counts(self):
        patch = make_patch((1, 1), ([0, 1], [0, 1]), 2, True)
        for method in (patch.RefineGeometryMap, lambda n: patch.Refinegrid(0, n)):
            for value in (-1, -3):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    method(value)
            for value in (1.5, '1', None, True, np.bool_(False)):
                with self.subTest(value=value), self.assertRaises(TypeError):
                    method(value)
        self.check_geometry(patch, np.int64(1))

    def test_refined_grid_preserves_multiplicities(self):
        patch = make_patch((2, 1), ([0, .4, .4, 1], [0, 1]), 2, True)
        np.testing.assert_allclose(patch.Refinegrid(0, 2),
                                   [0, .1, .2, .3, .4, .4, .55, .7, .85, 1])
        grid = patch.Refinegrid(0, 0)
        grid[0] = -1
        self.assertEqual(patch.grids[0][0], 0)
        self.assertEqual(patch.knots[0][0], 0)

    def test_xml_quarter_annulus(self):
        filename = Path(__file__).resolve().parents[2] / 'fields' / 'quart_annulus.xml'
        self.check_geometry(pyref_patch(filename, '0'), 2)


class InPlaceRefinementTests(unittest.TestCase):
    def test_geometry_and_metadata(self):
        cases = [((2, 1), ([0, .2, .2, .7, 1], [-2, 0, 3]), 2),
                 ((1, 2), ([0, .4, 1], [0, .6, 1]), 3),
                 ((1, 2, 1), ([0, .3, 1], [-1, 0, 2], [2, 5]), 3),
                 ((2, 1), ([0, .5, .5, .5, 1], [0, 1]), 2),
                 ((0, 1), ([0, .3, 1], [0, 1]), 2)]
        for degrees, grids, geo_dim in cases:
            for nurbs in (False, True):
                with self.subTest(degrees=degrees, geo_dim=geo_dim, nurbs=nurbs):
                    g = make_patch(degrees, grids, geo_dim, nurbs)
                    identity = g
                    g.boundaries = [1, 3]
                    old_nbasis = g.nbasis.copy()
                    old_nelements = g.nelements.copy()
                    w = g._weights.reshape(g.nbasis) if nurbs else np.ones(g.nbasis)
                    expected = evaluate(g.knots, g.degree, g.coefs, w)
                    self.assertIsNone(g.UnRefine(2))
                    self.assertIs(g, identity)
                    self.assertEqual(g.degree, list(degrees))
                    self.assertEqual(g.nelements, [4*ne for ne in old_nelements])
                    self.assertEqual(g.nbasis, [nb+3*ne for nb, ne in zip(old_nbasis, old_nelements)])
                    self.assertEqual(g._coefs.shape, (geo_dim, *g.nbasis))
                    self.assertEqual(g._weights.shape, (g.ndim,))
                    self.assertEqual(g.boundaries, [1, 3])
                    self.assertEqual(g.nurbs, nurbs)
                    for axis, (knots, p) in enumerate(zip(g.knots, degrees)):
                        np.testing.assert_array_equal(g.grids[axis], knots[p:len(knots)-p])
                        self.assertEqual(len(knots), g.nbasis[axis]+p+1)
                    w = g._weights.reshape(g.nbasis) if nurbs else np.ones(g.nbasis)
                    np.testing.assert_allclose(evaluate(g.knots, g.degree, g.coefs, w),
                                               expected, rtol=2e-13, atol=2e-13)

    def test_repeated_calls_refine_current_patch(self):
        for nurbs in (False, True):
            sequential = make_patch((2, 1), ([0, .3, .3, 1], [-1, 2]), 2, nurbs)
            combined = make_patch((2, 1), ([0, .3, .3, 1], [-1, 2]), 2, nurbs)
            sequential.UnRefine()
            sequential.UnRefine(np.int64(2))
            combined.UnRefine(3)
            self.assertEqual(sequential.knots_data, combined.knots_data)
            np.testing.assert_allclose(sequential._weights, combined._weights, atol=1e-14)
            np.testing.assert_allclose(sequential.coefs, combined.coefs, atol=1e-14)

    def test_no_op_and_invalid_counts_preserve_state(self):
        g = make_patch((1, 1), ([0, 1], [0, 1]), 2, True)
        state = g.__dict__.copy()
        self.assertIsNone(g.UnRefine(0))
        for value in (-1, 1.5, '1', None, True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                g.UnRefine(value)
        for key, value in state.items():
            self.assertIs(g.__dict__[key], value)

    def test_xml_evaluation_and_spaces(self):
        fields = Path(__file__).resolve().parents[2] / 'fields'
        for name, nurbs in (('quart_annulus.xml', True), ('cube.xml', False)):
            with self.subTest(name=name):
                g = pyref_patch(fields / name, '0', nurbs=nurbs, Dirichlet_all=False)
                expected = g.eval(nbpts=7)
                expected_gradient = g.gradient(nbpts=7)
                g.UnRefine(2)
                np.testing.assert_allclose(g.eval(nbpts=7), expected, atol=2e-13)
                np.testing.assert_allclose(g.gradient(nbpts=7), expected_gradient, atol=2e-13)
                self.assertEqual(list(g.space.nbasis), g.nbasis)
                self.assertEqual(list(g.space.nelements), g.nelements)
                for original, rebuilt in zip(g.knots, g.space.knots):
                    np.testing.assert_array_equal(original, rebuilt)
                for coef, vector in zip(g.coefs, g.stencil_mapping):
                    np.testing.assert_array_equal(coef, vector.tensor)
                self.assertFalse(g.boundaries)


class MultipatchRefinementTests(unittest.TestCase):
    def check_refinement(self, g):
        patches = list(g.patches)
        interfaces = [(p, q, list(edges)) for p, q, edges in g.getInterfaces()]
        orientations = g.interface_reversed.copy()
        dirichlet = np.array(g.dirichlet)
        corners = g.getCornerGroups()
        expected = [g.eval(i, nbpts=7) for i in range(1, g.nb_patches+1)]
        gradients = [g.gradient(i, nbpts=7) for i in range(1, g.nb_patches+1)]
        original_nelements = [patch.nelements.copy() for patch in patches]
        original_space = g.space
        self.assertIsNone(g.UnRefine(2))
        self.assertIsNot(g.space, original_space)
        self.assertEqual(g.getInterfaces(), interfaces)
        self.assertEqual(g.interface_reversed, orientations)
        np.testing.assert_array_equal(g.dirichlet, dirichlet)
        self.assertEqual(g.getCornerGroups(), corners)
        for i, (patch, nelements) in enumerate(zip(patches, original_nelements), start=1):
            self.assertIs(g.get_patch(i), patch)
            self.assertEqual(patch.nelements, [4*ne for ne in nelements])
            np.testing.assert_allclose(g.eval(i, nbpts=7), expected[i-1], atol=2e-13)
            np.testing.assert_allclose(g.gradient(i, nbpts=7), gradients[i-1], atol=2e-13)
            for coef, vector in zip(g.getcoefs(i), g.stencil_mapping(i)):
                np.testing.assert_array_equal(coef, vector.tensor)
        first = patches[0]
        self.assertEqual(g.degree, first.degree)
        self.assertEqual(g.grids, first.grids)
        self.assertEqual(list(g.space.nbasis), first.nbasis)
        self.assertEqual(list(g.space.nelements), first.nelements)
        for cached, actual, space_knots in zip(g.knots, first.knots, g.space.knots):
            np.testing.assert_array_equal(cached, actual)
            np.testing.assert_array_equal(space_knots, actual)
        for cached, actual, omega in zip(g.weights, first.weights, g.space.omega):
            np.testing.assert_array_equal(cached, actual)
            np.testing.assert_array_equal(omega, actual)

    def test_xml_geometries_and_shared_metadata(self):
        fields = Path(__file__).resolve().parents[2] / 'fields'
        for filename, ids, nurbs in [('quart_annulus.xml', (0, 1), True),
                                     ('unitSquare.xml', (0, 1, 2), False),
                                     ('cube.xml', (0,), False)]:
            with self.subTest(filename=filename):
                self.check_refinement(pyref_multipatch(fields / filename, ids, nurbs=nurbs))

    def test_reversed_interfaces(self):
        fields = Path(__file__).resolve().parents[2] / 'fields'
        tree = ET.parse(fields / 'unitSquare.xml')
        coefs = tree.find(".//Geometry[@id='0']//coefs")
        values = np.fromstring(coefs.text, sep=' ').reshape(3, 3, 2)[::-1, ::-1]
        coefs.text = '\n'.join(f'{x} {y}' for x, y in values.reshape(-1, 2))
        with TemporaryDirectory() as folder:
            filename = Path(folder) / 'reversed.xml'
            tree.write(filename)
            g = pyref_multipatch(filename, (0, 1, 2))
            self.assertTrue(all(g.interface_reversed.values()))
            self.check_refinement(g)

    def test_repeated_calls_and_invalid_counts(self):
        filename = Path(__file__).resolve().parents[2] / 'fields' / 'quart_annulus.xml'
        g = pyref_multipatch(filename, (0, 1), Dirichlet_all=False)
        state = g.__dict__.copy()
        patch_states = [patch.__dict__.copy() for patch in g.patches]
        self.assertIsNone(g.UnRefine(0))
        for value in (-1, 1.5, '1', None, True, np.bool_(False)):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                g.UnRefine(value)
        for current, original in [(g.__dict__, state)] + list(zip(
                [patch.__dict__ for patch in g.patches], patch_states)):
            for key, value in original.items():
                self.assertIs(current[key], value)
        g.UnRefine()
        g.UnRefine(np.int64(1))
        combined = pyref_multipatch(filename, (0, 1), Dirichlet_all=False)
        combined.UnRefine(2)
        for a, b in zip(g.patches, combined.patches):
            self.assertEqual(a.knots_data, b.knots_data)
            np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-14)
            np.testing.assert_allclose(a._weights, b._weights, atol=1e-14)
        self.assertFalse(np.any(g.dirichlet))


if __name__ == '__main__':
    unittest.main()
