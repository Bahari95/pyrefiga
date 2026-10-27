"""Run with python -I against an installed wheel, outside the source checkout."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pyrefiga
from pyrefiga import SplineSpace, assemble_mass1D, compile_kernel, load_xml
from pyrefiga.ad_mesh_core import assemble_massmatrix1D
from pyrefiga.elasticity import NurbsElasticity2D


class InstalledPackageTests(unittest.TestCase):
    def test_import_uses_installed_package(self):
        checkout = Path(__file__).resolve().parents[1]
        self.assertFalse(Path(pyrefiga.__file__).resolve().is_relative_to(checkout))

    def test_core_import_does_not_load_optional_dependencies(self):
        self.assertFalse({'pyccel', 'matplotlib', 'pyvista'} & sys.modules.keys())

    def test_mass_assembly_without_compilation(self):
        space = SplineSpace(2, nelements=4)
        mass = assemble_mass1D(space).tosparse().toarray()
        kernel = compile_kernel(assemble_massmatrix1D, arity=2, pyccel=False)
        np.testing.assert_allclose(kernel(space).tosparse().toarray(), mass)
        np.testing.assert_allclose(mass.sum(), 1., atol=1e-14)
        np.testing.assert_allclose(mass, mass.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(mass).min(), 0.)

    def test_bundled_geometry_and_elasticity_patch(self):
        for name in ('elasticity.xml', 'triangle.xml', 'quart_annulus.xml', 'unitSquare.xml'):
            filename = Path(load_xml(name))
            self.assertEqual(filename.parent, Path(pyrefiga.__file__).parent/'fields')
            ET.parse(filename)
        operator = NurbsElasticity2D(load_xml('elasticity.xml'), young=17.)
        x, y = operator.control.T
        expected = np.r_[.7*x-.2*y+.1, .4*x+.3*y-.2]
        prescribed = np.full((2, *operator.shape), np.nan)
        boundary = np.zeros(operator.shape, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        prescribed[:, boundary] = expected.reshape((2, *operator.shape))[:, boundary]
        solution = operator.solve(lambda *args: (0., 0.), edges=(), prescribed=prescribed)
        np.testing.assert_allclose(solution, expected, atol=2e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
