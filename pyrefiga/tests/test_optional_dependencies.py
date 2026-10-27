"""Core numerical operations must not require plotting or compiler packages."""

import subprocess
import sys
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import BSpline

from pyrefiga import SplineSpace, compile_kernel, plot_field_1d
from pyrefiga._optional import import_optional
from pyrefiga.ad_mesh_core import assemble_massmatrix1D
from pyrefiga.results import point_on_bspline_curve
from pyrefiga.post_processing_paraview import (
    paraview_nurbsAdMeshMultipatch, paraview_nurbsSolutionMultipatch,
    paraview_TimeSolutionMultipatch,
)


class OptionalDependencyTests(unittest.TestCase):
    def test_curve_sampling_and_optional_plot(self):
        space = SplineSpace(2, nelements=3)
        coefficients = np.linspace(-1., 2., space.nbasis)
        points = np.linspace(0., 1., 17)
        expected = BSpline(space.knots, coefficients, space.degree)(points)
        sampled = [point_on_bspline_curve(space.knots, coefficients[:, None], x)[0]
                   for x in points]
        np.testing.assert_allclose(sampled, expected, atol=1e-14)
        if find_spec('matplotlib') is not None:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots()
            try:
                plot_field_1d(space.knots, space.degree, coefficients, nx=len(points))
                np.testing.assert_allclose(axes.lines[-1].get_ydata(), expected, atol=1e-14)
            finally:
                plt.close(figure)

    def test_core_import_does_not_load_optional_packages(self):
        script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyccel', 'matplotlib', 'pyvista'}:
            raise ModuleNotFoundError(name=fullname.split('.')[0])

sys.meta_path.insert(0, BlockOptional())
import numpy as np
from pyrefiga import SplineSpace, compile_kernel, sol_field_2d
from pyrefiga.ad_mesh_core import assemble_massmatrix1D
space = SplineSpace(2, nelements=3)
mass = compile_kernel(assemble_massmatrix1D, arity=2)(space).tosparse()
np.testing.assert_allclose(mass.sum(), 1., atol=1e-14)
assert not {'pyccel', 'matplotlib', 'pyvista'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_explicit_python_kernel_does_not_import_compiler(self):
        space = SplineSpace(2, nelements=3)
        with patch('pyrefiga.api.import_optional') as compiler:
            mass = compile_kernel(assemble_massmatrix1D, arity=2, pyccel=False)(space)
        compiler.assert_not_called()
        np.testing.assert_allclose(mass.tosparse().sum(), 1., atol=1e-14)

    def test_available_compiler_is_used(self):
        compiler = SimpleNamespace(epyccel=Mock(return_value=assemble_massmatrix1D))
        for mode in (None, True):
            with self.subTest(mode=mode), patch('pyrefiga.api.import_optional', return_value=compiler):
                kernel = compile_kernel(assemble_massmatrix1D, arity=2, pyccel=mode)
                self.assertIs(kernel.args[0], assemble_massmatrix1D)
        self.assertEqual(compiler.epyccel.call_count, 2)

    def test_missing_compiler_falls_back_unless_explicitly_required(self):
        missing = ModuleNotFoundError("No module named 'pyccel'", name='pyccel')
        with patch('pyrefiga._optional.import_module', side_effect=missing):
            kernel = compile_kernel(assemble_massmatrix1D, arity=2)
            self.assertIs(kernel.args[0], assemble_massmatrix1D)
            with self.assertRaisesRegex(ImportError, r'pyrefiga\[accelerate\]'):
                compile_kernel(assemble_massmatrix1D, arity=2, pyccel=True)

    def test_compilation_failure_is_not_silently_ignored(self):
        compiler = SimpleNamespace(epyccel=Mock(side_effect=RuntimeError('compiler failed')))
        with patch('pyrefiga.api.import_optional', return_value=compiler):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                compile_kernel(assemble_massmatrix1D, arity=2)

    def test_missing_visualization_has_install_instructions(self):
        missing = ModuleNotFoundError("No module named 'matplotlib'", name='matplotlib')
        with patch('pyrefiga._optional.import_module', side_effect=missing):
            with self.assertRaisesRegex(ImportError, r'pyrefiga\[visualization\]'):
                plot_field_1d([], 0, [])
        missing = ModuleNotFoundError("No module named 'pyvista'", name='pyvista')
        for exporter in (paraview_nurbsAdMeshMultipatch, paraview_nurbsSolutionMultipatch,
                         paraview_TimeSolutionMultipatch):
            with self.subTest(exporter=exporter.__name__):
                with patch('pyrefiga._optional.import_module', side_effect=missing):
                    with self.assertRaisesRegex(ImportError, r'pyrefiga\[visualization\]'):
                        if exporter is paraview_nurbsAdMeshMultipatch:
                            exporter(5, None, moving_mesh=None)
                        else:
                            exporter(5, None)

    def test_broken_dependency_is_not_treated_as_absent(self):
        missing = ModuleNotFoundError("No module named 'dependency_of_pyccel'",
                                      name='dependency_of_pyccel')
        with patch('pyrefiga._optional.import_module', side_effect=missing):
            with self.assertRaises(ModuleNotFoundError) as error:
                import_optional('pyccel', 'accelerate', required=False)
        self.assertIs(error.exception, missing)


if __name__ == '__main__':
    unittest.main()
