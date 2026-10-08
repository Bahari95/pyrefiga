"""Interface terms must cancel when matching trace DOFs are merged."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from pyrefiga import SplineSpace, StencilNitsche, TensorSpace, pyref_multipatch
import pyrefiga.api as api


class TriangleInterfaceTests(unittest.TestCase):
    def test_merged_interface_terms_cancel(self):
        root = Path(__file__).resolve().parents[2]
        # Test source even when an older Pyccel extension shadows the .py file.
        spec = importlib.util.spec_from_file_location(
            "nitsche_source", root / "pyrefiga" / "nitsche_core.py"
        )
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)

        for core in (source, api.n_core):
            for ids, edges in (((0, 2), [4, 3]), ((1, 2), [4, 2])):
                for degree in (1, 2):
                    with self.subTest(core=core.__file__, ids=ids, degree=degree):
                        mp = pyref_multipatch(str(root / "fields" / "triangle.xml"), ids)
                        self.assertEqual(mp.getInterfaces(), [(1, 2, edges)])
                        V = TensorSpace(*[
                            SplineSpace(
                                degree=degree,
                                grid=mp.Refinegrid(axis, numElevate=3),
                                quad_degree=degree,
                            )
                            for axis in range(2)
                        ])
                        with patch.object(api, "n_core", core):
                            ni = StencilNitsche(V, mp.getspace(V), mp)
                            ni.assemble_nitsche()
                        # A continuous trial/test trace has no jump. Neither
                        # penalty nor consistency terms should survive merging.
                        merged = ni.nitsche_merge().tocsr()
                        np.testing.assert_allclose(merged.data, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
