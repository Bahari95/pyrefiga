"""Dirichlet vertices shared by patches whose incident edges are interfaces."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse.linalg import spsolve

from pyrefiga import SplineSpace, StencilMatrix, StencilNitsche, StencilVector, TensorSpace, pyref_multipatch
from pyrefiga.bsplines import greville
import pyrefiga.api as api

ROOT = Path(__file__).resolve().parents[2]


class CornerDirichletTests(unittest.TestCase):
    def solve_linear(self, ids, assembled_boundary, filename=None):
        mp = pyref_multipatch(str(filename or ROOT/'fields/unitSquare.xml'), ids)
        V = TensorSpace(*[SplineSpace(degree=2, grid=[0, .25, .5, .75, 1], quad_degree=2)
                          for _ in range(2)])
        W = mp.getspace(V)
        u, v = np.meshgrid(*[greville(k, p, False) for k, p in zip(V.knots, V.degree)], indexing='ij')
        exact = []
        for p in range(1, mp.nb_patches+1):
            x, y = mp.getcoefs(p)
            xp = x[0, 0]+u*(x[-1, 0]-x[0, 0])+v*(x[0, -1]-x[0, 0])
            yp = y[0, 0]+u*(y[-1, 0]-y[0, 0])+v*(y[0, -1]-y[0, 0])
            exact.append(1+xp+2*yp)
        if assembled_boundary:
            ud = mp.assemble_dirichlet(V, ['1+x+2*y'])
        else:
            ud = []
            for p, values in enumerate(exact, start=1):
                d = mp.getDirPatch(p)
                mask = np.ones(V.nbasis, dtype=bool)
                lo = [int(d[a][0]) for a in (0, 1)]
                hi = [V.nbasis[a]-int(d[a][1]) for a in (0, 1)]
                mask[lo[0]:hi[0], lo[1]:hi[1]] = False
                lift = StencilVector(V.vector_space)
                lift.from_array(V, np.where(mask, values, 0))
                ud.append(lift)
        ni = StencilNitsche(V, W, mp, ud)
        ni.assemble_nitsche()
        spec = importlib.util.spec_from_file_location('corner_poisson', ROOT/'docs/examples/gallery/gallery_section_06.py')
        gallery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gallery)
        for p in range(1, mp.nb_patches+1):
            A = api.assemble_matrix(gallery.assemble_matrix_un_ex01, W,
                                   fields=list(mp.stencil_mapping(p)),
                                   out=StencilMatrix(V.vector_space, V.vector_space))
            rhs = -(A.tosparse() @ ud[p-1].tensor.ravel())
            ni.append_block(api.apply_dirichlet(V, A, dirichlet=mp.getDirPatch(p)), p)
            ni.assemble_nitsche_rhs(rhs[ni._uniform_keep[p-1]], p)
        matrix = ni.nitsche_merge().tocsr()
        solution = spsolve(matrix, ni.nitsche_merge_rhs())
        for p, values in enumerate(exact, start=1):
            np.testing.assert_allclose(ni.extract_sol(solution, p).tensor, values, atol=1e-10)
        return mp, V, ni, solution

    def test_l_shape_prescribed_corner_and_patch_order(self):
        for ids in ((0, 1, 2), (1, 0, 2), (2, 1, 0)):
            for assembled_boundary in (False, True):
                with self.subTest(ids=ids, assembled_boundary=assembled_boundary):
                    mp, V, ni, solution = self.solve_linear(ids, assembled_boundary)
                    p = ids.index(0)+1
                    self.assertEqual(len(ni._corner_dofs), 1)
                    self.assertEqual(ni._corner_indices[p-1], [(V.nbasis[0]-1, V.nbasis[1]-1)])
                    self.assertTrue(all(ni.new_id[d] == -1 for d in ni._corner_dofs))
                    self.assertAlmostEqual(ni.u_d[p-1].tensor[-1, -1], 4)
                    # Both interface edges stay free except at the corner.
                    self.assertEqual(mp.getDirPatch(p), [[True, False], [True, False]])
                    extracted = ni.extract_sol(solution, p, u_last=StencilVector(V.vector_space))
                    self.assertAlmostEqual(extracted.tensor[-1, -1], 4)

    def test_reversed_interfaces_prescribe_the_correct_corner(self):
        tree = ET.parse(ROOT/'fields/unitSquare.xml')
        coefs = tree.find(".//Geometry[@id='0']//coefs")
        values = np.fromstring(coefs.text, sep=' ').reshape(3, 3, 2)[::-1, ::-1]
        coefs.text = '\n'.join(f'{x} {y}' for x, y in values.reshape(-1, 2))
        with TemporaryDirectory() as folder:
            filename = Path(folder)/'reversed_l_shape.xml'
            tree.write(filename)
            mp, _, ni, _ = self.solve_linear((0, 1, 2), True, filename)
            self.assertTrue(all(mp.isInterfaceReversed(i) for i in mp.getInterfaces()))
            self.assertEqual(ni._corner_indices[0], [(0, 0)])
            self.assertAlmostEqual(ni.u_d[0].tensor[0, 0], 4)

    def test_four_patch_interior_junction_stays_free(self):
        mp, V, ni, _ = self.solve_linear((0, 1, 2, 3), True)
        self.assertEqual(ni._corner_dofs, set())
        center = next(group for group in mp.getCornerGroups() if len(group) == 4)
        self.assertNotIn(center, mp.getDirichletCornerGroups())
        merged_ids = []
        for p, u, v in center:
            i, j = u*(V.nbasis[0]-1), v*(V.nbasis[1]-1)
            lo, hi = ni.elim_index[p-1, :, 0], ni.elim_index[p-1, :, 1]
            dof = ni._block_index[p-1]+(i-lo[0])*(hi[1]-lo[1])+j-lo[1]
            merged_ids.append(ni.new_id[dof])
        self.assertEqual(len(set(merged_ids)), 1)
        self.assertGreaterEqual(merged_ids[0], 0)

    def test_conflicting_corner_values_are_rejected(self):
        mp = pyref_multipatch(str(ROOT/'fields/unitSquare.xml'), (0, 1, 2))
        V = TensorSpace(*[SplineSpace(degree=2, grid=[0, .5, 1]) for _ in range(2)])
        ud = [StencilVector(V.vector_space) for _ in range(3)]
        ud[1][0, V.nbasis[1]-1] = 1
        ud[2][V.nbasis[0]-1, 0] = 2
        with self.assertRaisesRegex(ValueError, 'Incompatible Dirichlet values'):
            StencilNitsche(V, mp.getspace(V), mp, ud)


if __name__ == '__main__':
    unittest.main()
