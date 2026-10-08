"""All edge pairs, orientation-aware merging and manufactured solutions."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import spsolve

from pyrefiga import SplineSpace, StencilMatrix, StencilNitsche, StencilVector, TensorSpace, pyref_multipatch
from pyrefiga.bsplines import greville
from pyrefiga.interfaces import trace_indices
import pyrefiga.api as api

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rectangle_patch(pid, edge, source):
    """Positive affine mapping to [-1,0]x[0,1] or [0,1]x[0,1]."""
    axis, side = (edge-1)//2, (edge-1)%2
    sign = (1 if side else -1) if source else (-1 if side else 1)
    tangent_sign = sign if axis == 0 else -sign
    coords = []
    for u, v in ((0, 0), (0, 1), (1, 0), (1, 1)):
        n, t = (u, v)[axis], (u, v)[1-axis]
        x = (n-1 if side else -n) if source else (1-n if side else n)
        coords.append(f'{x} {t if tangent_sign == 1 else 1-t}')
    bases = ''.join(
        f'<Basis type="BSplineBasis" index="{i}">'
        '<KnotVector degree="1">0 0 1 1</KnotVector></Basis>'
        for i in (0, 1)
    )
    return (f'<Geometry type="TensorNurbs2" id="{pid}">'
            '<Basis type="TensorNurbsBasis2"><Basis type="TensorBSplineBasis2">'
            + bases + '<coefs geoDim="2">' + '\n'.join(coords)
            + '</coefs></Basis></Basis></Geometry>'), tangent_sign


def geometry(folder, a, b):
    pa, ta = rectangle_patch(0, a, True)
    pb, tb = rectangle_patch(1, b, False)
    path = Path(folder) / f'pair_{a}_{b}.xml'
    path.write_text('<xml>'+pa+pb+'</xml>')
    mp = pyref_multipatch(str(path), (0, 1))
    assert mp.getInterfaces() == [(1, 2, [a, b])]
    assert mp.isInterfaceReversed(mp.getInterfaces()[0]) == (ta != tb)
    return mp


def space(degree, grid=None):
    if grid is None:
        grid = [0, .1, .3, .7, .9, 1]
    return TensorSpace(*[SplineSpace(degree=p, grid=grid, quad_degree=p) for p in degree])


def exact_coefficients(mp, patch_nb, V):
    u, v = np.meshgrid(*[greville(k, p, False) for k, p in zip(V.knots, V.degree)], indexing='ij')
    c = np.asarray(mp.getcoefs(patch_nb))
    xy = (c[:, 0, 0, None, None]*(1-u)*(1-v)
          + c[:, 0, 1, None, None]*(1-u)*v
          + c[:, 1, 0, None, None]*u*(1-v)
          + c[:, 1, 1, None, None]*u*v)
    return xy[0]+2*xy[1]


class AllInterfaceTests(unittest.TestCase):
    def test_cancellation_and_physical_dof_pairing(self):
        source = load_source('nitsche_source', ROOT/'pyrefiga/nitsche_core.py')
        with TemporaryDirectory() as folder:
            for core in (source, api.n_core):
                for a in range(1, 5):
                    for b in range(1, 5):
                        degrees = [(1, 1), (2, 2)]
                        if (a-1)//2 == (b-1)//2:
                            degrees.append((2, 3))
                        for degree in degrees:
                            with self.subTest(core=core.__file__, edges=(a, b), degree=degree):
                                mp = geometry(folder, a, b)
                                V = space(degree)
                                with patch.object(api, 'n_core', core):
                                    ni = StencilNitsche(V, mp.getspace(V), mp)
                                    ni.assemble_nitsche()
                                np.testing.assert_allclose(ni.nitsche_merge().tocsr().data, 0, atol=1e-9)
                                reverse = mp.isInterfaceReversed(mp.getInterfaces()[0])
                                ip, iq = trace_indices(V, a), trace_indices(V, b, reverse)
                                ep, eq = exact_coefficients(mp, 1, V), exact_coefficients(mp, 2, V)
                                np.testing.assert_allclose(ep.ravel()[ip], eq.ravel()[iq], atol=1e-14)
                                # Verify actual merge equivalence at interior trace DOFs.
                                old = []
                                for p in range(2):
                                    lookup = np.full(int(np.prod(V.nbasis)), -1)
                                    lookup[ni._uniform_keep[p]] = ni._block_index[p]+np.arange(ni._nbasis[p])
                                    old.append(lookup)
                                for i, j in zip(ip[1:-1], iq[1:-1]):
                                    self.assertEqual(ni.new_id[old[0][i]], ni.new_id[old[1][j]])

    def test_linear_harmonic_solution_all_pairs(self):
        gallery = load_source('poisson_gallery', ROOT/'docs/examples/gallery/gallery_section_06.py')
        with TemporaryDirectory() as folder:
            for a in range(1, 5):
                for b in range(1, 5):
                    for degree in ((1, 1), (2, 2)):
                        with self.subTest(edges=(a, b), degree=degree):
                            mp = geometry(folder, a, b)
                            V = space(degree, [0, .25, .5, .75, 1])
                            W = mp.getspace(V)
                            ud, exact = [], []
                            for p in (1, 2):
                                values = exact_coefficients(mp, p, V)
                                exact.append(values)
                                boundary = np.ones(V.nbasis, dtype=bool)
                                d = mp.getDirPatch(p)
                                start = [1 if d[i][0] else 0 for i in (0, 1)]
                                stop = [V.nbasis[i]-int(d[i][1]) for i in (0, 1)]
                                boundary[start[0]:stop[0], start[1]:stop[1]] = False
                                lift = StencilVector(V.vector_space)
                                lift.from_array(V, np.where(boundary, values, 0))
                                ud.append(lift)
                            ni = StencilNitsche(V, W, mp, ud)
                            ni.assemble_nitsche()
                            for p in (1, 2):
                                stiffness = api.assemble_matrix(
                                    gallery.assemble_matrix_un_ex01, W,
                                    fields=list(mp.stencil_mapping(p)),
                                    out=StencilMatrix(V.vector_space, V.vector_space),
                                )
                                rhs = -(stiffness.tosparse() @ ud[p-1].tensor.ravel())
                                ni.append_block(api.apply_dirichlet(V, stiffness, dirichlet=mp.getDirPatch(p)), p)
                                ni.assemble_nitsche_rhs(rhs[ni._uniform_keep[p-1]], p)
                                previous = ni.b_dir.copy()
                                ni.assemble_nitsche_rhs(np.zeros(ni._nbasis[p-1]), p, accumulate=True)
                                np.testing.assert_array_equal(ni.b_dir, previous)
                            # Check consistency before merging as well: a
                            # cancellation-only test can hide incorrect fluxes.
                            exact_reduced = np.concatenate([
                                exact[p].ravel()[ni._uniform_keep[p]] for p in (0, 1)
                            ])
                            np.testing.assert_allclose(
                                ni.stencilNitsche @ exact_reduced-ni.b_dir, 0, atol=1e-10,
                            )
                            solution = spsolve(ni.nitsche_merge().tocsr(), ni.nitsche_merge_rhs())
                            for p in (1, 2):
                                np.testing.assert_allclose(ni.extract_sol(solution, p).tensor, exact[p-1], atol=1e-10)

    def test_nonmatching_trace_is_rejected(self):
        with TemporaryDirectory() as folder:
            mp = geometry(folder, 4, 2)
            V = space((1, 2))
            ni = StencilNitsche(V, mp.getspace(V), mp)
            with self.assertRaisesRegex(ValueError, 'matching trace'):
                ni.assemble_nitsche()
            # A failed preparation must not leave a usable partial cache.
            with self.assertRaisesRegex(ValueError, 'matching trace'):
                ni.assemble_nitsche()
            mp = geometry(folder, 4, 1)  # Reversed trace needs reflected knots.
            V = space((1, 1), [0, .1, .5, 1])
            ni = StencilNitsche(V, mp.getspace(V), mp)
            with self.assertRaisesRegex(ValueError, 'matching trace'):
                ni.assemble_nitsche()


if __name__ == '__main__':
    unittest.main()
