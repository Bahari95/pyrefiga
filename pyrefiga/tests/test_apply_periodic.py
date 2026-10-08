"""Periodic folding equals coefficient identification without dense matrices."""

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import coo_matrix

from pyrefiga import SplineSpace, TensorSpace, StencilMatrix, StencilVector, apply_periodic


class PeriodicStencilTests(unittest.TestCase):
    def check_fold(self, degrees, periodic, nelements=5):
        spaces = [SplineSpace(degree=p, grid=np.linspace(0, 1, nelements+1), periodic=flag)
                  for p, flag in zip(degrees, periodic)]
        V = spaces[0] if len(spaces) == 1 else TensorSpace(*spaces)
        shape = tuple(space.nbasis for space in spaces)
        reduced = tuple(n-p if flag else n for n, p, flag in zip(shape, degrees, periodic))
        A = StencilMatrix(V.vector_space, V.vector_space)
        rng = np.random.default_rng(7)
        for row in np.ndindex(shape):
            for offset in product(*[range(-p, p+1) for p in degrees]):
                column = tuple(i+k for i, k in zip(row, offset))
                if all(0 <= j < n for j, n in zip(column, shape)):
                    A[row+offset] = rng.normal()
        full_indices = np.indices(shape)
        target = tuple(full_indices[a] % reduced[a] if periodic[a] else full_indices[a]
                       for a in range(len(shape)))
        labels = np.ravel_multi_index(target, reduced).ravel()
        P = coo_matrix((np.ones(labels.size), (np.arange(labels.size), labels)),
                       shape=(labels.size, int(np.prod(reduced)))).tocsr()
        reference = P.T @ A.tosparse().tocsr() @ P
        original = A._data.copy()
        # Reduction must operate on stencil data, without a conversion detour.
        with patch.object(StencilMatrix, 'tosparse', side_effect=AssertionError('conversion during folding')):
            folded = apply_periodic(V, A, periodic)
        self.assertIsInstance(folded, StencilMatrix)
        self.assertEqual(folded.domain.npts, reduced)
        self.assertEqual(folded.domain.periods, periodic)
        np.testing.assert_allclose(folded.tosparse().toarray(), reference.toarray(), atol=1e-13)
        np.testing.assert_array_equal(A._data, original)

        vector = StencilVector(V.vector_space)
        values = rng.normal(size=shape)
        vector.from_array(V, values)
        folded_vector = apply_periodic(V, vector, periodic)
        self.assertIsInstance(folded_vector, StencilVector)
        np.testing.assert_allclose(folded_vector.toarray(), P.T @ values.ravel(), atol=1e-13)
        np.testing.assert_array_equal(vector.tensor, values)

        independent = rng.normal(size=reduced)
        reduced_vector = StencilVector(folded_vector.space)
        owned = tuple(slice(p, p+n) for p, n in zip(degrees, reduced))
        reduced_vector._data[owned] = independent
        extended = apply_periodic(V, reduced_vector, periodic, update=True)
        self.assertIsInstance(extended, StencilVector)
        np.testing.assert_array_equal(extended.toarray(), P @ independent.ravel())
        extended_array = apply_periodic(V, independent, periodic, update=True)
        np.testing.assert_array_equal(extended_array, extended.tensor)
        extended_flat = apply_periodic(V, independent.ravel(), periodic, update=True)
        np.testing.assert_array_equal(extended_flat, extended.tensor)

    def test_all_periodicity_combinations(self):
        for degrees in ((2,), (1, 2), (1, 2, 1)):
            for periodic in product((False, True), repeat=len(degrees)):
                with self.subTest(degrees=degrees, periodic=periodic):
                    self.check_fold(degrees, periodic)

    def test_small_periodic_space_with_aliased_diagonals(self):
        self.check_fold((2,), (True,), nelements=2)
        self.check_fold((2, 2), (True, True), nelements=2)

    def test_default_and_invalid_flags(self):
        V = SplineSpace(degree=2, grid=np.linspace(0, 1, 6), periodic=True)
        vector = StencilVector(V.vector_space)
        self.assertEqual(apply_periodic(V, vector).space.npts, (V.nbasis-V.degree,))
        with self.assertRaisesRegex(ValueError, 'one flag'):
            apply_periodic(V, vector, [True, False])
        with self.assertRaisesRegex(ValueError, 'Coefficient size'):
            apply_periodic(V, np.zeros(2), update=True)


if __name__ == '__main__':
    unittest.main()
