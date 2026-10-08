"""Physical trace assembly for matching two-dimensional patch interfaces."""

import numpy as np
from scipy.sparse import coo_matrix

from .bsplines import basis_funs_all_ders, find_span


def normalized_knots(space, axis):
    knots = np.asarray(space.knots[axis])
    p = space.degree[axis]
    left, right = knots[p], knots[-p-1]
    return (knots - left) / (right - left)


def validate_trace_space(V, source_edge, neighbor_edge, reversed_trace):
    """Merging requires identical (possibly reflected) trace basis functions."""
    a, b = 1 - (source_edge-1)//2, 1 - (neighbor_edge-1)//2
    ka, kb = normalized_knots(V, a), normalized_knots(V, b)
    wa, wb = np.asarray(V.omega[a]), np.asarray(V.omega[b])
    if reversed_trace:
        kb, wb = 1.0-kb[::-1], wb[::-1]
    if (V.degree[a] != V.degree[b] or ka.shape != kb.shape
            or not np.allclose(ka, kb, rtol=0, atol=1e-12)
            or not np.allclose(wa/wa[0], wb/wb[0], rtol=1e-12, atol=1e-12)):
        raise ValueError("Interface DOF merging requires matching trace degrees, knots and weights")


def trace_indices(V, edge, reversed_trace=False):
    """Full tensor indices along an edge, optionally in reverse order."""
    axis, side = (edge-1)//2, (edge-1)%2
    coords = np.zeros((V.nbasis[1-axis], 2), dtype=int)
    coords[:, axis] = side*(V.nbasis[axis]-1)
    coords[:, 1-axis] = np.arange(V.nbasis[1-axis])
    indices = np.ravel_multi_index(coords.T, V.nbasis)
    return indices[::-1] if reversed_trace else indices


def _basis(space, axis, x):
    knots, p = np.asarray(space.knots[axis]), space.degree[axis]
    span = find_span(knots, p, x)
    values = basis_funs_all_ders(knots, p, x, span, 1)
    indices = np.arange(span-p, span+1)
    weights = np.asarray(space.omega[axis])[indices]
    weighted = values*weights
    denominator, derivative = weighted.sum(axis=1)
    value = weighted[0]/denominator
    deriv = (weighted[1]-value*derivative)/denominator
    return indices, value, deriv


def _tensor_basis(space, edge, s):
    axis, side = (edge-1)//2, (edge-1)%2
    coordinates = []
    for a in range(2):
        knots, p = space.knots[a], space.degree[a]
        left, right = knots[p], knots[-p-1]
        coordinates.append(left+(right-left)*(side if a == axis else s))
    i, u, du = _basis(space, 0, coordinates[0])
    j, v, dv = _basis(space, 1, coordinates[1])
    indices = np.ravel_multi_index(
        (np.repeat(i, len(j)), np.tile(j, len(i))), space.nbasis
    )
    values = np.outer(u, v).ravel()
    derivatives = np.column_stack((np.outer(du, v).ravel(), np.outer(u, dv).ravel()))
    return indices, values, derivatives


def _trace(V, mapping, mapping_space, edge, s):
    ids, values, derivatives = _tensor_basis(V, edge, s)
    mid, mv, md = _tensor_basis(mapping_space, edge, s)
    control = np.asarray(mapping.coefs).reshape(2, -1)[:, mid]
    point, jacobian = control @ mv, control @ md
    if np.linalg.det(jacobian) <= 0:
        raise ValueError("Interface mapping must have a positive nonsingular Jacobian")
    inverse = np.linalg.inv(jacobian)
    axis, side = (edge-1)//2, (edge-1)%2
    normal = (2*side-1)*inverse.T[:, axis]
    normal /= np.linalg.norm(normal)
    tangent = 1-axis
    knots, p = mapping.knots[tangent], mapping.degree[tangent]
    measure = np.linalg.norm(jacobian[:, tangent])*(knots[-p-1]-knots[p])
    flux = (derivatives @ inverse) @ normal
    return ids, values, flux, point, normal, measure


def assemble_interface(V, source, neighbor, edges, reversed_trace, Kappa, normalS, kernel):
    """Return full patch diagonal and cross blocks for one interface."""
    a, b = edges
    validate_trace_space(V, a, b, reversed_trace)
    source_space, neighbor_space = source.space, neighbor.space
    ta, tb = 1-(a-1)//2, 1-(b-1)//2
    # Include geometry breaks as well as both trace meshes in quadrature.
    breaks = [normalized_knots(V, ta), normalized_knots(source_space, ta)]
    for space in (V, neighbor_space):
        knots = normalized_knots(space, tb)
        breaks.append(1-knots[::-1] if reversed_trace else knots)
    breaks = np.unique(np.concatenate(breaks))
    breaks = breaks[(breaks >= 0) & (breaks <= 1)]
    order = max(V.degree[ta], V.degree[tb], *source.degree, *neighbor.degree)+1
    gauss, gw = np.polynomial.legendre.leggauss(order)
    nq = (len(breaks)-1)*order
    local_size = 2*(V.degree[0]+1)*(V.degree[1]+1)
    jumps, fluxes = np.empty((nq, local_size)), np.empty((nq, local_size))
    ids = np.empty((nq, local_size), dtype=int)
    weights = np.empty(nq)
    n = int(np.prod(V.nbasis))
    q = 0
    for left, right in zip(breaks[:-1], breaks[1:]):
        for t, w in zip(gauss, gw):
            s = (left+right)/2+(right-left)*t/2
            ip, vp, fp, xp, np_, ds = _trace(V, source, source_space, a, s)
            iq, vq, fq, xq, nq_, dsq = _trace(V, neighbor, neighbor_space, b, 1-s if reversed_trace else s)
            if not (np.allclose(xp, xq, rtol=1e-10, atol=1e-10)
                    and np.allclose(np_, -nq_, atol=1e-9)
                    and np.isclose(ds, dsq, rtol=1e-9, atol=1e-10)):
                raise ValueError("Interface mappings must have matching physical traces and opposite normals")
            ids[q] = np.concatenate((ip, n+iq))
            jumps[q] = np.concatenate((vp, -vq))
            fluxes[q] = np.concatenate((fp, -fq))
            weights[q] = w*(right-left)/2*ds
            q += 1
    matrices = np.empty((nq, local_size, local_size))
    kernel(jumps, fluxes, weights, Kappa, normalS, matrices)
    rows = np.broadcast_to(ids[:, :, None], matrices.shape).ravel()
    cols = np.broadcast_to(ids[:, None, :], matrices.shape).ravel()
    matrix = coo_matrix((matrices.ravel(), (rows, cols)), shape=(2*n, 2*n)).tocsr()
    matrix.eliminate_zeros()
    return matrix[:n, :n], matrix[n:, n:], matrix[n:, :n]
