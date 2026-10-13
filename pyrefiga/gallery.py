__all__ = ['assemble_stiffness_1d',
           'assemble_rhs_1d',
           'assemble_stiffness_2d',
           'assemble_rhs_2d',
           'assemble_periodic_derivative_1d',
           'assemble_curlcurl_2d'
]

def assemble_periodic_derivative_1d(high, low):
    """Coefficient derivative S^p -> S^(p-1), with periodic DOFs folded.

    (D c)_j = p / (T[j+p+1] - T[j+1]) * (c[j+1] - c[j]).
    The degree-reduced space must have the trimmed knot vector.
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    if not high.periodic or not low.periodic:
        raise ValueError('Derivative spaces must be periodic')
    if low.degree != high.degree-1 or not np.array_equal(high.knots[1:-1], low.knots):
        raise ValueError('Derivative space must use the trimmed knot vector')
    n, p = high.nbasis-high.degree, high.degree
    j = np.arange(n)
    scale = p/(high.knots[j+p+1]-high.knots[j+1])
    return coo_matrix((np.r_[-scale, scale],
                       (np.r_[j, j], np.r_[j, (j+1) % n])), shape=(n, n)).tocsr()


def assemble_curlcurl_2d(mass_high, mass_low, derivatives, epsilon=1.0, mu=1.0):
    """Build periodic Cartesian H(curl) matrices in C-order.

    Inputs are folded 1D sparse masses for S^p and S^(p-1), and
    their coefficient derivatives (one per direction). Return
    (gradient, curl, mass, curl_mass, stiffness).
    """
    from scipy.sparse import block_diag, eye, hstack, kron, vstack

    mx_high, my_high = mass_high
    mx_low, my_low = mass_low
    dx, dy = derivatives
    ix, iy = (eye(d.shape[1], format='csr') for d in derivatives)
    # u_x in S^(px-1,py); u_y in S^(px,py-1).
    gradient = vstack((kron(dx, iy), kron(ix, dy)), format='csr')
    curl = hstack((-kron(ix, dy), kron(dx, iy)), format='csr')
    mass = epsilon*block_diag((kron(mx_low, my_high), kron(mx_high, my_low)), format='csr')
    curl_mass = kron(mx_low, my_low, format='csr')/mu
    stiffness = (curl.T @ curl_mass @ curl).tocsr()
    return gradient, curl, mass, curl_mass, stiffness

#==============================================================================
def assemble_stiffness_1d(nelements, degree, spans, basis, weights, points, matrix):
    """
    assembling the stiffness matrix using stencil forms
    """

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points

    k1 = weights.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                j1 = i_span_1 - p1 + jl_1

                v = 0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[ie1, il_1, 0, g1]
                    bi_x = basis_1[ie1, il_1, 1, g1]

                    bj_0 = basis_1[ie1, jl_1, 0, g1]
                    bj_x = basis_1[ie1, jl_1, 1, g1]

                    wvol = weights_1[ie1, g1]

                    v += (bi_x * bj_x) * wvol

                matrix[i1, j1-i1]  += v
    # ...

    return matrix

#==============================================================================
def assemble_rhs_1d(f, nelements, degree, spans, basis, weights, points, rhs):
    """
    Assembly procedure for the rhs
    """

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points

    k1 = weights.shape[1]
    # ...

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            i1 = i_span_1 - p1 + il_1

            v = 0.0
            for g1 in range(0, k1):
                bi_0 = basis_1[ie1, il_1, 0, g1]
                bi_x = basis_1[ie1, il_1, 1, g1]

                x1    = points_1[ie1, g1]
                wvol  = weights_1[ie1, g1]

                v += bi_0 * f(x1) * wvol

            rhs[i1] += v
    # ...

    return rhs

#==============================================================================
def assemble_stiffness_2d(nelements, degree, spans, basis, weights, points, matrix):
    """
    assembling the stiffness matrix using stencil forms
    """

    # ... sizes
    ne1,ne2              = nelements
    p1,p2                = degree
    spans_1, spans_2     = spans
    basis_1, basis_2     = basis
    weights_1, weights_2 = weights
    points_1, points_2   = points

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_0 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v += (bi_x * bj_x + bi_y * bj_y) * wvol

                            matrix[i1, i2, j1-i1, j2-i2]  += v
    # ...

    return matrix

#==============================================================================
def assemble_rhs_2d(f, nelements, degree, spans, basis, weights, points, rhs):
    """
    Assembly procedure for the rhs
    """

    # ... sizes
    ne1,ne2              = nelements
    p1,p2                = degree
    spans_1, spans_2     = spans
    basis_1, basis_2     = basis
    weights_1, weights_2 = weights
    points_1, points_2   = points

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                            x1    = points_1[ie1, g1]
                            x2    = points_2[ie2, g2]
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            v += bi_0 * f(x1,x2) * wvol

                    rhs[i1,i2] += v
    # ...

    # ...
    return rhs
    # ...