__all__ = ['assemble_stiffness_1d',
           'assemble_rhs_1d',
           'assemble_stiffness_2d',
           'assemble_rhs_2d'
]

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
__all__ = ['assemble_matrix_nitsche_ex00',
           'assemble_matrix_nitsche_ex01']
#------------------------------------------------------------------------------
#... Nitsche's method for assembling the diagonal matrices : from paper https://hal.science/hal-01338133/document
#------------------------------------------------------------------------------
def assemble_matrix_nitsche_ex00(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]',
    vector_m1: 'float[:,:]', vector_m2: 'float[:,:]',
    omega_1: 'float[:]', omega_2: 'float[:]',
    interface_nb: 'int', Kappa: 'float',
    normalS: 'float', matrix: 'float[:,:,:,:]'
):
    #..assemble  solution times noraml(test fuction)
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_m1 = zeros((p1+1,p2+1))
    lcoeffs_m2 = zeros((p1+1,p2+1))
    #......................................
    # .. Nitsches method
    #......................................
    F_1x       = zeros(k2)
    F_2x       = zeros(k2)
    F_1y       = zeros(k2)
    F_2y       = zeros(k2)
    J_mat2     = zeros(k2)
    # ...
    F1_1x      = zeros(k1)
    F1_2x      = zeros(k1)
    F1_1y      = zeros(k1)
    F1_2y      = zeros(k1)
    J_mat1     = zeros(k1)
    # ... build matrices
    if interface_nb == 1:
        bx_left = p1/(knots_1[p1+1]-knots_1[0])*(omega_1[1]/omega_1[0])
        #... Assemble the boundary condition for Nitsche (x=left)
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0     = basis_2[ie2,il_2,0,g2]
                    bj_y     = basis_2[ie2,il_2,1,g2]
                    coeff_m1 = lcoeffs_m1[0, il_2]
                    coeff_m2 = lcoeffs_m2[0, il_2]
                    coeff_m11= lcoeffs_m1[1, il_2]
                    coeff_m22= lcoeffs_m2[1, il_2]
                    
                    F1x     +=  (coeff_m11-coeff_m1) * bj_0 * bx_left
                    F2x     +=  (coeff_m22-coeff_m2) * bj_0 * bx_left
                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    vjp  = 0.0 # penultimate derivative
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = -1* bi_0 * bx_left
                        bi_px =  1* bi_0 * bx_left
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = -1* bj_0 * bx_left
                        bj_px =  1* bj_0 * bx_left
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_1          = +1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = +1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = +1 * ( F_2y[g2]*bj_px - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_4         += -1 * (-F_1y[g2]*bj_px + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol    = weights_2[ie2, g2]
                        # ...
                        v      +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)
                        vip    +=  normalS * (comp_3 * bj_0) * wvol
                        vjp    +=  normalS * (bi_0*comp_4) * wvol

                    matrix[p1, p2+i2, p1, p2+j2-i2]      += v
                    matrix[p1+1, p2+i2, p1-1, p2+j2-i2]  += vip
                    matrix[p1, p2+i2, p1+1, p2+j2-i2]    += vjp
    elif interface_nb == 2:
        bx_right = p1/(knots_1[ne1+2*p1]-knots_1[ne1+p1-1])*(omega_1[ne1+p1-2]/omega_1[ne1+p1-1])
        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0      = basis_2[ie2,il_2,0,g2]
                    bj_y      = basis_2[ie2,il_2,1,g2]

                    coeff_m1  = lcoeffs_m1[p1, il_2]
                    coeff_m2  = lcoeffs_m2[p1, il_2]                    
                    coeff_m10 = lcoeffs_m1[p1-1, il_2]
                    coeff_m20 = lcoeffs_m2[p1-1, il_2]

                    F1y      +=  coeff_m1 * bj_y
                    F2y      +=  coeff_m2 * bj_y
                    F1x      +=  (coeff_m1-coeff_m10) * bj_0 * bx_right
                    F2x      +=  (coeff_m2-coeff_m20) * bj_0 * bx_right
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)

            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):

                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v   = 0.0
                    vpi = 0.0 # penultimate derivative
                    vpj = 0.0 # penultimate derivative
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = bi_0 * bx_right
                        bi_px = -1.*bi_0 * bx_right
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = bj_0 * bx_right
                        bj_px = -1.*bj_0 * bx_right
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_1          = -1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = -1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = -1 * ( F_2y[g2]*bj_px - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_4         += +1 * (-F_1y[g2]*bj_px + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ... - 0.5*u1*v1_n - 0.5*u1_n*v1
                        v    += normalS * (bj_0 * comp_1 + comp_2 * bi_0)  * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)
                        vpi  += normalS * (bj_0 * comp_3)  * wvol 
                        vpj  += normalS * ( comp_4 * bi_0)  * wvol

                    matrix[i_span_1+p1, p2+i2, p1, p2+j2-i2]      += v
                    matrix[i_span_1+p1-1, p2+i2, p1+1, p2+j2-i2]  += vpi
                    matrix[i_span_1+p1, p2+i2, p1-1, p2+j2-i2]    += vpj
    if interface_nb == 3:
        by_left = p2/(knots_2[p2+1]-knots_2[0])*omega_2[1]/omega_2[0]
        #... Assemble the boundary condition for Nitsche (y=left)
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, 0]
                    coeff_m2 = lcoeffs_m2[il_1, 0]
                    coeff_m11= lcoeffs_m1[il_1, 1]
                    coeff_m22= lcoeffs_m2[il_1, 1]
                    
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                    F1y     +=  (coeff_m11-coeff_m1) * bj_0 * by_left
                    F2y     +=  (coeff_m22-coeff_m2) * bj_0 * by_left
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    vjp  = 0.0 # penultimate derivative
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = -1* bi_0 * by_left
                        bi_py =  1* bi_0 * by_left
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        bj_x  = basis_1[ie1, jl_1, 1, g1]
                        bj_y  = -1* bj_0 * by_left
                        bj_py =  1* bj_0 * by_left
                        # ...
                        comp_1          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_4         += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol            = weights_1[ie1, g1]
                        # ...
                        v              +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)
                        vip            +=  normalS * (comp_3 * bj_0) * wvol
                        vjp            +=  normalS * (bi_0*comp_4) * wvol

                    matrix[p1+i1, p2, p1+j1-i1, p2]     += v
                    matrix[p1+i1, p2+1, p1+j1-i1, p2-1] += vip
                    matrix[p1+i1, p2, p1+j1-i1, p2+1]   += vjp
    elif interface_nb == 4:
        by_right = p2/(knots_2[ne2+2*p2]-knots_2[ne2+p2-1])*(omega_2[ne2+p2-2]/omega_2[ne2+p2-1])
        #... Assemble the boundary condition for Nitsche (y=right)
        ie2      = ne2 -1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, p2]
                    coeff_m2 = lcoeffs_m2[il_1, p2]
                    coeff_m10= lcoeffs_m1[il_1, p2-1]
                    coeff_m20= lcoeffs_m2[il_1, p2-1]
                    
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                    F1y     +=  (coeff_m1-coeff_m10) * bj_0*by_right
                    F2y     +=  (coeff_m2-coeff_m20) * bj_0*by_right
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    vjp  = 0.0 # penultimate derivative
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = bi_0*by_right
                        bi_py  = -1.*bi_0*by_right
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        bj_x  = basis_1[ie1, jl_1, 1, g1]
                        bj_y  = bj_0*by_right
                        bj_py  = -1.*bj_0*by_right
                        # ...
                        comp_1          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_4         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)
                        vip  +=  normalS * (comp_3 * bj_0) * wvol
                        vjp  +=  normalS * (bi_0 * comp_4) * wvol
                    matrix[p1+i1, i_span_2+p2, p1+j1-i1, p2]     += v
                    matrix[p1+i1, i_span_2+p2-1, p1+j1-i1, p2+1] += vip
                    matrix[p1+i1, i_span_2+p2, p1+j1-i1, p2-1]   += vjp
    # ...
#------------------------------------------------------------------------------
#... Nitsche's method for assembling the matrix : normal derivative
#------------------------------------------------------------------------------
def assemble_matrix_nitsche_ex02(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]',
    vector_m1: 'float[:,:]', vector_m2: 'float[:,:]',
    vector_m3: 'float[:,:]', vector_m4: 'float[:,:]',
    omega_1: 'float[:]', omega_2: 'float[:]',
    interface_nb: 'int', Kappa: 'float',
    normalS:'float', matrix: 'float[:,:,:,:]'
):
    #..assemble  solution times noraml(test fuction)
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1          = weights_1.shape[1]
    k2          = weights_2.shape[1]
    # ..
    lcoeffs_m1  = zeros((p1+1,p2+1))
    lcoeffs_m2  = zeros((p1+1,p2+1))
    F_1x        = zeros(k2)
    F_2x        = zeros(k2)
    F_1y        = zeros(k2)
    F_2y        = zeros(k2)
    J_mat2      = zeros(k2)
    # ...
    F1_1x       = zeros(k1)
    F1_2x       = zeros(k1)
    F1_1y       = zeros(k1)
    F1_2y       = zeros(k1)
    J_mat1      = zeros(k1)
    # ... build matrices
    if interface_nb == 1 :
        bx_left  = p1/(knots_1[p1+1]-knots_1[0])*omega_1[1]/omega_1[0]
        bx_right = p1/(knots_1[ne1+2*p1]-knots_1[ne1+p1-1])*omega_1[ne1+p1-2]/omega_1[ne1+p1-1]
        # ... v1*u2
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_m1[ : , : ] = vector_m3[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m4[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0     = basis_2[ie2,il_2,0,g2]
                    bj_y     = basis_2[ie2,il_2,1,g2]
                    coeff_m1 = lcoeffs_m1[0, il_2]
                    coeff_m2 = lcoeffs_m2[0, il_2]
                    coeff_m11= lcoeffs_m1[1, il_2]
                    coeff_m22= lcoeffs_m2[1, il_2]
                    
                    F1x     +=  (coeff_m11-coeff_m1) * bj_0 * bx_left
                    F2x     +=  (coeff_m22-coeff_m2) * bj_0 * bx_left
                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v   = 0.0
                    vjp = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = -1* bj_0 * bx_left
                        bj_px =  1* bj_0 * bx_left
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_2          = -1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = -1 * ( F_2y[g2]*bj_px - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_4         += +1 * (-F_1y[g2]*bj_px + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ...  -0.5*u2_n*v1-k*u2*v1
                        v    +=  normalS* ( comp_2 * bi_0) * wvol - Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2+ F_2y[g2]**2)
                        vjp  +=  normalS* ( comp_4 * bi_0) * wvol

                    matrix[p1+spans_1[ne1-1], p2+i2, p1+0, p2+j2]  += v
                    matrix[p1+spans_1[ne1-1], p2+i2, p1+1, p2+j2]  += vjp
        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0      = basis_2[ie2,il_2,0,g2]
                    bj_y      = basis_2[ie2,il_2,1,g2]

                    coeff_m1  = lcoeffs_m1[p1, il_2]
                    coeff_m2  = lcoeffs_m2[p1, il_2]                    
                    coeff_m10 = lcoeffs_m1[p1-1, il_2]
                    coeff_m20 = lcoeffs_m2[p1-1, il_2]

                    F1y      +=  coeff_m1 * bj_y
                    F2y      +=  coeff_m2 * bj_y
                    F1x      +=  (coeff_m1-coeff_m10) * bj_0 * bx_right
                    F2x      +=  (coeff_m2-coeff_m20) * bj_0 * bx_right
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)

            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v   = 0.0
                    vip = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = bi_0 * bx_right
                        bi_px = -1.*bi_0 * bx_right
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        # ...
                        comp_1          = +1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_2[ie2, g2]
                        # ... 0.5*u2*v1_n
                        v    +=  normalS * (bj_0 * comp_1)  * wvol
                         # ... 0.5*u2*v1_n
                        vip  +=  normalS * (bj_0 * comp_3)  * wvol
                    matrix[p1+spans_1[ne1-1], p2+i2, p1, p2+j2]   += v
                    matrix[p1+spans_1[ne1-1]-1, p2+i2, p1, p2+j2] += vip

    elif interface_nb == 2:
        bx_left  = p1/(knots_1[p1+1]-knots_1[0])*omega_1[1]/omega_1[0]
        bx_right = p1/(knots_1[ne1+2*p1]-knots_1[ne1+p1-1])*omega_1[ne1+p1-2]/omega_1[ne1+p1-1]
        # ... u1*v2
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_m1[ : , : ] = vector_m3[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m4[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0     = basis_2[ie2,il_2,0,g2]
                    bj_y     = basis_2[ie2,il_2,1,g2]
                    coeff_m1 = lcoeffs_m1[0, il_2]
                    coeff_m2 = lcoeffs_m2[0, il_2]
                    coeff_m11= lcoeffs_m1[1, il_2]
                    coeff_m22= lcoeffs_m2[1, il_2]
                    
                    F1x     +=  (coeff_m11-coeff_m1) * bj_0 * bx_left
                    F2x     +=  (coeff_m22-coeff_m2) * bj_0 * bx_left
                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v   = 0.0
                    vip = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = -1.*bi_0 * bx_left
                        bi_px =  1.*bi_0 * bx_left
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        # ...
                        comp_1          = -1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ... -0.5*u1*v2_n  - k*u1*v2
                        v    +=  normalS * (comp_1 * bj_0)  * wvol - Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2+ F_2y[g2]**2)
                        vip  +=  normalS * (comp_3 * bj_0)  * wvol

                    matrix[p1, p2+i2, p1+spans_1[ne1-1], p2+j2]    += v
                    matrix[p1+1, p2+i2, p1+spans_1[ne1-1], p2+j2]  += vip
                    
        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_0      = basis_2[ie2,il_2,0,g2]
                    bj_y      = basis_2[ie2,il_2,1,g2]

                    coeff_m1  = lcoeffs_m1[p1, il_2]
                    coeff_m2  = lcoeffs_m2[p1, il_2]                    
                    coeff_m10 = lcoeffs_m1[p1-1, il_2]
                    coeff_m20 = lcoeffs_m2[p1-1, il_2]

                    F1y      +=  coeff_m1 * bj_y
                    F2y      +=  coeff_m2 * bj_y
                    F1x      +=  (coeff_m1-coeff_m10) * bj_0 * bx_right
                    F2x      +=  (coeff_m2-coeff_m20) * bj_0 * bx_right
                # ... compute the normal derivative
                F_1x[g2] = F1x
                F_2x[g2] = F2x
                F_1y[g2] = F1y
                F_2y[g2] = F2y
                # ....
                J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)

            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2

                    v   = 0.0
                    vjp = 0.0
                    for g2 in range(0, k2):
                        bi_0    = basis_2[ie2, il_2, 0, g2]
                        #...
                        bj_0    = basis_2[ie2, jl_2, 0, g2]
                        bj_x    = bj_0 * bx_right
                        bj_px   = -1.*bj_0 * bx_right
                        bj_y    = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_2  = +1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2 += -1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4  = +1 * ( F_2y[g2]*bj_px - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_4 += -1 * (-F_1y[g2]*bj_px + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol    = weights_2[ie2, g2]
                        # ... 0.5*u1_n*v2
                        v      +=  normalS * ( comp_2 * bi_0) * wvol
                        vjp    +=  normalS * ( comp_4 * bi_0) * wvol

                    matrix[p1, p2+i2, p1+spans_1[ne1-1], p2+j2]   += v
                    matrix[p1, p2+i2, p1+spans_1[ne1-1]-1, p2+j2] += vjp

    if interface_nb == 3:
        by_left  = p2/(knots_2[p2+1]-knots_2[0])*omega_2[1]/omega_2[0]
        by_right = p2/(knots_2[ne2+2*p2]-knots_2[ne2+p2-1])*omega_2[ne2+p2-2]/omega_2[ne2+p2-1]
        # ...
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m3[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m4[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, 0]
                    coeff_m2 = lcoeffs_m2[il_1, 0]
                    coeff_m11= lcoeffs_m1[il_1, 1]
                    coeff_m22= lcoeffs_m2[il_1, 1]
                    
                    F1y     +=  (coeff_m11-coeff_m1) * bj_0*by_left
                    F2y     +=  (coeff_m22-coeff_m2) * bj_0*by_left
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v   = 0.0
                    vjp = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        bj_x  = basis_1[ie1, jl_1, 1, g1]
                        bj_y  = -1*bj_0*by_left
                        bj_py =  1*bj_0*by_left
                        # ...
                        comp_2          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_py)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        comp_4         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_py)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ... 
                        v    += normalS * ( comp_2 * bi_0) * wvol - Kappa * bi_0 * bj_0 * wvol*sqrt(F1_1x[g1]**2+ F1_2x[g1]**2)
                        vjp  += normalS * ( comp_4 * bi_0) * wvol

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1, p2+1] += vjp

        #... Assemble the boundary condition for Nitsche (x=right)
        ie2      = ne2 -1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, p2]
                    coeff_m2 = lcoeffs_m2[il_1, p2]
                    coeff_m10= lcoeffs_m1[il_1, p2-1]
                    coeff_m20= lcoeffs_m2[il_1, p2-1]
                    
                    F1y     += (coeff_m1-coeff_m10) * bj_0*by_right
                    F2y     += (coeff_m2-coeff_m20) * bj_0*by_right
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v   = 0.0
                    vip = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = bi_0*by_right
                        bi_py = -1.*bi_0*by_right
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        # # ...
                        comp_1          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ... 0.5*u2*v1_n
                        v    +=  normalS * (bj_0 * comp_1)  * wvol
                         # ... 0.5*u2*v1_n
                        vip  +=  normalS * (bj_0 * comp_3)  * wvol

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1]-1, p1+j1, p2] += vip

    elif interface_nb == 4:
        by_left  = p2/(knots_2[p2+1]-knots_2[0])*omega_2[1]/omega_2[0]
        by_right = p2/(knots_2[ne2+2*p2]-knots_2[ne2+p2-1])*omega_2[ne2+p2-2]/omega_2[ne2+p2-1]
        #... Assemble the boundary condition for Nitsche (x=left)
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m3[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m4[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, 0]
                    coeff_m2 = lcoeffs_m2[il_1, 0]
                    coeff_m11= lcoeffs_m1[il_1, 1]
                    coeff_m22= lcoeffs_m2[il_1, 1]
                    
                    F1y     +=  (coeff_m11-coeff_m1) * bj_0*by_left
                    F2y     +=  (coeff_m22-coeff_m2) * bj_0*by_left
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v   = 0.0
                    vip = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = -1*bi_0*by_left
                        bi_py =  1*bi_0*by_left
                        # ...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        # ...
                        comp_1          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0) * wvol - Kappa * bi_0 * bj_0 * wvol*sqrt(F1_1x[g1]**2+ F1_2x[g1]**2)
                        vip  +=  normalS * (comp_3 * bj_0)  * wvol

                    matrix[p1+i1, p2, p1+j1, p2+spans_2[ne2-1]]   += v
                    matrix[p1+i1, p2+1, p1+j1, p2+spans_2[ne2-1]] += vip

        #... Assemble the boundary condition for Nitsche (x=right)
        ie2      = ne2 -1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):

                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_0     = basis_1[ie1,il_1,0,g1]
                    bj_x     = basis_1[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, p2]
                    coeff_m2 = lcoeffs_m2[il_1, p2]
                    coeff_m10= lcoeffs_m1[il_1, p2-1]
                    coeff_m20= lcoeffs_m2[il_1, p2-1]
                    
                    F1y     +=  (coeff_m1-coeff_m10) * bj_0*by_right
                    F2y     +=  (coeff_m2-coeff_m20) * bj_0*by_right
                    F1x     +=  coeff_m1 * bj_x
                    F2x     +=  coeff_m2 * bj_x
                # ... compute the normal derivative
                F1_1x[g1]  = F1x
                F1_2x[g1]  = F2x
                F1_1y[g1]  = F1y
                F1_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1

                    v   = 0.0
                    vjp = 0.0
                    for g1 in range(0, k1):
                        bi_0    = basis_1[ie1, il_1, 0, g1]
                        #...
                        bj_0    = basis_1[ie1, jl_1, 0, g1]
                        bj_x    = basis_1[ie1, jl_1, 1, g1]
                        bj_y    = bj_0*by_right
                        bj_py   = -1.*bj_0*by_right
                        # ...
                        comp_2  = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        comp_2 += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_4  = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_py)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        comp_4 += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_py)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol    = weights_1[ie1, g1]
                        # ... 0.5*u1_n*v2
                        v      +=  normalS * ( comp_2 * bi_0) * wvol
                        vjp    +=  normalS * ( comp_4 * bi_0) * wvol
                    matrix[p1+i1, p2, p1+j1, p2+spans_2[ne2-1]]   += v
                    matrix[p1+i1, p2, p1+j1, p2+spans_2[ne2-1]-1] += vjp
    # ...