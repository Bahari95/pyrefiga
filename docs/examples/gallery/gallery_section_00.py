#==============================================================================
#---2 : In adapted mesh Matrix
def assemble_matrix_mass_ex01(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]', matrix:'float64[:,:,:,:]'):

    # ... sizes
    from numpy import zeros
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #.. 
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))
    # ...
    J_mat      = zeros((k1,k2))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, g2]
                    
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1] 
                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):
                        for il_2 in range(0, p4+1):

                            bj_x     = basis_3[ie1,il_1,1,g1]*basis_4[ie2,il_2,0,g2]
                            bj_y     = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,1,g2]

                            coeff_m1 = lcoeffs_m1[il_1,il_2]
                            F1x     +=  coeff_m1 * bj_x
                            F1y     +=  coeff_m1 * bj_y

                            coeff_m2 = lcoeffs_m2[il_1,il_2]
                            F2x     +=  coeff_m2 * bj_x
                            F2y     +=  coeff_m2 * bj_y
                    # ...
                    J_mat[g1,g2] = abs(F1x*F2y-F1y*F2x)

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
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]

                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    += bi_0 * bj_0 * wvol * J_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================
# rhs in uniform mesh norm : FE space is different from mapping space
def assemble_vector_un_ex02(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]', rhs:'float64[:,:]'):

    from numpy import exp
    from numpy import cos, cosh
    from numpy import sin, sinh
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_m1  = zeros((p3+1,p4+1))
    lcoeffs_m2  = zeros((p3+1,p4+1))
    #..
    lvalues_u   = zeros((k1, k2))
    J_mat       = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, g2]
                    
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1] 
                    # ...
                    x    = 0.0
                    y    = 0.0
                    F1x  = 0.0
                    F1y  = 0.0
                    F2x  = 0.0
                    F2y  = 0.0
                    for il_1 in range(0, p3+1):
                        for il_2 in range(0, p4+1):
                            bj_0     = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                            bj_x     = basis_3[ie1,il_1,1,g1]*basis_4[ie2,il_2,0,g2]
                            bj_y     = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,1,g2]

                            coeff_m1 =  lcoeffs_m1[il_1,il_2]
                            x       +=  coeff_m1 * bj_0
                            F1x     +=  coeff_m1 * bj_x
                            F1y     +=  coeff_m1 * bj_y

                            coeff_m2 =  lcoeffs_m2[il_1,il_2]
                            y       +=  coeff_m2 * bj_0
                            F2x     +=  coeff_m2 * bj_x
                            F2y     +=  coeff_m2 * bj_y
                              
                    J_mat[g1,g2] = abs(F1x*F2y-F1y*F2x)
                    # ... test 0
                    f  = x**2+y**2
                    lvalues_u[g1,g2]   = f 
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            # ...
                            wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]
                            # ...
                            u     = lvalues_u[g1,g2]
                            # ...
                            v    += bi_0 * u * wvol * J_mat[g1,g2]

                    rhs[i1+p1,i2+p2] += v

#=================================================================================
# norm in uniform mesh norm : FE space is different from mapping space
def assemble_norm_un_ex02(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]', vector_u:'float64[:,:]', rhs:'float64[:,:]'):

    from numpy import exp
    from numpy import pi
    from numpy import sin, sinh
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
    lvalues_u  = zeros((k1, k2))

    error_l2 = 0.
    error_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]  = 0.0
            lvalues_ux[ : , : ]  = 0.0
            lvalues_uy[ : , : ]  = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2 = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]   += coeff_u*b1*b2
                            lvalues_ux[g1,g2]  += coeff_u*db1*b2
                            lvalues_uy[g1,g2]  += coeff_u*b1*db2

            w = 0.0
            v = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, g2]
                    
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1] 
                    x   = 0.0
                    y   = 0.0
                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):

                              bj_0     = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                              bj_x     = basis_3[ie1,il_1,1,g1]*basis_4[ie2,il_2,0,g2]
                              bj_y     = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,1,g2]

                              coeff_m1 =  lcoeffs_m1[il_1,il_2]
                              x       +=  coeff_m1 * bj_0
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 =  lcoeffs_m2[il_1,il_2]
                              y       +=  coeff_m2 * bj_0
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                    det_J = abs(F1x*F2y-F1y*F2x)
                    
                    # ...                              
                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                    uh    = lvalues_u[g1,g2]
                    sx    = lvalues_ux[g1,g2]
                    sy    = lvalues_uy[g1,g2]

                    #... TEST 1
                    # f    = sin(2.*pi*x)*sin(2.*pi*y) 
                    # fx   = 2.*pi*cos(2.*pi*x)*sin(2.*pi*y) 
                    # fy   = 2.*pi*sin(2.*pi*x)*cos(2.*pi*y) 
                    #... TEST 2
                    f    = x**2+y**2
                    fx   = x*2.
                    fy   = y*2.
                    # ...
                    uhx   = (F2y*sx-F2x*sy)/det_J
                    uhy   = (F1x*sy-F1y*sx)/det_J

                    w    += ((uhx-fx)**2 +(uhy-fy)**2)* wvol * det_J
                    v    += (uh-f)**2 * wvol * det_J

            error_H1      += w
            error_l2      += v
    rhs[p1,p2]   = sqrt(error_l2)
    rhs[p1,p2+1] = sqrt(error_H1)
    #...