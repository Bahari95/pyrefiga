__all__ = ['assemble_matrix__un_ex01',
           'assemble_vector_un_ex01',
           'assemble_norm_ex02']

#==============================================================================
#---2 : In Stiffness mesh Matrix for laplacian : FE space is different from mapping space
#==============================================================================
def assemble_matrix_un_ex01(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]',PrStPar:'int', matrix:'float64[:,:,:,:]'):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #.. 

    # ...
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))

    # ...
    arr_J_mat0 = zeros((k1,k2))
    arr_J_mat1 = zeros((k1,k2))
    arr_J_mat2 = zeros((k1,k2))
    arr_J_mat3 = zeros((k1,k2))
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
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x

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

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x  = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y  = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x  = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y  = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 


                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    += (1-PrStPar)* bi_0 * bj_0 * wvol* J_mat[g1, g2] + PrStPar * (bi_x * bj_x + bi_y * bj_y ) * wvol / J_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================
# rhs in uniform mesh norm : FE space is different from mapping space
def assemble_vector_un_ex01(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]', x0:'float', y0:'float', rhs:'float64[:,:]'):

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
                    #.. Test 1: solution at 
                    f = 0.
                    if x <= 0.5:
                        f = 1.

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
# norm in uniform mesh norm
def assemble_norm_un_ex01(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int',
                            spans_1:'int[:]', spans_2:'int[:]',spans_3:'int[:,:]', spans_4:'int[:,:]',
                            basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]',  basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]',
                            weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', 
                            vector_m1:'float64[:,:]', vector_m2:'float64[:,:]', vector_u:'float64[:,:]', x0:'float', y0:'float', t:'float', rhs:'float64[:,:]'):

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

    #.. circle

    # ...
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
    lvalues_u  = zeros((k1, k2))

    error_l2   = 0.
    error_H1   = 0.
    integr_L1e = 0.
    integr_L1a = 0.
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

            wa = 0.0
            we = 0.0
            w  = 0.0
            v  = 0.0
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
                    # f0     = 10.*exp(-100.*((x-x0)**2+(y-y0)**2))
                    # f      = 10.*exp(-100.*((x*cos(t)+y*sin(t)-x0)**2+(y*cos(t)-x*sin(t)-y0)**2))
                    # fx     = -200.*(cos(t)*(x*cos(t)+y*sin(t)-x0)-sin(t)*(y*cos(t)-x*sin(t)-y0))*f
                    # fy     = -200.*(sin(t)*(x*cos(t)+y*sin(t)-x0)+cos(t)*(y*cos(t)-x*sin(t)-y0))*f
                    # ...
                    f0  =   exp(-( x**2 + y**2-0.6)**2/0.01)
                    f   =   exp(t -( x**2 + y**2-0.6)**2/0.01)
                    fx  =  -4 * x/0.01 * ( x**2 + y**2-0.6) * f
                    fy  =  -4 * y/0.01 * ( x**2 + y**2-0.6) * f
                    # ...
                    uhx   = (F2y*sx-F2x*sy)/det_J
                    uhy   = (F1x*sy-F1y*sx)/det_J

                    w    += ((uhx-fx)**2 +(uhy-fy)**2)* wvol * det_J
                    v    += (uh-f)**2 * wvol * det_J
                    # ...
                    wa    += (uh-f0) * wvol*det_J
                    we    += f  * wvol*det_J
            error_H1       += w
            error_l2       += v
            integr_L1a     += wa
            integr_L1e     += we
    rhs[p1,p2]   = sqrt(error_l2)
    rhs[p1,p2+1] = sqrt(error_H1)
    rhs[p1,p2+2] = abs(integr_L1a)
    rhs[p1,p2+3] = integr_L1e
    #...