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

    # ...
    arr_J_mat11 = zeros((k1,k2))
    arr_J_mat22 = zeros((k1,k2))
    arr_J_mat12 = zeros((k1,k2))
    arr_J_mat21 = zeros((k1,k2))
    Jac_mat     = zeros((k1,k2))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 = lcoeffs_m1[il_1,il_2]
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 = lcoeffs_m2[il_1,il_2]
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                    # ...
                    arr_J_mat11[g1,g2] = F2y
                    arr_J_mat22[g1,g2] = F1x
                    arr_J_mat12[g1,g2] = F1y
                    arr_J_mat21[g1,g2] = F2x

                    Jac_mat[g1,g2] = abs(F1x*F2y-F1y*F2x)

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

                                    bi_x  = arr_J_mat11[g1,g2] * bi_x1 - arr_J_mat21[g1,g2] * bi_x2
                                    bi_y  = arr_J_mat22[g1,g2] * bi_x2 - arr_J_mat12[g1,g2] * bi_x1

                                    bj_x  = arr_J_mat11[g1,g2] * bj_x1 - arr_J_mat21[g1,g2] * bj_x2 
                                    bj_y  = arr_J_mat22[g1,g2] * bj_x2 - arr_J_mat12[g1,g2] * bj_x1 


                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    += (bi_x * bj_x + bi_y * bj_y ) * wvol / Jac_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
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
                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = -1* bi_0 * bx_left
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = -1* bj_0 * bx_left
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_1          = -1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = -1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_2[ie2, g2]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)

                    matrix[p1, p2+i2, p1, p2+j2-i2]  += v
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

                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = bi_0 * bx_right
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = bj_0 * bx_right
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_1          = +1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = +1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ...
                        v    += normalS * (comp_1 * bj_0 + bi_0 * comp_2)  * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)

                    matrix[i_span_1+p1, p2+i2, p1, p2+j2-i2]  += v
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
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = -1* bi_0 * by_left
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        bj_x  = basis_1[ie1, jl_1, 1, g1]
                        bj_y  = -1* bj_0 * by_left
                        # ...
                        comp_1          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol            = weights_1[ie1, g1]
                        # ...
                        v              +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)

                    matrix[p1+i1, p2, p1+j1-i1, p2]  += v
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
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = bi_0*by_right
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        bj_x  = basis_1[ie1, jl_1, 1, g1]
                        bj_y  = bj_0*by_right
                        # ...
                        comp_1          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0 + bi_0*comp_2) * wvol + Kappa * bi_0 * bj_0 * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)

                    matrix[p1+i1, i_span_2+p2, p1+j1-i1, p2]  += v
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
        # ...
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

                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        bj_x  = -1* bj_0 * bx_left
                        bj_y  = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_2          = +1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ... 0. 
                        v    += normalS* ( bi_0*comp_2) * wvol- Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2+ F_2y[g2]**2)

                    matrix[p1+spans_1[ne1-1], p2+i2, p1-spans_1[ne1-1], p2+j2-i2]  += v

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

                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = bi_0 * bx_right
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        # ...
                        comp_1          = -1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_2[ie2, g2]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0)  * wvol
                    matrix[p1+spans_1[ne1-1], p2+i2, p1-spans_1[ne1-1], p2+j2-i2]  += v
    elif interface_nb == 2:
        bx_left  = p1/(knots_1[p1+1]-knots_1[0])*omega_1[1]/omega_1[0]
        bx_right = p1/(knots_1[ne1+2*p1]-knots_1[ne1+p1-1])*omega_1[ne1+p1-2]/omega_1[ne1+p1-1]
        # ...
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

                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = -1.*bi_0 * bx_left
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        # ...
                        comp_1          = +1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0)  * wvol - Kappa * bi_0 * bj_0 * wvol * sqrt(F_1y[g2]**2+ F_2y[g2]**2)

                    matrix[p1, p2+i2, p1+spans_1[ne1-1], p2+j2-i2]  += v

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

                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0    = basis_2[ie2, il_2, 0, g2]
                        #...
                        bj_0    = basis_2[ie2, jl_2, 0, g2]
                        bj_x    = bj_0 * bx_right
                        bj_y    = basis_2[ie2, jl_2, 1, g2]
                        # ...
                        comp_2  = -1 * ( F_2y[g2]*bj_x - F_2x[g2]*bj_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2 += +1 * (-F_1y[g2]*bj_x + F_1x[g2]*bj_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol    = weights_2[ie2, g2]
                        # ...
                        v      +=  normalS * ( bi_0*comp_2) * wvol
                    matrix[p1, p2+i2, p1+spans_1[ne1-1], p2+j2-i2]  += v

    if interface_nb == 3:
        by_left  =  p2*(1/(knots_2[p2+1]-knots_2[0]))*(omega_2[1]/omega_2[0])
        by_right =  p2*(1/(knots_2[ne2+2*p2]-knots_2[ne2+p2-1]))*(omega_2[ne2+p2-2]/omega_2[ne2+p2-1])
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
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        # bi_x  = basis_1[ie1, il_1, 1, g1]
                        # bi_y  = -1*bi_0*by_left
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        # bj_x  = basis_1[ie1, jl_1, 1, g1]
                        # bj_y  = -1*bj_0*by_left
                        # ...
                        # comp_1          = -1 * ( F_2y[g1]*bi_x - F_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        # comp_1         += +1 * (-F_1y[g1]*bi_x + F_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        # comp_2          = +1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        # comp_2         += -1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ... 0.* ( bi_0*comp_2) * wvol
                        v    +=  - Kappa * bi_0 * bj_0 * wvol*sqrt(F1_1x[g1]**2+ F1_2x[g1]**2)

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2] += v
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
                F_1x[g1]  = F1x
                F_2x[g1]  = F2x
                F_1y[g1]  = F1y
                F_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = bi_0*by_right
                        #...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        # bj_x  = basis_1[ie1, jl_1, 1, g1]
                        # bj_y  = bj_0*by_right
                        # # ...
                        comp_1          = -1 * ( F_2y[g1]*bi_x - F_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g1]*bi_x + F_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        # comp_2          = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        # comp_2         += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * (comp_1 * bj_0)  * wvol
                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2]  += v
    elif interface_nb == 4:
        by_left  =  p2*(1/(knots_2[p2+1]-knots_2[0]))*(omega_2[1]/omega_2[0])
        by_right =  p2*(1/(knots_2[ne2+2*p2]-knots_2[ne2+p2-1]))*(omega_2[ne2+p2-2]/omega_2[ne2+p2-1])
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
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = -1*bi_0*by_left
                        # ...
                        bj_0  = basis_1[ie1, jl_1, 0, g1]
                        # bj_x  = basis_1[ie1, jl_1, 1, g1]
                        # bj_y  = -1*bj_0*by_left
                        # ...
                        comp_1          = +1 * ( F_2y[g1]*bi_x - F_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g1]*bi_x + F_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        # comp_2          = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        # comp_2         += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * ( bj_0*comp_1) * wvol - Kappa * bi_0 * bj_0 * wvol*sqrt(F1_1x[g1]**2+ F1_2x[g1]**2)

                    matrix[p1+i1, p2, p1+j1-i1, p2] += v
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
                F_1x[g1]  = F1x
                F_2x[g1]  = F2x
                F_1y[g1]  = F1y
                F_2y[g1]  = F2y
                # ....
                J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p1 + jl_1
                    v  = 0.0
                    for g1 in range(0, k1):
                        bi_0    = basis_1[ie1, il_1, 0, g1]
                        # bi_x  = basis_1[ie1, il_1, 1, g1]
                        # bi_y  = bi_0*by_right
                        #...
                        bj_0    = basis_1[ie1, jl_1, 0, g1]
                        bj_x    = basis_1[ie1, jl_1, 1, g1]
                        bj_y    = bj_0*by_right
                        # # ...
                        # comp_1 = -1 * ( F_2y[g1]*bi_x - F_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        # comp_1+= +1 * (-F_1y[g1]*bi_x + F_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2  = -1 * ( F1_2y[g1]*bj_x - F1_2x[g1]*bj_y)/J_mat1[g1] * F1_2x[g1]#/sqrt(F1y**2+ F2y**2)
                        comp_2 += +1 * (-F1_1y[g1]*bj_x + F1_1x[g1]*bj_y)/J_mat1[g1] * F1_1x[g1]#/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol    = weights_1[ie1, g1]
                        # ...
                        v      +=  normalS * (comp_2 * bi_0)  * wvol
                    matrix[p1+i1, p2, p1+j1-i1, p2]  += v
    # ...