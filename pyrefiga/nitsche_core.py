# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: M. BAHARI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#------------------------------------------------------------------------------
# ... Nitsche's method for assembling the matrices : from paper https://hal.science/hal-01338133/document
# ... case where mapping is in the same spline space as for FE
# diagonal matrix with respect to given patch
#------------------------------------------------------------------------------
def assemble_matrix_diagnitsche(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]',
    vector_m1: 'float[:,:]', vector_m2: 'float[:,:]',
    omega_1: 'float[:]', omega_2: 'float[:]',
    interfaces: 'int[:]', Kappa: 'float',
    normalS: 'float', matrix: 'float[:,:,:,:]'
):
    #..assemble  solution times noraml(test fuction)
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #...
    nb_interfaces = interfaces.shape[0]
    n_omega1    = omega_1.shape[0]-1
    n_omega2    = omega_2.shape[0]-1
    n_knots1    = knots_1.shape[0]-1
    n_knots2    = knots_2.shape[0]-1
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
    for j in range(0, nb_interfaces):
        interface_nb = interfaces[j]
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
        if interface_nb == 2:
            bx_right = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])*(omega_1[n_omega1-1]/omega_1[n_omega1])
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
        if interface_nb == 4:
            by_right = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])*(omega_2[n_omega2-1]/omega_2[n_omega2])
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
#... Nitsche's method for assembling the under diagonal matrix: with respects to
#... the off-diagonal terms of the stiffness matrix
#------------------------------------------------------------------------------
def assemble_matrix_offdiagnitsche(
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
    n_omega1    = omega_1.shape[0]-1
    n_omega2    = omega_2.shape[0]-1
    n_knots1    = knots_1.shape[0]-1
    n_knots2    = knots_2.shape[0]-1
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
        bx_right = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])*omega_1[n_omega1-1]/omega_1[n_omega1]
        # ... v1*u2
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

                    matrix[p1+spans_1[ne1-1], p2+i2, p1, p2+j2-i2]  += v
                    matrix[p1+spans_1[ne1-1], p2+i2, p1-1, p2+j2-i2]  += vjp

        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
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
                    matrix[p1+spans_1[ne1-1], p2+i2, p1, p2+j2-i2]   += v
                    matrix[p1+spans_1[ne1-1]-1, p2+i2, p1+1, p2+j2-i2] += vip

    elif interface_nb == 2:
        bx_left  = p1/(knots_1[p1+1]-knots_1[0])*omega_1[1]/omega_1[0]
        bx_right = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])*omega_1[n_omega1-1]/omega_1[n_omega1]
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

                    matrix[p1, p2+i2, p1, p2+j2-i2]    += v
                    matrix[p1+1, p2+i2, p1-1, p2+j2-i2]  += vip

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

                    matrix[p1, p2+i2, p1, p2+j2-i2]   += v
                    matrix[p1, p2+i2, p1+1, p2+j2-i2] += vjp

    elif interface_nb == 3:
        by_left  = p2/(knots_2[p2+1]-knots_2[0])*omega_2[1]/omega_2[0]
        by_right = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])*omega_2[n_omega2-1]/omega_2[n_omega2]
        # ...
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

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2-1] += vjp

        #... Assemble the boundary condition for Nitsche (x=right)
        ie2      = ne2 -1
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

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1]-1, p1+j1-i1, p2+1] += vip

    elif interface_nb == 4:
        by_left  = p2/(knots_2[p2+1]-knots_2[0])*omega_2[1]/omega_2[0]
        by_right = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])*omega_2[n_omega2-1]/omega_2[n_omega2]
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

                    matrix[p1+i1, p2, p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+1, p1+j1-i1, p2-1] += vip

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
                    matrix[p1+i1, p2, p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2, p1+j1-i1, p2+1] += vjp
    # ...


#-------------------------------------------------------------------------------------------------------
# ... Nitsche's method for assembling the matrices : from paper https://hal.science/hal-01338133/document
# ===============================================================
# different spaces for FE and mapping can be implemented here
# ===============================================================
# diagonal matrix with respect to given patch
#--------------------------------------------------------------------------------------------------------
def assemble_matrix_DiffSpacediagnitsche(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int', p3: 'int', p4: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',spans_3: 'int[:,:]', spans_4: 'int[:,:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',basis_3: 'float[:,:,:,:]', basis_4: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]', knots_3: 'float[:]', knots_4: 'float[:]',
    vector_m1: 'float[:,:]', vector_m2: 'float[:,:]',
    omega_1: 'float[:]', omega_2: 'float[:]',
    interfaces: 'int[:]', Kappa: 'float',
    normalS: 'float', matrix: 'float[:,:,:,:]'
):
    #..assemble  solution times noraml(test fuction)
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #...
    nb_interfaces = interfaces.shape[0]
    n_omega1    = omega_1.shape[0]-1
    n_omega2    = omega_2.shape[0]-1
    n_knots1    = knots_1.shape[0]-1
    n_knots2    = knots_2.shape[0]-1
    n_knots3    = knots_3.shape[0]-1
    n_knots4    = knots_4.shape[0]-1
    # ...
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))

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
    for j in range(0, nb_interfaces):
        interface_nb = interfaces[j]
        if interface_nb == 1:
            bx_left   = p3/(knots_3[p3+1]-knots_3[0])*(omega_1[1]/omega_1[0])
            bx_leftFE = p1/(knots_1[p1+1]-knots_1[0])
            #... Assemble the boundary condition for Nitsche (x=left)
            ie1      = 0
            i_span_1 = spans_1[ie1]
            for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]

                for g2 in range(0, k2):

                    i_span_3 = spans_3[ie1, 0]
                    i_span_4 = spans_4[ie2, g2]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_2 in range(0, p4+1):

                        bj_0     = basis_4[ie2,il_2,0,g2]
                        bj_y     = basis_4[ie2,il_2,1,g2]
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
                            bi_x  = -1* bi_0 * bx_leftFE
                            bi_px =  1* bi_0 * bx_leftFE
                            bi_y  = basis_2[ie2, il_2, 1, g2]
                            #...
                            bj_0  = basis_2[ie2, jl_2, 0, g2]
                            bj_x  = -1* bj_0 * bx_leftFE
                            bj_px =  1* bj_0 * bx_leftFE
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
        if interface_nb == 2:
            bx_right   = p3/(knots_3[n_knots3]-knots_3[n_knots3-p3-1])*(omega_1[n_omega1-1]/omega_1[n_omega1])
            bx_rightFE = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])
            #... Assemble the boundary condition for Nitsche (x=right)
            ie1      = ne1 -1
            i_span_1 = spans_1[ie1]
            for ie2 in range(0, ne2):         
                i_span_2 = spans_2[ie2]
                
                for g2 in range(0, k2):
                    i_span_3 = spans_3[ie1, k1-1]
                    i_span_4 = spans_4[ie2, g2]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_2 in range(0, p4+1):

                        bj_0      = basis_4[ie2,il_2,0,g2]
                        bj_y      = basis_4[ie2,il_2,1,g2]

                        coeff_m1  = lcoeffs_m1[p3, il_2]
                        coeff_m2  = lcoeffs_m2[p3, il_2]                    
                        coeff_m10 = lcoeffs_m1[p3-1, il_2]
                        coeff_m20 = lcoeffs_m2[p3-1, il_2]

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
                            bi_x  = bi_0 * bx_rightFE
                            bi_px = -1.*bi_0 * bx_rightFE
                            bi_y  = basis_2[ie2, il_2, 1, g2]
                            #...
                            bj_0  = basis_2[ie2, jl_2, 0, g2]
                            bj_x  = bj_0 * bx_rightFE
                            bj_px = -1.*bj_0 * bx_rightFE
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
            by_left   = p4/(knots_4[p4+1]-knots_4[0])*omega_2[1]/omega_2[0]
            by_leftFE = p2/(knots_2[p2+1]-knots_2[0])
            #... Assemble the boundary condition for Nitsche (y=left)
            ie2      = 0
            i_span_2 = spans_2[ie2]
            for ie1 in range(0, ne1):
                i_span_1 = spans_1[ie1]

                for g1 in range(0, k1):
                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, 0]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):

                        bj_0     = basis_3[ie1,il_1,0,g1]
                        bj_x     = basis_3[ie1,il_1,1,g1]
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
                            bi_y  = -1* bi_0 * by_leftFE
                            bi_py =  1* bi_0 * by_leftFE
                            #...
                            bj_0  = basis_1[ie1, jl_1, 0, g1]
                            bj_x  = basis_1[ie1, jl_1, 1, g1]
                            bj_y  = -1* bj_0 * by_leftFE
                            bj_py =  1* bj_0 * by_leftFE
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
        if interface_nb == 4:
            by_right   = p4/(knots_4[n_knots4]-knots_4[n_knots4-p4-1])*(omega_2[n_omega2-1]/omega_2[n_omega2])
            by_rightFE = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])
            #... Assemble the boundary condition for Nitsche (y=right)
            ie2      = ne2 -1
            i_span_2 = spans_2[ie2]
            for ie1 in range(0, ne1):
                i_span_1 = spans_1[ie1]

                for g1 in range(0, k1):
                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, k2-1]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):

                        bj_0     = basis_3[ie1,il_1,0,g1]
                        bj_x     = basis_3[ie1,il_1,1,g1]
                        coeff_m1 = lcoeffs_m1[il_1, p4]
                        coeff_m2 = lcoeffs_m2[il_1, p4]
                        coeff_m10= lcoeffs_m1[il_1, p4-1]
                        coeff_m20= lcoeffs_m2[il_1, p4-1]
                        
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
                            bi_y  = bi_0*by_rightFE
                            bi_py  = -1.*bi_0*by_rightFE
                            #...
                            bj_0  = basis_1[ie1, jl_1, 0, g1]
                            bj_x  = basis_1[ie1, jl_1, 1, g1]
                            bj_y  = bj_0*by_rightFE
                            bj_py  = -1.*bj_0*by_rightFE
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
#... Nitsche's method for assembling the under diagonal matrix: with respects to
#... the off-diagonal terms of the stiffness matrix
#------------------------------------------------------------------------------
def assemble_matrix_DiffSpaceoffdiagnitsche(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int', p3: 'int', p4: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',spans_3: 'int[:,:]', spans_4: 'int[:,:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',basis_3: 'float[:,:,:,:]', basis_4: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]', knots_3: 'float[:]', knots_4: 'float[:]',
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
    n_omega1    = omega_1.shape[0]-1
    n_omega2    = omega_2.shape[0]-1
    n_knots1    = knots_1.shape[0]-1
    n_knots2    = knots_2.shape[0]-1
    n_knots3    = knots_3.shape[0]-1
    n_knots4    = knots_4.shape[0]-1
    # ...
    lcoeffs_m1  = zeros((p3+1,p4+1))
    lcoeffs_m2  = zeros((p3+1,p4+1))
    #...
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
        bx_left  = p3/(knots_3[p3+1]-knots_3[0])*omega_1[1]/omega_1[0]
        bx_right = p3/(knots_3[n_knots3]-knots_3[n_knots3-p3-1])*omega_1[n_omega1-1]/omega_1[n_omega1]
        # ...
        bx_leftFE  = p1/(knots_1[p1+1]-knots_1[0])
        bx_rightFE = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])
        # ... v1*u2
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g2 in range(0, k2):

                i_span_3 = spans_3[ie1, 0]
                i_span_4 = spans_4[ie2, g2]
                lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p4+1):

                    bj_0     = basis_4[ie2,il_2,0,g2]
                    bj_y     = basis_4[ie2,il_2,1,g2]
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
                        bj_x  = -1* bj_0 * bx_leftFE
                        bj_px =  1* bj_0 * bx_leftFE
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

                    matrix[p1+spans_1[ne1-1], p2+i2, p1, p2+j2-i2]  += v
                    matrix[p1+spans_1[ne1-1], p2+i2, p1-1, p2+j2-i2]  += vjp

        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]
            
            for g2 in range(0, k2):
                i_span_3 = spans_3[ie1, k1-1]
                i_span_4 = spans_4[ie2, g2]
                lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p4+1):

                    bj_0      = basis_4[ie2,il_2,0,g2]
                    bj_y      = basis_4[ie2,il_2,1,g2]

                    coeff_m1  = lcoeffs_m1[p3, il_2]
                    coeff_m2  = lcoeffs_m2[p3, il_2]                    
                    coeff_m10 = lcoeffs_m1[p3-1, il_2]
                    coeff_m20 = lcoeffs_m2[p3-1, il_2]

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
                        bi_x  = bi_0 * bx_rightFE
                        bi_px = -1.*bi_0 * bx_rightFE
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
                    matrix[p1+spans_1[ne1-1], p2+i2, p1, p2+j2-i2]   += v
                    matrix[p1+spans_1[ne1-1]-1, p2+i2, p1+1, p2+j2-i2] += vip

    elif interface_nb == 2:
        bx_left  = p3/(knots_3[p3+1]-knots_3[0])*omega_1[1]/omega_1[0]
        bx_right = p3/(knots_3[n_knots3]-knots_3[n_knots3-p3-1])*omega_1[n_omega1-1]/omega_1[n_omega1]
        # ... 
        bx_leftFE  = p1/(knots_1[p1+1]-knots_1[0])
        bx_rightFE = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])
        # ... u1*v2
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g2 in range(0, k2):

                i_span_3 = spans_3[ie1, 0]
                i_span_4 = spans_4[ie2, g2]
                lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p4+1):

                    bj_0     = basis_4[ie2,il_2,0,g2]
                    bj_y     = basis_4[ie2,il_2,1,g2]
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
                        bi_x  = -1.*bi_0 * bx_leftFE
                        bi_px =  1.*bi_0 * bx_leftFE
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

                    matrix[p1, p2+i2, p1, p2+j2-i2]    += v
                    matrix[p1+1, p2+i2, p1-1, p2+j2-i2]  += vip

        #... Assemble the boundary condition for Nitsche (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]

            for g2 in range(0, k2):
                i_span_3 = spans_3[ie1, k1-1]
                i_span_4 = spans_4[ie2, g2]
                lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p4+1):

                    bj_0      = basis_4[ie2,il_2,0,g2]
                    bj_y      = basis_4[ie2,il_2,1,g2]

                    coeff_m1  = lcoeffs_m1[p3, il_2]
                    coeff_m2  = lcoeffs_m2[p3, il_2]                    
                    coeff_m10 = lcoeffs_m1[p3-1, il_2]
                    coeff_m20 = lcoeffs_m2[p3-1, il_2]

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
                        bj_x    = bj_0 * bx_rightFE
                        bj_px   = -1.*bj_0 * bx_rightFE
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

                    matrix[p1, p2+i2, p1, p2+j2-i2]   += v
                    matrix[p1, p2+i2, p1+1, p2+j2-i2] += vjp

    elif interface_nb == 3:
        by_left    = p4/(knots_4[p4+1]-knots_4[0])*omega_2[1]/omega_2[0]
        by_right   = p4/(knots_4[n_knots4]-knots_4[n_knots4-p4-1])*omega_2[n_omega2-1]/omega_2[n_omega2]
        # ...
        by_leftFE  = p2/(knots_2[p2+1]-knots_2[0])
        by_rightFE = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])
        # ...
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            for g1 in range(0, k1):
                i_span_3 = spans_3[ie1, g1]
                i_span_4 = spans_4[ie2, 0]
                lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p3+1):

                    bj_0     = basis_3[ie1,il_1,0,g1]
                    bj_x     = basis_3[ie1,il_1,1,g1]
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
                        bj_y  = -1*bj_0*by_leftFE
                        bj_py =  1*bj_0*by_leftFE
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

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2-1] += vjp

        #... Assemble the boundary condition for Nitsche (x=right)
        ie2      = ne2 -1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            for g1 in range(0, k1):
                i_span_3 = spans_3[ie1, g1]
                i_span_4 = spans_4[ie2, k2-1]
                lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p3+1):

                    bj_0     = basis_3[ie1,il_1,0,g1]
                    bj_x     = basis_3[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, p4]
                    coeff_m2 = lcoeffs_m2[il_1, p4]
                    coeff_m10= lcoeffs_m1[il_1, p4-1]
                    coeff_m20= lcoeffs_m2[il_1, p4-1]
                    
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
                        bi_y  = bi_0*by_rightFE
                        bi_py = -1.*bi_0*by_rightFE
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

                    matrix[p1+i1, p2+spans_2[ne2-1], p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+spans_2[ne2-1]-1, p1+j1-i1, p2+1] += vip

    elif interface_nb == 4:
        by_left    = p4/(knots_4[p4+1]-knots_4[0])*omega_2[1]/omega_2[0]
        by_right   = p4/(knots_4[n_knots4]-knots_4[n_knots4-p4-1])*omega_2[n_omega2-1]/omega_2[n_omega2]
        # ...
        by_leftFE  = p2/(knots_2[p2+1]-knots_2[0])
        by_rightFE = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])
        #... Assemble the boundary condition for Nitsche (x=left)
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            for g1 in range(0, k1):
                i_span_3 = spans_3[ie1, g1]
                i_span_4 = spans_4[ie2, 0]
                lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p3+1):

                    bj_0     = basis_3[ie1,il_1,0,g1]
                    bj_x     = basis_3[ie1,il_1,1,g1]
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
                        bi_y  = -1*bi_0*by_leftFE
                        bi_py =  1*bi_0*by_leftFE
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

                    matrix[p1+i1, p2, p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2+1, p1+j1-i1, p2-1] += vip

        #... Assemble the boundary condition for Nitsche (x=right)
        ie2      = ne2 -1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            for g1 in range(0, k1):
                i_span_3 = spans_3[ie1, g1]
                i_span_4 = spans_4[ie2, k2-1]
                lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                # ... compute the normal derivative
                F1x = 0.0
                F2x = 0.0
                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p3+1):

                    bj_0     = basis_3[ie1,il_1,0,g1]
                    bj_x     = basis_3[ie1,il_1,1,g1]
                    coeff_m1 = lcoeffs_m1[il_1, p4]
                    coeff_m2 = lcoeffs_m2[il_1, p4]
                    coeff_m10= lcoeffs_m1[il_1, p4-1]
                    coeff_m20= lcoeffs_m2[il_1, p4-1]
                    
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
                        bj_y    = bj_0*by_rightFE
                        bj_py   = -1.*bj_0*by_rightFE
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
                    matrix[p1+i1, p2, p1+j1-i1, p2]   += v
                    matrix[p1+i1, p2, p1+j1-i1, p2+1] += vjp
    # ...

#-------------------------------------------------------------------------------------------------------
# vector with respect to given patch Dirichlet !!!!
#--------------------------------------------------------------------------------------------------------
def assemble_vector_Dirichlet(
    ne1: 'int', ne2: 'int',
    p1: 'int', p2: 'int', p3: 'int', p4: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',spans_3: 'int[:,:]', spans_4: 'int[:,:]',
    basis_1: 'float[:,:,:,:]', basis_2: 'float[:,:,:,:]',basis_3: 'float[:,:,:,:]', basis_4: 'float[:,:,:,:]',
    weights_1: 'float[:,:]', weights_2: 'float[:,:]',
    points_1: 'float[:,:]', points_2: 'float[:,:]',
    knots_1: 'float[:]', knots_2: 'float[:]', knots_3: 'float[:]', knots_4: 'float[:]',
    vector_m1: 'float[:,:]', vector_m2: 'float[:,:]', vector_d: 'float[:,:]',
    omega_1: 'float[:]', omega_2: 'float[:]',
    interfaces: 'int[:]', Kappa: 'float',
    normalS: 'float', rhs: 'float[:,:]'
):
    #..assemble  solution times noraml(test fuction)
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #...
    nb_interfaces = interfaces.shape[0]
    n_omega1    = omega_1.shape[0]-1
    n_omega2    = omega_2.shape[0]-1
    n_knots1    = knots_1.shape[0]-1
    n_knots2    = knots_2.shape[0]-1
    n_knots3    = knots_3.shape[0]-1
    n_knots4    = knots_4.shape[0]-1
    # ...
    lcoeffs_m1 = zeros((p3+1,p4+1))
    lcoeffs_m2 = zeros((p3+1,p4+1))
    lcoeffs_d  = zeros((p1+1,p2+1))

    #......................................
    # .. Nitsches method
    #......................................
    u_d2       = zeros(k2)
    u_d2x      = zeros(k2)
    u_d2y      = zeros(k2)
    F_1x       = zeros(k2)
    F_2x       = zeros(k2)
    F_1y       = zeros(k2)
    F_2y       = zeros(k2)
    J_mat2     = zeros(k2)
    # ...
    u_d1       = zeros(k1)
    u_d1x      = zeros(k1)
    u_d1y      = zeros(k1)
    F1_1x      = zeros(k1)
    F1_2x      = zeros(k1)
    F1_1y      = zeros(k1)
    F1_2y      = zeros(k1)
    J_mat1     = zeros(k1)
    # ... build matrices
    for j_interface in range(0, nb_interfaces):
        interface_nb = interfaces[j_interface]
        if interface_nb == 1:
            bx_left   = p3/(knots_3[p3+1]-knots_3[0])*(omega_1[1]/omega_1[0])
            bx_leftFE = p1/(knots_1[p1+1]-knots_1[0])
            #... Assemble the boundary condition for Nitsche (x=left)
            ie1      = 0
            i_span_1 = spans_1[ie1]
            for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]

                lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                for g2 in range(0, k2):

                    i_span_3 = spans_3[ie1, 0]
                    i_span_4 = spans_4[ie2, g2]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_2 in range(0, p4+1):

                        bj_0     = basis_4[ie2,il_2,0,g2]
                        bj_y     = basis_4[ie2,il_2,1,g2]
                        coeff_m1 = lcoeffs_m1[0, il_2]
                        coeff_m2 = lcoeffs_m2[0, il_2]
                        coeff_m11= lcoeffs_m1[1, il_2]
                        coeff_m22= lcoeffs_m2[1, il_2]
                        
                        F1x     +=  (coeff_m11-coeff_m1) * bj_0 * bx_left
                        F2x     +=  (coeff_m22-coeff_m2) * bj_0 * bx_left
                        F1y     +=  coeff_m1 * bj_y
                        F2y     +=  coeff_m2 * bj_y
                    # ... compute dirichlet
                    ud  = 0.0
                    udx = 0.0
                    udy = 0.0
                    for il_2 in range(0, p2+1):

                        bj_0     = basis_2[ie2,il_2,0,g2]
                        bj_y     = basis_2[ie2,il_2,1,g2]
                        coeff_d  = lcoeffs_d[0, il_2]
                        coeff_d1 = lcoeffs_d[1, il_2]
                        ud      +=  coeff_d * bj_0
                        udx     +=  (coeff_d1-coeff_d) * bj_0 * bx_leftFE
                        udy     +=  coeff_d * bj_y

                    u_d2[g2]  = ud
                    u_d2x[g2] = udx
                    u_d2y[g2] = udy
                    # ... compute the normal derivative
                    F_1x[g2] = F1x
                    F_2x[g2] = F2x
                    F_1y[g2] = F1y
                    F_2y[g2] = F2y
                    # ....
                    J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
                for il_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = -1* bi_0 * bx_leftFE
                        bi_px =  1* bi_0 * bx_leftFE
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        ud    = u_d2[g2]
                        udx   = u_d2x[g2]
                        udy   = u_d2y[g2]
                        # ...
                        comp_1          = +1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = +1 * ( F_2y[g2]*udx - F_2x[g2]*udy)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += -1 * (-F_1y[g2]*udx + F_1x[g2]*udy)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        wvol    = weights_2[ie2, g2]
                        # ...
                        v      +=  normalS * (comp_1 * ud  + 0.*bi_0*comp_2) * wvol + Kappa * bi_0 * ud * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)
                        vip    +=  normalS * (comp_3 * ud) * wvol

                    rhs[p1, p2+i2]    += v
                    rhs[p1+1, p2+i2]  += vip
        if interface_nb == 2:
            bx_right   = p3/(knots_3[n_knots3]-knots_3[n_knots3-p3-1])*(omega_1[n_omega1-1]/omega_1[n_omega1])
            bx_rightFE = p1/(knots_1[n_knots1]-knots_1[n_knots1-p1-1])
            #... Assemble the boundary condition for Nitsche (x=right)
            ie1      = ne1 -1
            i_span_1 = spans_1[ie1]
            for ie2 in range(0, ne2):         
                i_span_2 = spans_2[ie2]
                lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]                
                for g2 in range(0, k2):
                    i_span_3 = spans_3[ie1, k1-1]
                    i_span_4 = spans_4[ie2, g2]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_2 in range(0, p4+1):

                        bj_0      = basis_4[ie2,il_2,0,g2]
                        bj_y      = basis_4[ie2,il_2,1,g2]

                        coeff_m1  = lcoeffs_m1[p3, il_2]
                        coeff_m2  = lcoeffs_m2[p3, il_2]                    
                        coeff_m10 = lcoeffs_m1[p3-1, il_2]
                        coeff_m20 = lcoeffs_m2[p3-1, il_2]

                        F1y      +=  coeff_m1 * bj_y
                        F2y      +=  coeff_m2 * bj_y
                        F1x      +=  (coeff_m1-coeff_m10) * bj_0 * bx_right
                        F2x      +=  (coeff_m2-coeff_m20) * bj_0 * bx_right
                    # ... compute dirichlet
                    ud  = 0.0
                    udx = 0.0
                    udy = 0.0
                    for il_2 in range(0, p2+1):

                        bj_0     = basis_2[ie2,il_2,0,g2]
                        bj_y     = basis_2[ie2,il_2,1,g2]
                        coeff_d  = lcoeffs_d[p1, il_2]
                        coeff_d1 = lcoeffs_d[p1-1, il_2]
                        ud      +=  coeff_d * bj_0
                        udx     +=  (coeff_d-coeff_d1) * bj_0 * bx_leftFE
                        udy     +=  coeff_d * bj_y

                    u_d2[g2]  = ud
                    u_d2x[g2] = udx
                    u_d2y[g2] = udy
                    # ... compute the normal derivative
                    F_1x[g2] = F1x
                    F_2x[g2] = F2x
                    F_1y[g2] = F1y
                    F_2y[g2] = F2y
                    # ....
                    J_mat2[g2] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)

                for il_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2

                    v   = 0.0
                    vpi = 0.0 # penultimate derivative
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bi_x  = bi_0 * bx_rightFE
                        bi_px = -1.*bi_0 * bx_rightFE
                        bi_y  = basis_2[ie2, il_2, 1, g2]
                        #...
                        ud    = u_d2[g2]
                        udx   = u_d2x[g2]
                        udy   = u_d2y[g2]
                        # ...
                        comp_1          = -1 * ( F_2y[g2]*bi_x - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F_1y[g2]*bi_x + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_2          = -1 * ( F_2y[g2]*udx - F_2x[g2]*udy)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_2         += +1 * (-F_1y[g2]*udx + F_1x[g2]*udy)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F_2y[g2]*bi_px - F_2x[g2]*bi_y)/J_mat2[g2] * F_2y[g2] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F_1y[g2]*bi_px + F_1x[g2]*bi_y)/J_mat2[g2] * F_1y[g2] #/sqrt(F1y**2+ F2y**2)
                        #...
                        wvol  = weights_2[ie2, g2]
                        # ... - 0.5*u1*v1_n - 0.5*u1_n*v1
                        v    += normalS * (comp_1 * ud + 0.*bi_0*comp_2)  * wvol + Kappa * bi_0 * ud * wvol * sqrt(F_1y[g2]**2 + F_2y[g2]**2)
                        vpi  += normalS * (comp_3 * ud)  * wvol 

                    rhs[i_span_1+p1, p2+i2]      += v
                    rhs[i_span_1+p1-1, p2+i2]    += vpi
        if interface_nb == 3:
            by_left   = p4/(knots_4[p4+1]-knots_4[0])*omega_2[1]/omega_2[0]
            by_leftFE = p2/(knots_2[p2+1]-knots_2[0])
            #... Assemble the boundary condition for Nitsche (y=left)
            ie2      = 0
            i_span_2 = spans_2[ie2]
            for ie1 in range(0, ne1):
                i_span_1 = spans_1[ie1]

                lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                for g1 in range(0, k1):
                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, 0]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    # ... compute the normal derivative
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):

                        bj_0     = basis_3[ie1,il_1,0,g1]
                        bj_x     = basis_3[ie1,il_1,1,g1]
                        coeff_m1 = lcoeffs_m1[il_1, 0]
                        coeff_m2 = lcoeffs_m2[il_1, 0]
                        coeff_m11= lcoeffs_m1[il_1, 1]
                        coeff_m22= lcoeffs_m2[il_1, 1]
                        
                        F1x     +=  coeff_m1 * bj_x
                        F2x     +=  coeff_m2 * bj_x
                        F1y     +=  (coeff_m11-coeff_m1) * bj_0 * by_left
                        F2y     +=  (coeff_m22-coeff_m2) * bj_0 * by_left
                    # ... compute dirichlet
                    ud  = 0.0
                    for il_1 in range(0, p1+1):

                        bj_0     = basis_1[ie1,il_1,0,g1]
                        coeff_d  = lcoeffs_d[il_1, 0]
                        ud      +=  coeff_d * bj_0
                    u_d1[g1] = ud
                    # ... compute the normal derivative
                    F1_1x[g1]  = F1x
                    F1_2x[g1]  = F2x
                    F1_1y[g1]  = F1y
                    F1_2y[g1]  = F2y
                    # ....
                    J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
                for il_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1

                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = -1* bi_0 * by_leftFE
                        bi_py =  1* bi_0 * by_leftFE
                        # ...
                        comp_1          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = -1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += +1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        ud = u_d1[g1]
                        #...
                        wvol            = weights_1[ie1, g1]
                        # ...
                        v              +=  normalS * (comp_1 * ud) * wvol + Kappa * bi_0 * ud * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)
                        vip            +=  normalS * (comp_3 * ud) * wvol

                    rhs[p1+i1, p2]   += v
                    rhs[p1+i1, p2+1] += vip
        if interface_nb == 4:
            by_right   = p4/(knots_4[n_knots4]-knots_4[n_knots4-p4-1])*(omega_2[n_omega2-1]/omega_2[n_omega2])
            by_rightFE = p2/(knots_2[n_knots2]-knots_2[n_knots2-p2-1])
            #... Assemble the boundary condition for Nitsche (y=right)
            ie2      = ne2 -1
            i_span_2 = spans_2[ie2]
            for ie1 in range(0, ne1):
                i_span_1 = spans_1[ie1]

                lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                for g1 in range(0, k1):
                    i_span_3 = spans_3[ie1, g1]
                    i_span_4 = spans_4[ie2, k2-1]
                    lcoeffs_m1[ : , : ] = vector_m1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    lcoeffs_m2[ : , : ] = vector_m2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
                    F1x = 0.0
                    F2x = 0.0
                    F1y = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p3+1):

                        bj_0     = basis_3[ie1,il_1,0,g1]
                        bj_x     = basis_3[ie1,il_1,1,g1]
                        coeff_m1 = lcoeffs_m1[il_1, p4]
                        coeff_m2 = lcoeffs_m2[il_1, p4]
                        coeff_m10= lcoeffs_m1[il_1, p4-1]
                        coeff_m20= lcoeffs_m2[il_1, p4-1]
                        
                        F1x     +=  coeff_m1 * bj_x
                        F2x     +=  coeff_m2 * bj_x
                        F1y     +=  (coeff_m1-coeff_m10) * bj_0*by_right
                        F2y     +=  (coeff_m2-coeff_m20) * bj_0*by_right
                    # ... compute dirichlet
                    ud  = 0.0
                    for il_1 in range(0, p1+1):

                        bj_0     = basis_1[ie1,il_1,0,g1]
                        coeff_d  = lcoeffs_d[il_1, p2]
                        ud      +=  coeff_d * bj_0
                    u_d1[g1] = ud
                    # ... compute the normal derivative
                    F1_1x[g1]  = F1x
                    F1_2x[g1]  = F2x
                    F1_1y[g1]  = F1y
                    F1_2y[g1]  = F2y
                    # ....
                    J_mat1[g1] = abs(F1x*F2y-F1y*F2x) #sqrt(F1y**2 + F2y**2)
                for il_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1

                    v    = 0.0
                    vip  = 0.0 # penultimate derivative
                    for g1 in range(0, k1):
                        bi_0  = basis_1[ie1, il_1, 0, g1]
                        bi_x  = basis_1[ie1, il_1, 1, g1]
                        bi_y  = bi_0*by_rightFE
                        bi_py  = -1.*bi_0*by_rightFE
                        # ...
                        comp_1          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_y)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_1         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_y)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        comp_3          = +1 * ( F1_2y[g1]*bi_x - F1_2x[g1]*bi_py)/J_mat1[g1] * F1_2x[g1] #/sqrt(F1y**2+ F2y**2)
                        comp_3         += -1 * (-F1_1y[g1]*bi_x + F1_1x[g1]*bi_py)/J_mat1[g1] * F1_1x[g1] #/sqrt(F1y**2+ F2y**2)
                        # ...
                        ud = u_d1[g1]
                        #...
                        wvol  = weights_1[ie1, g1]
                        # ...
                        v    +=  normalS * (comp_1 *ud) * wvol + Kappa * bi_0 * ud * wvol * sqrt(F1_1x[g1]**2 + F1_2x[g1]**2)
                        vip  +=  normalS * (comp_3 *ud) * wvol
                    rhs[p1+i1, i_span_2+p2]   += v
                    rhs[p1+i1, i_span_2+p2-1] += vip
    # ...