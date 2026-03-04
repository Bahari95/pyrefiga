__all__ = ['assemble_vector_ex01',
           'assemble_vector_ex02'
]

#==============================================================================
#
def assemble_vector_ex01(ne1:'int', ne2:'int', p1:'int', p2:'int', 
                         spans_1:'int[:]', spans_2:'int[:]', 
                         basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]', 
                         weights_1:'float64[:,:]', weights_2:'float64[:,:]', 
                         points_1:'float64[:,:]', points_2:'float64[:,:]', 
                         vector_u:'float64[:,:]', vector_w:'float64[:,:]', vector_v:'float64[:,:]', rho:'float', 
                         rhs:'float64[:,:]'):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p1+1,p2+1))
    lcoeffs_v = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            #... Integration of Dirichlet boundary conditions
            lcoeffs_v[ : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            # ...
                            coeff_u   = lcoeffs_v[il_1,il_2]
                            # ...
                            sx      +=  coeff_u*bj_x
                            sy      +=  coeff_u*bj_y
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy

            lvalues_u[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]            
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    ty = 0.0
                    x  = 0.0
                    y  = 0.0
                    #/:
                    sx = 0.0
                    sy = 0.0
                    #..
                    sxx = 0.0
                    syy = 0.0
                    sxy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):
                              bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              # ...
                              bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # ...
                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              #...
                              coeff_u = lcoeffs_u[il_1,il_2]
                              sx     +=  coeff_u*bj_x
                              sy     +=  coeff_u*bj_y
                              sxx    +=  coeff_u*bj_xx
                              syy    +=  coeff_u*bj_yy
                              sxy    +=  coeff_u*bj_xy
                              #...
                              coeff_w = lcoeffs_w[il_1,il_2]
                              ty +=  coeff_w*bj_y
                              x  +=  coeff_u*bj_0
                              y  +=  coeff_w*bj_0                                                            
                    #.. 
                    G_u = sqrt(sx**2 + ty**2 + 2.*sy**2 + 2.*rho)
                    DG_u= sxx*sx + syy*ty + 2.*sxy*sy 
                    lvalues_u[g1,g2] = -DG_u/G_u 
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            ux   = lvalues_ux[g1,g2]
                            uy   = lvalues_uy[g1,g2]
                            #..
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...

def assemble_vector_ex02(ne1:'int', ne2:'int', p1:'int', p2:'int', 
                        spans_1:'int[:]', spans_2:'int[:]',
                        basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]', 
                        weights_1:'float64[:,:]', weights_2:'float64[:,:]', 
                        points_1:'float64[:,:]', points_2:'float64[:,:]', 
                        vector_u:'float64[:,:]', vector_w:'float64[:,:]', vector_v:'float64[:,:]', rho:'float', 
                        rhs:'float64[:,:]'):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p1+1,p2+1))
    lcoeffs_v = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))          
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            #... Integration of Dirichlet boundary conditions
            lcoeffs_v[ : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            # ...
                            coeff_u   = lcoeffs_v[il_1,il_2]
                            # ...
                            sx      +=  coeff_u*bj_x
                            sy      +=  coeff_u*bj_y
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy
                    
            lvalues_u[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]            
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    sx = 0.0
                    x  = 0.
                    y  = 0.
                    #/:
                    tx = 0.0
                    ty = 0.0
                    #..
                    txx = 0.0
                    tyy = 0.0
                    txy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              # ...
                              bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # ...
                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              #...
                              coeff_w = lcoeffs_w[il_1,il_2]
                              tx     +=  coeff_w*bj_x
                              ty     +=  coeff_w*bj_y
                              txx    +=  coeff_w*bj_xx
                              tyy    +=  coeff_w*bj_yy
                              txy    +=  coeff_w*bj_xy
                              #...
                              coeff_u = lcoeffs_u[il_1,il_2]
                              sx +=  coeff_u*bj_x
                              x  +=  coeff_u*bj_0
                              y  +=  coeff_w*bj_0
                                                            
                    #.. 
                    G_u = sqrt(sx**2 + ty**2 + 2.*tx**2 + 2.*rho)
                    DG_u= txx*sx + tyy*ty + 2.*txy*tx
                    lvalues_u[g1,g2] = -DG_u/G_u
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            ux   = lvalues_ux[g1,g2]
                            uy   = lvalues_uy[g1,g2]
                            #..
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...