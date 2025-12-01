__all__ = ['assemble_vector_ex01',
           'assemble_vector_ex02',
           'assemble_vector_ex03',
           'assemble_vector_ex04',
           'assemble_Quality_ex01'
]
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
def assemble_vector_ex01(ne1: 'int', ne2: 'int', p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float64[:,:,:,:]', basis_2: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', weights_2: 'float64[:,:]',
    points_1: 'float64[:,:]', points_2: 'float64[:,:]',
    vector_u: 'float64[:,:]', vector_w: 'float64[:,:]',
    vector_v1: 'float64[:,:]', vector_v2: 'float64[:,:]',
    spans_ad1: 'int[:,:,:,:]', spans_ad2: 'int[:,:,:,:]',
    basis_ad1: 'float64[:,:,:,:,:,:]', basis_ad2: 'float64[:,:,:,:,:,:]',
    corners: 'float64[:]', rhs: 'float64[:,:]'
):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    # ... sizes
    k1          = weights_1.shape[1]
    k2          = weights_2.shape[1]
    lcoeffs_u   = zeros((p1+1,p2+1))
    lcoeffs_w   = zeros((p1+1,p2+1))
    lcoeffs_v1  = zeros((p1+1,p2+1))
    lcoeffs_v2  = zeros((p1+1,p2+1))
    lvalues_u   = zeros((k1, k2))
    # ...
    lvalues_u1  = zeros((k1, k2))
    lvalues_u1x = zeros((k1, k2))
    lvalues_u1y = zeros((k1, k2))
    lvalues_u2  = zeros((k1, k2))
    lvalues_u2x = zeros((k1, k2))
    lvalues_u2y = zeros((k1, k2))

    # ... build rhs
    # ...coefficient of normalisation
    Crho      = 0.0
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    x         = 0.0
                    y         = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              
                              bi_0      = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              # ...
                              x        +=  coef_v1*bi_0
                              y        +=  coef_v2*bi_0
                    
                    #.. Test 1
                    #rho  = (1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) )
                    #rho  = (1.+5./(1.+exp(100.*((x-0.5)**2+(y-0.5)**2-0.1))))
                    #rho  = (2.+sin(10.*pi*sqrt((x-0.6)**2+(y-0.6)**2)) )#0.8
                    #rho  = (1.+ 9./(1.+(10.*sqrt((x-0.-0.25*0.)**2+(y-0.)**2)*cos(arctan2(y-0.,x-0.-0.25*0.) -20.*((x-0.-0.25*0.)**2+(y-0.)**2)))**2) )
                    #rho   = (1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2)
                    #rho   =  5./(2.+cos(4.*pi*sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2)))
                    rho   = 9./(2.+cos(10.*pi*sqrt((x)**2+(y+2.)**2)))
                    #rho   =  1.+9.*exp(-10.*abs((x-0.5-0.0*cos(2.*pi*0.))**2-(y-0.5-0.5 *sin(2.*pi*0.))**2- 0.09))
                    #rho   =  1.+5.*exp(-0.25*abs((x-0.)**2+(y-0.)**2-1.05**2))
                    #rho   = 1.+12./cosh( 80.*((x + y) )**2 )
                    #rho   = 1.+5./cosh(40.*(2./(y**2-x*(x-1)**2+1.)-2.))**2+5./cosh(10.*(2./(y**2-x*(x-1)**2+1.)-2.))**2
                    #rho   = 1+10.*exp(-50.*abs(x**2+y**2-0.5))
                    Crho += rho * wvol 
    #..                
    int_rhsP    = 0.
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            #..
            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2x[ : , : ] = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ]   = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_2[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1[g1,g2]  += coeff_u*b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2
                            #...
                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2x[g1,g2] += coeff_w*db1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2

            lvalues_u[ : , : ] = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1   = lvalues_u1[g1,g2]
                    x2   = lvalues_u2[g1,g2]
                    #... We compute firstly the span in new adapted points
                    span_5    = spans_ad1[ie1, ie2, g1, g2]
                    span_6    = spans_ad2[ie1, ie2, g1, g2]  

                    #------------------   
                    lcoeffs_v1[ : , : ]  =  vector_v1[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                    lcoeffs_v2[ : , : ]  =  vector_v2[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                    
                    #x    = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #y    = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    x     = 0.0
                    y     = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              bi_0      = basis_ad1[ie1, ie2, il_1, 0, g1, g2]*basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                              # ...
                              x        +=  coef_v1*bi_0
                              y        +=  coef_v2*bi_0
                    #.. Test 1
                    #rho  = Crho/(1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) )
                    #rho  = Crho/(2.+sin(10.*pi*sqrt((x-0.6)**2+(y-0.6)**2)) )#0.8
                    #rho  = Crho/(1.+5./(1.+exp(100.*((x-0.5)**2+(y-0.5)**2-0.1))))
                    #rho  = Crho/(1.+ 9./(1.+(10.*sqrt((x-0.-0.25*0.)**2+(y-0.)**2)*cos(arctan2(y-0.,x-0.-0.25*0.) -20.*((x-0.-0.25*0.)**2+(y-0.)**2)))**2) )
                    #rho  = Crho/(1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2)
                    #rho  = Crho/(5./(2.+cos(4.*pi*sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2))))
                    rho  = Crho/(9./(2.+cos(10.*pi*sqrt((x)**2+(y+2.)**2))))
                    #rho   =  Crho/(1.+9.*exp(-10.*abs((x-0.5-0.0*cos(2.*pi*0.))**2-(y-0.5-0.5 *sin(2.*pi*0.))**2- 0.09)))
                    #rho   = Crho/(1.+5.*exp(-0.25*abs((x-0.)**2+(y-0.)**2-1.05**2)))
                    #rho   = Crho/(1.+12./cosh( 80.*((x + y) )**2 ))
                    #rho   = Crho/(1.+5./cosh(40.*(2./(y**2-x*(x-1)**2+1.)-2.))**2+5./cosh(10.*(2./(y**2-x*(x-1)**2+1.)-2.))**2)
                    #rho   = Crho/(1+10.*exp(-50.*abs(x**2+y**2-0.5)))

                    #...
                    u1x              = lvalues_u1x[g1,g2]
                    u1y              = lvalues_u1y[g1,g2]
                    #___
                    u2x              = lvalues_u2x[g1,g2]
                    u2y              = lvalues_u2y[g1,g2]
                    lvalues_u[g1,g2] = -sqrt((u1x + u2y)**2 + 2. * (rho - (u1x*u2y-u2x*u1y)))
                    wvol             = weights_1[ie1, g1]*weights_2[ie2, g2]
                    int_rhsP        += sqrt((u1x + u2y)**2 + 2. * (rho - (u1x*u2y-u2x*u1y)))*wvol 
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u     = lvalues_u[g1,g2]
                            v    += bi_0 * u * wvol

                    rhs[i1+p1,i2+p2] += v   
    # Integral in Neumann boundary
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int_N = corners[1] - corners[0] + corners[3] - corners[2]
    #Assuring Compatiblity condition
    coefs = int_N/int_rhsP  
    rhs   = rhs*coefs
     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann bundary Condition
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  #x1       =  points_1[ie1, g1]
                  
                  vx_0    += -1.*bi_0*corners[2] * wleng_x
                  vx_1    += bi_0*corners[3] * wleng_x

           rhs[i1+p1,0+p2]       += vx_0
           rhs[i1+p1,ne2+2*p2-1] += vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  wleng_y =  weights_2[ie2, g2]
                  #x2      =  points_2[ie2, g2]
                           
                  vy_0   += -1.*bi_0 * corners[0] * wleng_y
                  vy_1   += bi_0 * corners[1] * wleng_y

           rhs[0+p1,i2+p2]       += vy_0
           rhs[ne1-1+2*p1,i2+p2] += vy_1
    # ...

#==============================================================================
#
def assemble_vector_ex02(
    ne1: 'int', ne2: 'int', p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float64[:,:,:,:]', basis_2: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', weights_2: 'float64[:,:]',
    points_1: 'float64[:,:]', points_2: 'float64[:,:]',
    vector_v: 'float64[:,:]', dx: 'int', rhs: 'float64[:,:]'
):

    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_v = zeros((p1+1,p2+1))
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
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            ux   = lvalues_ux[g1,g2]
                            uy   = lvalues_uy[g1,g2]
                            #..
                            v += bi_0 * ( ux * dx + uy * (1.-dx)) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...

    #==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
def assemble_vectorbasis_ex02(
    ne1: 'int', p1: 'int', spans_1: 'int[:]', basis_1: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', points_1: 'float64[:,:]', knots_1: 'float64[:]',
    ovlp_value: 'float64', rhs: 'float64[:]'
):
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1        = weights_1.shape[1]
    # ...
    #---Computes All basis in a new points
    nders          = 1 # number of derivatives
    #...
    degree         = p1
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array

    #span = find_span( knots, degree, xq )
    xq   = ovlp_value
    #~~~~~~~~~~~~~~~
    # Knot index at left/right boundary
    low  = degree
    high = len(knots_1)-1-degree
    # Check if point is exactly on left/right boundary, or outside domain
    if xq <= knots_1[low ]: 
         span = low
    if xq >= knots_1[high]: 
         span = high-1
    else :
      # Perform binary search
      span = (low+high)//2
      while xq < knots_1[span] or xq >= knots_1[span+1]:
         if xq < knots_1[span]:
             high = span
         else:
             low  = span
         span = (low+high)//2
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = xq - knots_1[span-j]
        right[j] = knots_1[span+1+j] - xq
        saved    = 0.0
        for r in range(0,j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
            # compute basis functions and save them into upper triangular part of ndu
            temp       = ndu[r,j] * ndu[j+1,r]
            ndu[r,j+1] = saved + right[r] * temp
            saved      = left[j-r] * temp
        ndu[j+1,j+1] = saved	

    # Compute derivatives in 2D output array 'ders'
    ders[0,:] = ndu[:,degree]
    for r in range(0,degree+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1.0
        for k in range(1,nders+1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2,0] = a[s1,0] * ndu[pk+1,rk]
               d = a[s2,0] * ndu[rk,pk]
            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r
            for ij in range(j1,j2+1):
                a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
            for ij in range(j1,j2+1):
                d += a[s2,ij]* ndu[rk+ij,pk]
            if r <= pk:
               a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
               d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j  = s1
            s1 = s2
            s2 = j
    ders[1,:]   = ders[1,:] * r
    
    # ... assemble the non vanishing point in the overlap value
    for il_1 in range(0, p1+1):
           rhs[p1+il_1]   = ders[0,il_1]
           rhs[2*p1+il_1] = ders[1,il_1]
    # ... 

# Assembles Quality of mesh adaptation
#==============================================================================
def assemble_vector_ex03(
    ne1: 'int', ne2: 'int', p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float64[:,:,:,:]', basis_2: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', weights_2: 'float64[:,:]',
    points_1: 'float64[:,:]', points_2: 'float64[:,:]',
    vector_v1: 'float64[:,:]',
    spans_ad1: 'int[:,:,:,:]', spans_ad2: 'int[:,:,:,:]',
    basis_ad1: 'float64[:,:,:,:,:,:]', basis_ad2: 'float64[:,:,:,:,:,:]',
    rhs: 'float64[:,:]'
):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lvalues_u    = zeros((k1, k2))
    # ...
    lcoeffs_v1   = zeros((p1+1,p2+1))

    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    #... We compute firstly the span in new adapted points
                    span_5 = spans_ad1[ie1, ie2, g1, g2]
                    span_6 = spans_ad2[ie1, ie2, g1, g2]
                    #------------------   
                    lcoeffs_v1[ : , : ]  =  vector_v1[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                    #...
                    x     = 0.0                    
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):
                            coef_v1   = lcoeffs_v1[il_1,il_2]
                            bi_0      = basis_ad1[ie1, ie2, il_1, 0, g1, g2] * basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                            # ...
                            x        +=  coef_v1*bi_0

                    lvalues_u[g1,g2] = x
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            u     = lvalues_u[g1,g2]
                            #..
                            v    += bi_0 * u * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...
# Assembles Quality of mesh adaptation
#==============================================================================
def assemble_Quality_ex01(
    ne1: 'int', ne2: 'int', p1: 'int', p2: 'int',
    spans_1: 'int[:]', spans_2: 'int[:]',
    basis_1: 'float64[:,:,:,:]', basis_2: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', weights_2: 'float64[:,:]',
    points_1: 'float64[:,:]', points_2: 'float64[:,:]',
    knots_1: 'float64[:]', knots_2: 'float64[:]',
    vector_u: 'float64[:,:]', vector_w: 'float64[:,:]',
    vector_v1: 'float64[:,:]', vector_v2: 'float64[:,:]',
    vector_c1: 'float64[:,:]', vector_c2: 'float64[:,:]',
    times: 'float', omega_1: 'float64[:]', omega_2: 'float64[:]',
    spans_ad1: 'int[:,:,:,:]', spans_ad2: 'int[:,:,:,:]',
    basis_ad1: 'float64[:,:,:,:,:,:]', basis_ad2: 'float64[:,:,:,:,:,:]',
    rhs: 'float64[:,:]'
):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lcoeffs_u    = zeros((p1+1,p2+1))
    lcoeffs_w    = zeros((p1+1,p2+1))
    # ...
    lcoeffs_v1  = zeros((p1+1,p2+1))
    lcoeffs_v2  = zeros((p1+1,p2+1))
    # ...
    lcoeffs_u1  = zeros((p1+1,p2+1))
    lcoeffs_u2  = zeros((p1+1,p2+1))
    # ...
    lcoeffs_c1  = zeros((p1+1,p2+1))
    lcoeffs_c2  = zeros((p1+1,p2+1))

    Qual_l2      = 0.                                
    displacement = 0.
    area_exact   = 0.
    area_comp    = 0.
    area_appr    = 0.
    cst          = 1.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ]   = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u1[ : , : ]  = vector_v1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u2[ : , : ]  = vector_v2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_c1[ : , : ]  = vector_c1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_c2[ : , : ]  = vector_c2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            v             = 0.0
            w             = 0.0
            warea_exact   = 0.0
            warea_comp    = 0.0
            warea_appr    = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    #... We compute firstly the span in new adapted points
                    span_5 = spans_ad1[ie1, ie2, g1, g2]
                    span_6 = spans_ad2[ie1, ie2, g1, g2]

                    #------------------   
                    lcoeffs_v1[ : , : ]  =  vector_v1[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                    lcoeffs_v2[ : , : ]  =  vector_v2[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                    # ...
                    x     = 0.0
                    y     = 0.0
                    uhx   = 0.0
                    uhy   = 0.0
                    vhx   = 0.0
                    vhy   = 0.0
                    # ...
                    x1    = 0.0
                    x2    = 0.0
                    I1x   = 0.0
                    I1y   = 0.0
                    I2x   = 0.0
                    I2y   = 0.0
                    # ...
                    F1x   = 0.0
                    F1y   = 0.0
                    F2x   = 0.0
                    F2y   = 0.0                    
                    y1    = 0.0
                    y2    = 0.0
                    G1x   = 0.0
                    G1y   = 0.0
                    G2x   = 0.0
                    G2y   = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_0      = basis_ad1[ie1, ie2, il_1, 0, g1, g2] * basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                            bj_x      = basis_ad1[ie1, ie2, il_1, 1, g1, g2] * basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                            bj_y      = basis_ad1[ie1, ie2, il_1, 0, g1, g2] * basis_ad2[ie1, ie2, il_2, 1, g1, g2]
                            # ...
                            coef_v1   = lcoeffs_v1[il_1,il_2]
                            coef_v2   = lcoeffs_v2[il_1,il_2]
                            # ...in adapted points
                            x        +=  coef_v1*bj_0
                            y        +=  coef_v2*bj_0
                            F1x      +=  coef_v1*bj_x
                            F1y      +=  coef_v1*bj_y
                            F2x      +=  coef_v2*bj_x
                            F2y      +=  coef_v2*bj_y
                            # ...
                            coeff_c1  = lcoeffs_c1[il_1,il_2]
                            coeff_c2  = lcoeffs_c2[il_1,il_2]
                            coeff_u1  = lcoeffs_u1[il_1,il_2]
                            coeff_u2  = lcoeffs_u2[il_1,il_2]
                            coeff_u   = lcoeffs_u[il_1,il_2]
                            coeff_w   = lcoeffs_w[il_1,il_2]
                            #...
                            bi_0      = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x      = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y      = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            #... approximation of composite functions
                            y1       +=  coeff_c1*bi_0
                            y2       +=  coeff_c2*bi_0
                            G1x      +=  coeff_c1*bi_x
                            G1y      +=  coeff_c1*bi_y
                            G2x      +=  coeff_c2*bi_x
                            G2y      +=  coeff_c2*bi_y
                            #.. Initial mapping
                            x1       += coeff_u1*bi_0
                            x2       += coeff_u2*bi_0
                            I1x      += coeff_u1*bi_x
                            I1y      += coeff_u1*bi_y
                            I2x      += coeff_u2*bi_x
                            I2y      += coeff_u2*bi_y                                
                            #.. optimal mapping
                            uhx      += coeff_u*bi_x
                            uhy      += coeff_u*bi_y
                            vhx      += coeff_w*bi_x
                            vhy      += coeff_w*bi_y
                    #.. Test 1
                    #rho   = 1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) 
                    # Test 5
                    #rho  = (1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2) )
                    #rho  = (2.+sin(10.*pi*sqrt((x-0.6)**2+(y-0.6)**2)) )#0.8
                    #rho   = (1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2)
                    #rho   =  5./(2.+cos(4.*pi*sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2)))
                    rho   =  9./(2.+cos(10.*pi*sqrt((x)**2+(y+2.)**2)))
                    #rho   =  1.+9.*exp(-10.*abs((x-0.5-0.0*cos(2.*pi*0.))**2-(y-0.5-0.5 *sin(2.*pi*0.))**2- 0.09))
                    #rho   =  1.+5.*exp(-0.25*abs((x-0.)**2+(y-0.)**2-1.05**2))
                    #rho   = 1.+12./cosh( 80.*((x + y) )**2 )
                    #rho   = 1.+5./cosh(40.*(2./(y**2-x*(x-1)**2+1.)-2.))**2+5./cosh(10.*(2./(y**2-x*(x-1)**2+1.)-2.))**2
                    #rho   = 1+10.*exp(-50.*abs(x**2+y**2-0.5))
                    if ie1 == 0 and ie2 == 0 and g1 == 0 and g2 == 0:
                       cst = rho*(uhx*vhy-uhy*vhx)

                    wvol   = weights_1[ie1, g1] * weights_2[ie2, g2]
                    v    += (rho*(uhx*vhy-uhy*vhx)-cst)**2 * wvol
                    w    += ((x-x1)**2+(y-x2)**2) * wvol
                    # DF components: F1x, F1y, F2x, F2y  (columns F1=(F1x,F1y), F2=(F2x,F2y))
                    # DU components: uhx, uhy, vhx, vhy  (DU = [[uhx, uhy],[vhx, vhy]])

                    # new composed Jacobian J' = DF * DU
                    # F_1x =  F1x*uhx + F2x*vhx   # entry (0,0)
                    # F_1y =  F1y*uhx + F2y*vhx   # entry (1,0)
                    # F_2x =  F1x*uhy + F2x*vhy   # entry (0,1)
                    # F_2y =  F1y*uhy + F2y*vhy   # entry (1,1)
                    # F_2y = uhy*F2x+vhy*F2y #F'2y
                    # F_1x = uhx*F1x+vhx*F1y #F'1x
                    # F_1y = uhy*F1x+vhy*F1y #F'1y
                    # F_2x = uhx*F2x+vhx*F2y #F'2x
                    det_DF   = F2y*F1x - F1y*F2x
                    det_DU   = uhx*vhy - uhy*vhx
                    det_prod = abs(det_DF) * abs(det_DU)
                    # det_comp = (F_2y * F_1x - F_1y * F_2x)
                    # these two should be equal (within roundoff)
                    warea_exact +=  abs(I2y*I1x-I1y*I2x) * wvol
                    warea_comp  +=  det_prod * wvol
                    warea_appr  +=  abs(G2y*G1x-G1y*G2x) * wvol                        
            Qual_l2      += v
            displacement += w
            area_exact   += warea_exact
            area_comp    += warea_comp 
            area_appr    += warea_appr 
    rhs[p1,p2]   = sqrt(Qual_l2)
    rhs[p1,p2+1] = sqrt(displacement)
    rhs[p1,p2+2] = abs((area_comp)-(area_exact))
    rhs[p1,p2+3] = abs((area_appr)-(area_exact))
    rhs[p1,p2+5] = abs(area_exact)
    #---Computes All basis in a new points
    nders          = 1
    degree         = p1
    # ...
    xx             = zeros(k1)

    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis1         = zeros( (ne1, degree+1, nders+1, k1))
    for ie1 in range(ne1):
        i_span_1 = spans_1[ie1]
        lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, spans_2[0] : spans_2[0]+p2+1]
        xx[:] = 0.0
        for g1 in range(0, k1):
            v = 0.0
            for il_1 in range(0, p1+1):
                #...
                bi_0    = basis_1[ie1, il_1, 0, g1]
                coef_u  = lcoeffs_u[il_1,0]
                v      +=  coef_u*bi_0        
            xx[g1] = v
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_1)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_1[low ]: 
                 span = low
            elif xq >= knots_1[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_1[span] or xq >= knots_1[span+1]:
                 if xq < knots_1[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_1[span-j]
                right[j] = knots_1[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis1[ie1,:,0,iq] = ders[0,:]*omega_1[span-degree:span+1]
            basis1[ie1,:,0,iq] /= sum(basis1[ie1,:,0,iq])
        lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, spans_2[ne2-1] : spans_2[ne2-1]+p2+1]
        xx[:] = 0.0
        for g1 in range(0, k1):
            v = 0.0
            for il_1 in range(0, p1+1):
                #...
                bi_0    = basis_1[ie1, il_1, 0, g1]
                coef_u  = lcoeffs_u[il_1,p2]
                v      +=  coef_u*bi_0        
            xx[g1] = v
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_1)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_1[low ]: 
                 span = low
            elif xq >= knots_1[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_1[span] or xq >= knots_1[span+1]:
                 if xq < knots_1[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_1[span-j]
                right[j] = knots_1[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis1[ie1,:,1,iq] = ders[0,:]*omega_1[span-degree:span+1]
            basis1[ie1,:,1,iq] /= sum(basis1[ie1,:,1,iq])
    degree         = p2
    basis2         = zeros( (ne2, degree+1, nders+1, k2))
    for ie2 in range(ne2):
        i_span_2 = spans_2[ie2]
        lcoeffs_w[ : , : ]   = vector_w[spans_1[0] : spans_1[0]+p1+1, i_span_2 : i_span_2+p2+1]
        xx[:] = 0.0
        for g2 in range(0, k2):
            v = 0.0
            for il_2 in range(0, p2+1):
                #...
                bi_0    = basis_2[ie2, il_2, 0, g2]
                coef_w  = lcoeffs_w[0,il_2]
                v      +=  coef_w*bi_0        
            xx[g2] = v
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_2)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_2[low ]: 
                 span = low
            elif xq >= knots_2[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_2[span] or xq >= knots_2[span+1]:
                 if xq < knots_2[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_2[span-j]
                right[j] = knots_2[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis2[ie2,:,0,iq] = ders[0,:]*omega_2[span-degree:span+1]
            basis2[ie2,:,0,iq] /= sum(basis2[ie2,:,0,iq])
        lcoeffs_w[ : , : ]   = vector_w[spans_1[ne1-1] : spans_1[ne1-1]+p1+1, i_span_2 : i_span_2+p2+1]
        xx[:] = 0.0
        for g2 in range(0, k2):
            v = 0.0
            for il_2 in range(0, p2+1):
                #...
                bi_0    = basis_2[ie2, il_2, 0, g2]
                coef_w  = lcoeffs_w[p1,il_2]
                v      +=  coef_w*bi_0        
            xx[g2] = v
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_2)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_2[low ]: 
                 span = low
            elif xq >= knots_2[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_2[span] or xq >= knots_2[span+1]:
                 if xq < knots_2[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_2[span-j]
                right[j] = knots_2[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis2[ie2,:,1,iq] = ders[0,:]*omega_2[span-degree:span+1]
            basis2[ie2,:,1,iq] /= sum(basis2[ie2,:,1,iq])
    #... We compute the error at the boundary
    boundary_error = 0.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        ie2      = 0
        i_span_2 = spans_2[ie2]
        # ...
        lcoeffs_c1[ : , : ]  = vector_c1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_c2[ : , : ]  = vector_c2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        #... We compute firstly the span in new adapted points
        lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for g1 in range(0, k1):
            xq = 0.0
            for il_1 in range(0, p1+1):
                #...
                bi_0    = basis_1[ie1, il_1, 0, g1]
                coef_u  = lcoeffs_u[il_1,0]
                xq     +=  coef_u*bi_0        
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            degree = p1
            low  = degree
            high = len(knots_1)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_1[low ]: 
                span = low
            elif xq >= knots_1[high]: 
                span = high-1
            else : 
                # Perform binary search
                span = (low+high)//2
                while xq < knots_1[span] or xq >= knots_1[span+1]:
                    if xq < knots_1[span]:
                        high = span
                    else:
                        low  = span
                    span = (low+high)//2

            #------------------   
            lcoeffs_v1[ : , : ]  =  vector_v1[span : span+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[span : span+p1+1, i_span_2 : i_span_2+p2+1]
            #...
            x     = 0.0
            y     = 0.0
            y1    = 0.0
            y2    = 0.0
            for il_1 in range(0, p1+1):
                bi_0      = basis1[ie1, il_1, 0, g1]
                # ...in adapted points
                coef_v1   = lcoeffs_v1[il_1,0]
                coef_v2   = lcoeffs_v2[il_1,0]
                x        +=  coef_v1*bi_0
                y        +=  coef_v2*bi_0
                # ...
                bi_0      = basis_1[ie1, il_1, 0, g1]
                #... approximation of composite functions
                coeff_c1  = lcoeffs_c1[il_1,0]
                coeff_c2  = lcoeffs_c2[il_1,0]
                y1       +=  coeff_c1*bi_0
                y2       +=  coeff_c2*bi_0
            wvol   = weights_1[ie1, g1]
            boundary_error += ((x-y1)**2+(y-y2)**2) * wvol
        ie2      = ne2-1
        i_span_2 = spans_2[ie2]
        # ...
        lcoeffs_c1[ : , : ]  = vector_c1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_c2[ : , : ]  = vector_c2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        #... We compute firstly the span in new adapted points
        lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for g1 in range(0, k1):
            g2 = 0
            xq = 0.0
            for il_1 in range(0, p1+1):
                #...
                bi_0    = basis_1[ie1, il_1, 0, g1]
                coef_u  = lcoeffs_u[il_1,p2]
                xq    +=  coef_u*bi_0        
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            degree = p1
            low  = degree
            high = len(knots_1)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_1[low ]: 
                span = low
            elif xq >= knots_1[high]: 
                span = high-1
            else : 
                # Perform binary search
                span = (low+high)//2
                while xq < knots_1[span] or xq >= knots_1[span+1]:
                    if xq < knots_1[span]:
                        high = span
                    else:
                        low  = span
                    span = (low+high)//2

            #------------------   
            lcoeffs_v1[ : , : ]  =  vector_v1[span : span+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[span : span+p1+1, i_span_2 : i_span_2+p2+1]
            #...
            x     = 0.0
            y     = 0.0
            y1    = 0.0
            y2    = 0.0
            for il_1 in range(0, p1+1):
                #...
                bi_0      = basis1[ie1, il_1, 1, g1]
                # ...in adapted points
                coef_v1   = lcoeffs_v1[il_1,p2]
                coef_v2   = lcoeffs_v2[il_1,p2]
                x        +=  coef_v1*bi_0
                y        +=  coef_v2*bi_0
                # ...
                bi_0      = basis_1[ie1, il_1, 0, g1]
                #... approximation of composite functions
                coeff_c1  = lcoeffs_c1[il_1,p2]
                coeff_c2  = lcoeffs_c2[il_1,p2]
                y1       +=  coeff_c1*bi_0
                y2       +=  coeff_c2*bi_0
            wvol   = weights_1[ie1, g1]
            boundary_error += ((x-y1)**2+(y-y2)**2) * wvol
    for ie2 in range(0, ne2):
        ie1      = 0
        i_span_1 = spans_1[ie1]
        i_span_2 = spans_2[ie2]
        g1       = k1
        lcoeffs_c1[ : , : ]  = vector_c1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_c2[ : , : ]  = vector_c2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        #... We compute firstly the span in new adapted points
        lcoeffs_w[ : , : ]   = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for g2 in range(0, k2):
            xq = 0.0
            for il_2 in range(0, p2+1):
                #...
                bi_0    = basis_2[ie2, il_2, 0, g2]
                coef_w  = lcoeffs_w[0,il_2]
                xq    +=  coef_w*bi_0        
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            degree = p2
            low  = degree
            high = len(knots_2)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_2[low ]: 
                span = low
            elif xq >= knots_2[high]: 
                span = high-1
            else : 
                # Perform binary search
                span = (low+high)//2
                while xq < knots_2[span] or xq >= knots_2[span+1]:
                    if xq < knots_2[span]:
                        high = span
                    else:
                        low  = span
                    span = (low+high)//2
            #------------------   
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_1 : i_span_1+p1+1, span : span+p2+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_1 : i_span_1+p1+1, span : span+p2+1]
            #...
            x     = 0.0
            y     = 0.0
            y1    = 0.0
            y2    = 0.0
            for il_2 in range(0, p2+1):

                bi_0      = basis2[ie2, il_2, 0, g2]
                # ...in adapted points
                coef_v1   = lcoeffs_v1[0,il_2]
                coef_v2   = lcoeffs_v2[0,il_2]
                x        +=  coef_v1*bi_0
                y        +=  coef_v2*bi_0
                # ...
                bi_0      = basis_2[ie2, il_2, 0, g2]
                #... approximation of composite functions
                coeff_c1  = lcoeffs_c1[0,il_2]
                coeff_c2  = lcoeffs_c2[0,il_2]
                y1       +=  coeff_c1*bi_0
                y2       +=  coeff_c2*bi_0
            wvol   = weights_2[ie2, g2]
            boundary_error += ((x-y1)**2+(y-y2)**2) * wvol
        ie1      = ne1-1
        i_span_1 = spans_1[ie1]
        i_span_2 = spans_2[ie2]
        lcoeffs_c1[ : , : ]  = vector_c1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_c2[ : , : ]  = vector_c2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        #... We compute firstly the span in new adapted points
        lcoeffs_w[ : , : ]   = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for g2 in range(0, k2):
            xq = 0.0
            for il_2 in range(0, p2+1):
                #...
                bi_0    = basis_2[ie2, il_2, 0, g2]
                coef_w  = lcoeffs_w[p1,il_2]
                xq    +=  coef_w*bi_0        
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            degree = p2
            low  = degree
            high = len(knots_2)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_2[low ]: 
                span = low
            elif xq >= knots_2[high]: 
                span = high-1
            else : 
                # Perform binary search
                span = (low+high)//2
                while xq < knots_2[span] or xq >= knots_2[span+1]:
                    if xq < knots_2[span]:
                        high = span
                    else:
                        low  = span
                    span = (low+high)//2
            #------------------   
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_1 : i_span_1+p1+1, span : span+p2+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_1 : i_span_1+p1+1, span : span+p2+1]
            #...
            x     = 0.0
            y     = 0.0
            y1    = 0.0
            y2    = 0.0
            il_1  = 0
            for il_2 in range(0, p2+1):

                bi_0      = basis2[ie2, il_2, 1, g2]
                # ...in adapted points
                coef_v1   = lcoeffs_v1[p1,il_2]
                coef_v2   = lcoeffs_v2[p1,il_2]
                x        +=  coef_v1*bi_0
                y        +=  coef_v2*bi_0
                # ...
                bi_0      = basis_2[ie2, il_2, 0, g2]
                #... approximation of composite functions
                coeff_c1  = lcoeffs_c1[p1,il_2]
                coeff_c2  = lcoeffs_c2[p1,il_2]
                y1       +=  coeff_c1*bi_0
                y2       +=  coeff_c2*bi_0
            wvol   = weights_2[ie2, g2]
            boundary_error += ((x-y1)**2+(y-y2)**2) * wvol
    rhs[p1,p2+4] = sqrt(boundary_error)
    # ...

# Assembles Quality of mesh adaptation
#==============================================================================
def assemble_vector_ex04(
    ne1: 'int', ne2: 'int', p1: 'int', p2: 'int',
    spans_1: 'int[:,:]', spans_2: 'int[:,:]',
    basis_1: 'float64[:,:,:,:]', basis_2: 'float64[:,:,:,:]',
    weights_1: 'float64[:,:]', weights_2: 'float64[:,:]',
    points_1: 'float64[:,:]', points_2: 'float64[:,:]',
    knots_1: 'float64[:]', knots_2: 'float64[:]',
    vector_u1: 'float64[:,:]', vector_u2: 'float64[:,:]',
    vector_v1: 'float64[:,:]', omega_1: 'float64[:]', omega_2: 'float64[:]',
    rhs: 'float64[:,:]'
):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lvalues_u    = zeros((k1, k2))
    # ...
    lcoeffs_v1   = zeros((p1+1,p2+1))
    # ___
    lcoeffs_u1   = zeros((p1+1,p2+1))
    lcoeffs_u2   = zeros((p1+1,p2+1))

    #   ---Computes All basis in a new points    
    degree         = p1
    nders          = 0 # number of derivatives 
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis_ad1      = empty((ne1, ne2, p1+1, nders+1, k1, k2))
    basis_ad2      = empty((ne1, ne2, p1+1, nders+1, k1, k2))
    spans_ad1      = zeros((ne1, ne2, k1, k2)	, int)
    spans_ad2      = zeros((ne1, ne2, k1, k2)	, int)
    # ...
    degree         = p2
    left2          = empty( degree )
    right2         = empty( degree )
    a2             = empty( (       2, degree+1) )
    ndu2           = empty( (degree+1, degree+1) )
    ders2          = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 degree   = p1
                 i_span_1 = spans_1[ie1, g1]
                 i_span_2 = spans_2[ie2, g2]
                 lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                 xq = 0.0
                 for il_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        bj_0    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                        coeff_u1 = lcoeffs_u1[il_1,il_2]
                        xq     += coeff_u1 * bj_0

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_1)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
                 if xq <= knots_1[low ]: 
                      span = low
                 elif xq >= knots_1[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_1[span] or xq >= knots_1[span+1]:
                      if xq < knots_1[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 span_5   =  span
                 # ...
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_1[span-j]
                     right[j] = knots_1[span+1+j] - xq
                     saved    = 0.0
                     for r in range(0,j+1):
                         # compute inverse of knot differences and save them into lower triangular part of ndu
                         ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                         # compute basis functions and save them into upper triangular part of ndu
                         temp       = ndu[r,j] * ndu[j+1,r]
                         ndu[r,j+1] = saved + right[r] * temp
                         saved      = left[j-r] * temp
                     ndu[j+1,j+1] = saved	
               
                 # Compute derivatives in 2D output array 'ders'
                 ders[0,:] = ndu[:,degree]
                 for r in range(0,degree+1):
                     s1 = 0
                     s2 = 1
                     a[0,0] = 1.0
                     for k in range(1,nders+1):
                         d  = 0.0
                         rk = r-k
                         pk = degree-k
                         if r >= k:
                            a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                            d = a[s2,0] * ndu[rk,pk]
                         j1 = 1   if (rk  > -1 ) else -rk
                         j2 = k-1 if (r-1 <= pk) else degree-r
                         for ij in range(j1,j2+1):
                             a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                         for ij in range(j1,j2+1):
                             d += a[s2,ij]* ndu[rk+ij,pk]
                         if r <= pk:
                            a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                            d += a[s2,k] * ndu[r,pk]
                         ders[k,r] = d
                         j  = s1
                         s1 = s2
                         s2 = j
                 # ...first compute R1
                 ders[0,:]     = ders[0,:] * omega_1[span-degree:span+1]
                 sum_basisx    = sum(ders[0,:])
                 ders[0,:]     = ders[0,:]/sum_basisx
                 # ... Now for the second direction
                 degree   = p2
                 lcoeffs_u2[ : , : ] = vector_u2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                 xq = 0.0
                 for il_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):

                        bj_0     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                        coeff_u2 = lcoeffs_u2[il_1,il_2]
                        xq      += coeff_u2 * bj_0
                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left2/right2 boundary
                 low  = degree
                 high = len(knots_2)-1-degree
                 # Check if point is exactly on left2/right2 boundary, or outside domain
                 if xq <= knots_2[low ]: 
                      span = low
                 elif xq >= knots_2[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_2[span] or xq >= knots_2[span+1]:
                      if xq < knots_2[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 span_6 = span
                 # ...
                 ndu2[0,0] = 1.0
                 for j in range(0,degree):
                     left2 [j] = xq - knots_2[span-j]
                     right2[j] = knots_2[span+1+j] - xq
                     saved    = 0.0
                     for r in range(0,j+1):
                         # compute inverse of knot differences and save them into lower triangular part of ndu2
                         ndu2[j+1,r] = 1.0 / (right2[r] + left2[j-r])
                         # compute basis functions and save them into upper triangular part of ndu2
                         temp       = ndu2[r,j] * ndu2[j+1,r]
                         ndu2[r,j+1] = saved + right2[r] * temp
                         saved      = left2[j-r] * temp
                     ndu2[j+1,j+1] = saved	
               
                 # Compute derivatives in 2D output array 'ders2'
                 ders2[0,:] = ndu2[:,degree]
                 for r in range(0,degree+1):
                     s1 = 0
                     s2 = 1
                     a2[0,0] = 1.0
                     for k in range(1,nders+1):
                         d  = 0.0
                         rk = r-k
                         pk = degree-k
                         if r >= k:
                            a2[s2,0] = a2[s1,0] * ndu2[pk+1,rk]
                            d = a2[s2,0] * ndu2[rk,pk]
                         j1 = 1   if (rk  > -1 ) else -rk
                         j2 = k-1 if (r-1 <= pk) else degree-r
                         for ij in range(j1,j2+1):
                             a2[s2,ij] = (a2[s1,ij] - a2[s1,ij-1]) * ndu2[pk+1,rk+ij]
                         for ij in range(j1,j2+1):
                             d += a2[s2,ij]* ndu2[rk+ij,pk]
                         if r <= pk:
                            a2[s2,k] = - a2[s1,k-1] * ndu2[pk+1,r]
                            d += a2[s2,k] * ndu2[r,pk]
                         ders2[k,r] = d
                         j  = s1
                         s1 = s2
                         s2 = j
                 # ...first compute R1
                 ders2[0,:] = ders2[0,:] * omega_2[span-degree:span+1]
                 sum_basisy = sum(ders2[0,:])
                 ders2[0,:] = ders2[0,:]/sum_basisy
                 #------------------   
                 lcoeffs_v1[ : , : ]  =  vector_v1[span_5 : span_5+p1+1, span_6 : span_6+p2+1]
                 #...
                 x     = 0.0                    
                 for il_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        coef_v1   = lcoeffs_v1[il_1,il_2]
                        bi_0      = ders[0,il_1] * ders2[0,il_2]
                        # ...
                        x        +=  coef_v1*bi_0

                 lvalues_u[g1,g2] = x
          for il_1 in range(0, p1+1):
            for il_2 in range(0, p2+1):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        i_span_1 = spans_1[ie1, g1]
                        i_span_2 = spans_2[ie2, g2]
                        i1 = i_span_1 - p1 + il_1
                        i2 = i_span_2 - p2 + il_2
                        #...
                        bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                        #...
                        wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                        # ...Dirichlet boundary conditions
                        u     = lvalues_u[g1,g2]
                        #..
                        rhs[i1+p1,i2+p2] += bi_0 * u * wvol
    # ...    