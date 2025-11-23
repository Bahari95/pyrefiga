"""
Module docstring for nurbs_utilities_core.py
This module provides utility functions to evaluate (rational) B-spline / NURBS
field values and their first derivatives on structured evaluation grids or on
arrays of mesh coordinates. The implementations compute non-rational B-spline
basis functions and their first derivatives using the standard triangular
ndu/au algorithm, convert them into rational (weighted) basis functions using
per-direction weight arrays (omega1, omega2, omega3), and then evaluate the
field and its derivatives by linear combination of control/coefficients uh.
The functions mutate the provided output array Q in-place; they do not return
new arrays. They are designed for 1D, 2D and 3D scalar fields (field value
and gradient components).
Common assumptions and conventions
- Knot vectors Tu, Tv, Tw are non-decreasing sequences (iterables of floats).
    For degree p and number of control points n, the knot vector length should
    equal n + p + 1.
- Degrees pu, pv, pw are non-negative integers.
- Control/solution coefficients uh are provided on a grid matching the number
    of control points implied by the knot vectors:
        * 1D: uh has length nu = len(Tu) - pu - 1
        * 2D: uh shape (nu, nv) where nu = len(Tu) - pu - 1, nv = len(Tv) - pv - 1
        * 3D: uh shape (nu, nv, nw) with analogous definitions
- Weight arrays omega1, omega2, omega3 contain positive weights associated
    with control points in each parametric direction. Their lengths should match
    nu, nv, nw respectively.
- The output array Q is expected to have enough columns / last-dimension size
    to hold the computed values and optionally the mesh coordinates (depending on
    the function variant):
        * sol_field_1D_meshes: Q shape (nx, >=3). Input x-coordinates are read
            from Q[:, 2]. After call Q[:, 0] = field value, Q[:, 1] = d/dx (value).
        * sol_field_2D_meshes: Q shape (nx, ny, >=5). Input coords read from
            Q[..., 3] (x) and Q[..., 4] (y). After call Q[..., 0] = field value,
            Q[..., 1] = d/dx, Q[..., 2] = d/dy.
        * sol_field_2D: Q shape (nx, ny, >=3). Input coords are provided by xs and ys
            arrays (1D arrays of length nx and ny). After call Q[..., 0] = field,
            Q[..., 1] = d/dx, Q[..., 2] = d/dy.
        * sol_field_3D_meshes: Q shape (nx, ny, nz, >=7). Input coords read from
            Q[..., 4] (x), Q[..., 5] (y), Q[..., 6] (z). After call,
            Q[..., 0] = field, Q[..., 1:4] = gradients (d/dx, d/dy, d/dz), and
            Q[..., 4:7] remain x,y,z.
        * sol_field_3D_mesh: same semantics as sol_field_3D_meshes (legacy naming),
            but the functions differ only in where they obtain the input coordinates:
            - sol_field_3D: uses xs, ys, zs arrays
            - sol_field_3D_mesh / sol_field_2D_meshes / sol_field_1D_meshes:
                read coordinates embedded inside Q
- All functions compute only up to the first derivative (nders = 1). The code
    multiplies derivative basis values by the degree to obtain the physical
    derivative with respect to the parametric coordinate, then converts to
    rational (weighted) derivatives using the quotient rule.
Function summaries
- sol_field_1D_meshes(nx, uh, Tu, pu, omega1, Q)
        Evaluate a 1D NURBS field and its first derivative at nx sample points
        whose x-coordinates are stored in Q[:, 2]. Fills Q[:, 0] with the field
        value and Q[:, 1] with d/dx. Uses knot vector Tu, degree pu, control
        coefficients uh (1D array length nu), and weights omega1 (length nu).
- sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, omega1, omega2, Q)
        Evaluate a 2D NURBS field and its first derivatives at a structured
        (nx x ny) grid where coordinates are stored inside Q[..., 3] and Q[..., 4]
        (x and y). Fills Q[..., 0] with the field value and Q[..., 1:3] with the
        derivatives (d/dx, d/dy). The control array uh is expected shape (nu, nv)
        and P is a copy of uh used for evaluation.
- sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, omega1, omega2, Q)
        Same as sol_field_2D_meshes but reads evaluation coordinates from separate
        1D arrays xs (length nx) and ys (length ny) instead of embedding them in Q.
- sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tw, pu, pv, pw,
                                omega1, omega2, omega3, Q)
        Evaluate a 3D NURBS scalar field and first derivatives on a structured
        grid defined by xs, ys, zs (1D coordinate arrays). The control array uh
        should have shape (nu, nv, nw). Results are stored in Q[..., 0:4] with
        order (value, d/dx, d/dy, d/dz). Coordinates are not read from Q.
- sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tw, pu, pv, pw,
                                        omega1, omega2, omega3, Q)
        Same as sol_field_3D but reads the evaluation coordinates from Q[..., 4:7]
        (x,y,z) and writes results into Q[..., 0:4]. This variant is intended
        for situations where Q already contains mesh coordinates.
Notes on numerical behavior and errors
- The algorithm performs binary search to find the knot span and computes
    basis functions via the standard stable recurrence. If knot vectors are
    malformed (e.g., repeated knots leading to zero denominators) a ZeroDivisionError
    or floating-point exception may occur. The functions do not validate all
    input shapes extensively; mismatched shapes may raise IndexError.
- The rational basis conversion divides by the weighted sum of basis
    function values (sum_basisx, sum_basisy, sum_basisz). If these sums are
    numerically zero due to weight/configuration issues, a division by zero
    will occur.
- The functions copy uh into a local array P before evaluation; for large
    control grids this duplicates memory temporarily.
Recommendations
- Ensure Tu, Tv, Tw and uh shapes are consistent before calling these
    routines.
- Pre-allocate Q with the expected shape and place coordinates in the expected
    positions for the "_meshes" variants.
- For production use, consider adding explicit shape/consistency checks,
    vectorizing common operations, or replacing the bespoke basis-function
    code with a numerically robust NURBS library if available.
References
- The basis-function and derivative algorithm implemented here follows the
    standard recurrence/triangular "ndu" approach commonly described in
    NURBS references (e.g., the standard algorithm for B-spline basis
    evaluation and derivatives).

    Authors: M. Bahari
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_1D_meshes(nx:'int', uh:'float[:]', Tu:'float[:]', pu:'int', omega1:'float[:]', Q:'float[:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #...
    for i_x in range(nx):          
            x = Q[i_x,2]
            # ... for x ----
            #--Computes All basis in a new points              
            #..
            xq         = x
            dersu[:,:] = 0.
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = pu
            high = len(Tu)-1-pu
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= Tu[low ]: 
                   span = low
            elif xq >= Tu[high]: 
                  span = high-1
            else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
            ndu[0,0] = 1.0
            for j in range(0,pu):
                leftu [j] = xq - Tu[span-j]
                rightu[j] = Tu[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + rightu[r] * temp
                    saved      = leftu[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            dersu[0,:] = ndu[:,pu]
            for r in range(0,pu+1):
                s1 = 0
                s2 = 1
                au[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = pu-k
                    if r >= k:
                        au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                        d = au[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else pu-r
                    for ij in range(j1,j2+1):
                        au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += au[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                        au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                        d += au[s2,k] * ndu[r,pk]
                    dersu[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            # Multiply derivatives by correct factors
            r = pu
            dersu[1,:] = dersu[1,:] * r
            basis_x = dersu
            span_u  = span
            #...
            basis_x[0,:]  = basis_x[0,:] * omega1[span_u-pu:span_u+1]
            sum_basisx    = sum(basis_x[0,:])
            basis_x[0,:]  = basis_x[0,:]/sum_basisx
            #...                        
            basis_x[1,:]  = basis_x[1,:] * omega1[span_u-pu:span_u+1]
            sum_dbasisx   = sum(basis_x[1,:])
            basis_x[1,:]  = (basis_x[1,:] - basis_x[0,:]*sum_dbasisx)
            basis_x[1,:] /= sum_basisx
            # ... start assembling
            c       = 0.
            cx      = 0.
            for ku in range(0, pu+1):
                c  += basis_x[0,ku]*uh[span_u-pu+ku]
                cx += basis_x[1,ku]*uh[span_u-pu+ku]
            #..
            Q[i_x, 0]   = c
            Q[i_x, 1]   = cx
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_2D_meshes(nx:'int', ny:'int', uh:'float[:,:]', Tu:'float[:]', Tv:'float[:]', pu:'int', pv:'int', omega1:'float[:]', omega2:'float[:]', Q:'float[:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          
              x = Q[i_x,j_y,3]
              y = Q[i_x,j_y,4]
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r          = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y    = dersv
              span_v     = span             
              #...
              basis_x[0,:]  = basis_x[0,:] * omega1[span_u-pu:span_u+1]
              sum_basisx    = sum(basis_x[0,:])
              basis_x[0,:]  = basis_x[0,:]/sum_basisx
              basis_y[0,:]  = basis_y[0,:] * omega2[span_v-pv:span_v+1]              
              sum_basisy    = sum(basis_y[0,:])
              basis_y[0,:]  = basis_y[0,:]/sum_basisy
              #..
              basis_x[1,:]  = basis_x[1,:] * omega1[span_u-pu:span_u+1]
              sum_dbasisx   = sum(basis_x[1,:])
              basis_y[1,:]  = basis_y[1,:] * omega2[span_v-pv:span_v+1]
              sum_dbasisy   = sum(basis_y[1,:])
              #...                        
              basis_x[1,:]  = (basis_x[1,:] - basis_x[0,:]*sum_dbasisx)
              basis_x[1,:] /= sum_basisx
              basis_y[1,:]  = (basis_y[1,:] - basis_y[0,:]*sum_dbasisy)
              basis_y[1,:] /= sum_basisy
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += basis_x[0,ku]*basis_y[0,kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cx += basis_x[1,ku]*basis_y[0,kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cy += basis_x[0,ku]*basis_y[1,kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              Q[i_x, j_y, 1]   = cx
              Q[i_x, j_y, 2]   = cy
              

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_2D(nx:'int', ny:'int', xs:'float[:]', ys:'float[:]', uh:'float[:,:]', Tu:'float[:]', Tv:'float[:]', pu:'int', pv:'int', omega1:'float[:]', omega2:'float[:]', Q:'float[:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          
              x = xs[i_x]
              y = ys[j_y]
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #...
              basis_x[0,:]  = basis_x[0,:] * omega1[span_u-pu:span_u+1]
              sum_basisx    = sum(basis_x[0,:])
              basis_x[0,:]  = basis_x[0,:]/sum_basisx
              basis_y[0,:]  = basis_y[0,:] * omega2[span_v-pv:span_v+1]              
              sum_basisy    = sum(basis_y[0,:])
              basis_y[0,:]  = basis_y[0,:]/sum_basisy
              #..
              basis_x[1,:]  = basis_x[1,:] * omega1[span_u-pu:span_u+1]
              sum_dbasisx   = sum(basis_x[1,:])
              basis_y[1,:]  = basis_y[1,:] * omega2[span_v-pv:span_v+1]
              sum_dbasisy   = sum(basis_y[1,:])
              #...                       
              basis_x[1,:]  = (basis_x[1,:] - basis_x[0,:]*sum_dbasisx)
              basis_x[1,:] /= sum_basisx
              basis_y[1,:]  = (basis_y[1,:] - basis_y[0,:]*sum_dbasisy)
              basis_y[1,:] /= sum_basisy
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += basis_x[0,ku]*basis_y[0,kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cx += basis_x[1,ku]*basis_y[0,kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cy += basis_x[0,ku]*basis_y[1,kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              Q[i_x, j_y, 1]   = cx
              Q[i_x, j_y, 2]   = cy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_3D(nx:'int', ny:'int', nz:'int', xs:'float[:]', ys:'float[:]', zs:'float[:]', uh:'float[:,:,:]', Tu:'float[:]', Tv:'float[:]', Tw:'float[:]', pu:'int', pv:'int', pw:'int',  omega1:'float[:]', omega2:'float[:]',  omega3:'float[:]', Q:'float[:,:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty
    from numpy import linspace

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    nw = len(Tw) - pw - 1

    P = zeros((nu, nv, nw))
    
    for i in range(nu):  
       for j in range(nv):
          for k in range(nw):
             P[i, j, k] = uh[i, j, k]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    #
    leftw      = empty( pw )
    rightw     = empty( pw )
    ndw        = empty( (pw+1, pw+1) )
    aw         = empty( (       2, pw+1) )
    dersw      = zeros( (     nders+1, pw+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          for k_z in range(nz):
          
              x = xs[i_x]
              y = ys[j_y]
              z = zs[k_z]    
              
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #basis_z = basis_funs_all_ders( Tw, pw, z, span_w, 1 )
              #--Computes All basis in a new points
              xq         = z
              dersw[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pw
              high = len(Tw)-1-pw
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tw[low ]: 
                   span = low
              elif xq >= Tw[high]: 
                  span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tw[span] or xq >= Tw[span+1]:
                   if xq < Tw[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndw[0,0] = 1.0
              for j in range(0,pw):
                  leftw [j] = xq - Tw[span-j]
                  rightw[j] = Tw[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndw[j+1,r] = 1.0 / (rightw[r] + leftw[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndw[r,j] * ndw[j+1,r]
                      ndw[r,j+1] = saved + rightw[r] * temp
                      saved      = leftw[j-r] * temp
                  ndw[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersw[0,:] = ndw[:,pw]
              for r in range(0,pw+1):
                  s1 = 0
                  s2 = 1
                  aw[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pw-k
                      if r >= k:
                         aw[s2,0] = aw[s1,0] * ndw[pk+1,rk]
                         d = aw[s2,0] * ndw[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pw-r
                      for ij in range(j1,j2+1):
                          aw[s2,ij] = (aw[s1,ij] - aw[s1,ij-1]) * ndw[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += aw[s2,ij]* ndw[rk+ij,pk]
                      if r <= pk:
                         aw[s2,k] = - aw[s1,k-1] * ndw[pk+1,r]
                         d += aw[s2,k] * ndw[r,pk]
                      dersw[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pw
              dersw[1,:] = dersw[1,:] * r
              basis_z = dersw
              span_w  = span
              #...
              basis_x[0,:]  = basis_x[0,:] * omega1[span_u-pu:span_u+1]
              sum_basisx    = sum(basis_x[0,:])
              basis_x[0,:]  = basis_x[0,:]/sum_basisx
              basis_y[0,:]  = basis_y[0,:] * omega2[span_v-pv:span_v+1]              
              sum_basisy    = sum(basis_y[0,:])
              basis_y[0,:]  = basis_y[0,:]/sum_basisy
              basis_z[0,:] = basis_z[0,:] * omega3[span_v-pv:span_v+1]
              sum_basisz    = sum(basis_z[0,:])
              basis_z[0,:]  = basis_z[0,:]/sum_basisz
              #..
              basis_x[1,:]  = basis_x[1,:] * omega1[span_u-pu:span_u+1]
              sum_dbasisx   = sum(basis_x[1,:])
              basis_y[1,:]  = basis_y[1,:] * omega2[span_v-pv:span_v+1]
              sum_dbasisy   = sum(basis_y[1,:])
              basis_z[1,:]  = basis_z[1,:] * omega3[span_v-pv:span_v+1]
              sum_dbasisz   = sum(basis_z[1,:])
              #...                       
              basis_x[1,:]  = (basis_x[1,:] - basis_x[0,:]*sum_dbasisx)
              basis_x[1,:] /= sum_basisx
              basis_y[1,:]  = (basis_y[1,:] - basis_y[0,:]*sum_dbasisy)
              basis_y[1,:] /= sum_basisy
              basis_z[1,:]  = (basis_z[1,:] - basis_z[0,:]*sum_dbasisz)
              basis_z[1,:] /= sum_basisz
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              cz      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                    for kw in range(0, pw+1):
                      c  += basis_x[0,ku]*basis_y[0,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cx += basis_x[1,ku]*basis_y[0,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cy += basis_x[0,ku]*basis_y[1,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cz += basis_x[0,ku]*basis_y[0,kv]*basis_z[1,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
              #..
              Q[i_x, j_y, k_z,0]   = c
              Q[i_x, j_y, k_z,1]   = cx
              Q[i_x, j_y, k_z,2]   = cy
              Q[i_x, j_y, k_z,3]   = cz
              Q[i_x, j_y, k_z,4]   = x
              Q[i_x, j_y, k_z,5]   = y
              Q[i_x, j_y, k_z,6]   = z
              
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_3D_mesh(nx:'int', ny:'int', nz:'int', uh:'float[:,:,:]', Tu:'float[:]', Tv:'float[:]', Tw:'float[:]', pu:'int', pv:'int', pw:'int',  omega1:'float[:]', omega2:'float[:]',  omega3:'float[:]', Q:'float[:,:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty
    from numpy import linspace

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    nw = len(Tw) - pw - 1

    P = zeros((nu, nv, nw))
    
    for i in range(nu):  
       for j in range(nv):
          for k in range(nw):
             P[i, j, k] = uh[i, j, k]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    #
    leftw      = empty( pw )
    rightw     = empty( pw )
    ndw        = empty( (pw+1, pw+1) )
    aw         = empty( (       2, pw+1) )
    dersw      = zeros( (     nders+1, pw+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          for k_z in range(nz):
          
              x = Q[i_x, j_y, k_z,4]
              y = Q[i_x, j_y, k_z,5]
              z = Q[i_x, j_y, k_z,6]   
              
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #basis_z = basis_funs_all_ders( Tw, pw, z, span_w, 1 )
              #--Computes All basis in a new points
              xq         = z
              dersw[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pw
              high = len(Tw)-1-pw
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tw[low ]: 
                   span = low
              elif xq >= Tw[high]: 
                  span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tw[span] or xq >= Tw[span+1]:
                   if xq < Tw[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndw[0,0] = 1.0
              for j in range(0,pw):
                  leftw [j] = xq - Tw[span-j]
                  rightw[j] = Tw[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndw[j+1,r] = 1.0 / (rightw[r] + leftw[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndw[r,j] * ndw[j+1,r]
                      ndw[r,j+1] = saved + rightw[r] * temp
                      saved      = leftw[j-r] * temp
                  ndw[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersw[0,:] = ndw[:,pw]
              for r in range(0,pw+1):
                  s1 = 0
                  s2 = 1
                  aw[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pw-k
                      if r >= k:
                         aw[s2,0] = aw[s1,0] * ndw[pk+1,rk]
                         d = aw[s2,0] * ndw[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pw-r
                      for ij in range(j1,j2+1):
                          aw[s2,ij] = (aw[s1,ij] - aw[s1,ij-1]) * ndw[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += aw[s2,ij]* ndw[rk+ij,pk]
                      if r <= pk:
                         aw[s2,k] = - aw[s1,k-1] * ndw[pk+1,r]
                         d += aw[s2,k] * ndw[r,pk]
                      dersw[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pw
              dersw[1,:] = dersw[1,:] * r
              basis_z = dersw
              span_w  = span
              #...
              basis_x[0,:]  = basis_x[0,:] * omega1[span_u-pu:span_u+1]
              sum_basisx    = sum(basis_x[0,:])
              basis_x[0,:]  = basis_x[0,:]/sum_basisx
              basis_y[0,:]  = basis_y[0,:] * omega2[span_v-pv:span_v+1]              
              sum_basisy    = sum(basis_y[0,:])
              basis_y[0,:]  = basis_y[0,:]/sum_basisy
              basis_z[0,:] = basis_z[0,:] * omega3[span_v-pv:span_v+1]
              sum_basisz    = sum(basis_z[0,:])
              basis_z[0,:]  = basis_z[0,:]/sum_basisz
              #..
              basis_x[1,:]  = basis_x[1,:] * omega1[span_u-pu:span_u+1]
              sum_dbasisx   = sum(basis_x[1,:])
              basis_y[1,:]  = basis_y[1,:] * omega2[span_v-pv:span_v+1]
              sum_dbasisy   = sum(basis_y[1,:])
              basis_z[1,:]  = basis_z[1,:] * omega3[span_v-pv:span_v+1]
              sum_dbasisz   = sum(basis_z[1,:])
              #...                       
              basis_x[1,:]  = (basis_x[1,:] - basis_x[0,:]*sum_dbasisx)
              basis_x[1,:] /= sum_basisx
              basis_y[1,:]  = (basis_y[1,:] - basis_y[0,:]*sum_dbasisy)
              basis_y[1,:] /= sum_basisy
              basis_z[1,:]  = (basis_z[1,:] - basis_z[0,:]*sum_dbasisz)
              basis_z[1,:] /= sum_basisz
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              cz      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                    for kw in range(0, pw+1):
                      c  += basis_x[0,ku]*basis_y[0,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cx += basis_x[1,ku]*basis_y[0,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cy += basis_x[0,ku]*basis_y[1,kv]*basis_z[0,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cz += basis_x[0,ku]*basis_y[0,kv]*basis_z[1,kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
              #..
              Q[i_x, j_y, k_z,0]   = c
              Q[i_x, j_y, k_z,1]   = cx
              Q[i_x, j_y, k_z,2]   = cy
              Q[i_x, j_y, k_z,3]   = cz
