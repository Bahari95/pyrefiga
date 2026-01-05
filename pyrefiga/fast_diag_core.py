# =========================================================================
def solve_unit_sylvester_system_2d(d1: 'float64[:]',
                                   d2: 'float64[:]',
                                   b: 'float64[:,:]',
                                   tau: 'float',
                                   x: 'float64[:,:]'):
    """
    Solves the linear system (D1 x I2 + I1 x D2 + tau I1 x I2) x = b
    """
    n1 = len(d1)
    n2 = len(d2)

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = b[i1,i2] / (d1[i1] + d2[i2] + tau)

# =========================================================================
def solve_unit_sylvester_system_3d(d1: 'float64[:]',
                                   d2: 'float64[:]',
                                   d3: 'float64[:]',
                                   b: 'float64[:,:,:]',
                                   tau: 'float',
                                   x: 'float64[:,:,:]'):
    """
    Solves the linear system (D1 x I2 x I3 + I1 x D2 x I3 + I1 x I2 x D3 + tau I1 x I2 x I3) x = b
    """
    n1 = len(d1)
    n2 = len(d2)
    n3 = len(d3)

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                x[i1, i2, i3] = b[i1, i2, i3] / (d1[i1] + d2[i2] + d3[i3] + tau)


# =========================================================================
def build_identity_mapping_2d(gx: 'float64[:]',
                                   gy: 'float64[:]',
                                   x: 'float64[:,:]',
                                   y: 'float64[:,:]'):
    """
    assemble identity mapping from greville points
    """
    n1 = len(gx)
    n2 = len(gy)

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = gx[i1]
            y[i1,i2] = gy[i2]

# =========================================================================
def build_identity_mapping_3d(gx: 'float64[:]',
                                   gy: 'float64[:]',
                                   gz: 'float64[:]',
                                   x: 'float64[:,:,:]',
                                   y: 'float64[:,:,:]',
                                   z: 'float64[:,:,:]'):
    """
    assemble identity mapping from greville points
    """
    n1 = len(gx)
    n2 = len(gy)
    n3 = len(gz)

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                x[i1, i2, i3] = gx[i1] 
                y[i1, i2, i3] = gy[i2] 
                z[i1, i2, i3] = gz[i3]