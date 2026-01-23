"""
This script provides tools for working with B-spline solutions in one dimension.

Features:
- Plotting a B-spline solution in 1D.
- Construction of the prolongation matrix between hierarchical spaces.
- save_geometry_to_xml`: Saves a computed geometry mapping to a .xml file.
- pyref_patch`: Extracts a geometry mapping from a .xml file.

@author: Mustapha Bahari
"""
import numpy            as     np
from   functools        import reduce
from   matplotlib       import pyplot as plt
from   scipy.sparse     import kron, csr_matrix
from   .cad             import point_on_bspline_curve
from   .cad             import point_on_bspline_surface
from   .bsplines        import hrefinement_matrix
from   .bsplines        import greville
from   .linalg          import StencilVector
from   .spaces          import SplineSpace
from   .spaces          import TensorSpace
from   .results_f90     import least_square_Bspline
from   .results_f90     import pyccel_sol_field_2d
from   .results_f90     import sol_field_NURBS_2d
from   .results_f90     import least_square_NURBspline

__all__ = ['plot_field_1d', 
           'prolongation_matrix',
           'save_geometry_to_xml',
           'compute_eoc'
           'pyref_patch',
           'pyref_multpatch',
           'pyrefInterface',
           'load_xml']

# ==========================================================
def plot_field_1d(knots, degree, u, nx=101, color='b', xmin = None, xmax = None, label = None):
    n = len(knots) - degree - 1

    if xmin is None :
        xmin = knots[degree]
    if xmax is None :
        xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    if label is not None :
        plt.plot(xs, Q[:,0], label = label)
    else :
        plt.plot(xs, Q[:,0])

# ==========================================================
def prolongation_matrix(VH, Vh):
    # TODO not working for duplicated internal knots : to be fixed soon
    # ...
    assert VH.dim == Vh.dim, "Spaces must have the same number of dimensions"
    for i in range(VH.dim):
        knots_VH = set(VH.knots[i])
        knots_Vh = set(Vh.knots[i])
        assert knots_VH.issubset(knots_Vh), f"Knots in VH not included in Vh at dimension {i}"

    # ...
    mats = []
    for Wh, WH in zip(Vh.spaces, VH.spaces):
        ths = Wh.knots
        tHs = WH.knots
        ts = set(ths) - set(tHs)
        ts = np.array(list(ts))

        M = hrefinement_matrix( ts, Wh.degree, tHs )
        mats.append(csr_matrix(M))
    # ...

    M = reduce(kron, (m for m in mats))

    return M

#====================================
# ...  Identity B-spline mapping
#======================================
def identity_bspline_mapping(V):
    """
    Construct control points for the identity B-spline mapping F(xi,eta)=(xi,eta).

    Parameters
    ----------
    V   : TensorSpace or SplineSpace (1D)

    Returns
    -------
    P : ndarray (n_xi, n_eta, 2)
        Control points of the identity mapping
    """
    from .               import fast_diag_core as core
    if isinstance(V, TensorSpace):
        if V.dim == 2:
            gx      = greville(V.knots[0], V.degree[0], V.spaces[0].periodic)
            gy      = greville(V.knots[1], V.degree[1], V.spaces[1].periodic)

            nx, ny  = len(gx), len(gy)
            Px      = np.zeros((nx, ny))
            Py      = np.zeros((nx, ny))
            # ....
            core.build_identity_mapping_2d(gx, gy, Px, Py)
            return Px, Py
        elif V.dim == 3:
            gx         = greville(V.knots[0], V.degree[0], V.spaces[0].periodic)
            gy         = greville(V.knots[1], V.degree[1], V.spaces[1].periodic)
            gz         = greville(V.knots[2], V.degree[2], V.spaces[2].periodic)

            nx, ny, nz = len(gx), len(gy), len(gz)
            Px         = np.zeros((nx, ny, nz))
            Py         = np.zeros((nx, ny, nz))
            Pz         = np.zeros((nx, ny, nz))
            # ....
            core.build_identity_mapping_3d(gx, gy, gz, Px, Py, Pz)
            return Px, Py, Pz
    elif isinstance(V, SplineSpace):
        gx      = greville(V.knots, V.degree, V.periodic)
        return gx
    else:
        raise NotImplementedError('Please give V as TensorSpace or SplineSpace in 1D case!')  
#========================================================================
# ... computes the order of convergence
#========================================================================
def compute_eoc(err):
    '''
    compute_eoc: computes the order of convergence
    err: list of errors
    '''
    err       = np.asarray(err, dtype=float)
    numRefine = err.shape[0]
    if numRefine > 1:
        eoc       = np.zeros(numRefine)
        # ...
        eoc[1:] = np.log(err[:-1] / err[1:])/ np.log(2.)
        # eoc[1:] = np.log(err[:-1] / err[1:]) / np.log(size_mesh[1:] / size_mesh[:-1])
        return eoc
    else:
        return [0.]

#========================================================================
# ... build Dirichlet in two dimensions from analytic form
#========================================================================
def build_dirichlet(V, f, map = None, admap = None, refinN = 10, Boundaries = None):
    '''
    V    : FE space
    f[0] : on the left
    f[1] : on the right
    f[2] : on the bottom
    f[3] : on the top
    map = (x,y, V) : control points and associated space
    admap = (x, V1, y, V2) control points and associated space
    refinN : number of refinement for least square projection
    Boundaries : list of Boundaries to apply Dirichlet BCs on them
                                                 ___4___
                                                |       |
                                              1 |       | 2
                                                |_______|
                                                    3
    Returns
    -------
    u_d : StencilVector
        Dirichlet vector
    '''
    assert all(V.nelements[i] > 1 for i in range(V.dim)), \
       "Please refine space at least one time, works for nelements > 1"

    if map is None:
        pass
    elif len(map) == V.dim:
        map = [*map, V]

    if Boundaries is None:
        Boundaries = [1,2,3,4]

    #... check    
    if not isinstance(Boundaries, (list, np.ndarray)) \
    or not all(isinstance(int(b), int) for b in Boundaries):
        raise TypeError(
            "Boundaries must be a list or array of integers like [1, 2, 3, 4] or [1,3,4] ..."
        )

    #... 
    boundary_map = {
        1: (0, slice(None)),
        2: (-1, slice(None)),
        3: (slice(None), 0),
        4: (slice(None), -1),
    }
    # ...
    Space_map = {
        1: 1,
        2: 1,
        3: 0,
        4: 0,
    }
    if len(f) > 1 :
        fx0      = lambda   y :  eval(f[0])
        fx1      = lambda   y :  eval(f[1])
        fy0      = lambda x   :  eval(f[2])
        fy1      = lambda x   :  eval(f[3])
    elif map is None:
        xi       = V.grid[0]
        yi       = V.grid[1]
        sol      = lambda x,y :  eval(f[0]) 
        fx0      = lambda   y :  sol(xi[0],y)
        fx1      = lambda   y :  sol(xi[-1],y) 
        fy0      = lambda x   :  sol(x,yi[0])
        fy1      = lambda x   :  sol(x,yi[-1])
    elif len(map)==3 :
        #surface 2D TODO
        fx0      = lambda x,y :  eval(f[0])
        fx1      = lambda x,y :  eval(f[0])
        fy0      = lambda x,y :  eval(f[0])
        fy1      = lambda x,y :  eval(f[0]) 
    else :
        #surface 3D TODO
        fx0      = lambda x,y,z :  eval(f[0])
        fx1      = lambda x,y,z :  eval(f[0])
        fy0      = lambda x,y,z :  eval(f[0])
        fy1      = lambda x,y,z :  eval(f[0])     
    f_dir        = [fx0, fx1, fy0, fy1]
    u_d          = StencilVector(V.vector_space)
    x_d          = np.zeros(V.nbasis)
    n_dir        = (V.nbasis[0]*refinN+refinN, V.nbasis[1]*refinN+refinN)
    if V.omega[0] is None :
        #... B-spline space
        if V.dim == 2:
            if map is None:
                #------------------------------
                #.. In the parametric domain
                for i in range(len(Boundaries)):
                    x_d[boundary_map[Boundaries[i]]] = least_square_Bspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], f_dir[Boundaries[i]-1])
            elif admap is None :
                #-------------------------------------------------
                #.. In the phyisacl domain without adaptive mapping
                if len(map) == 3:
                    #planar mapping
                    if map[2].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[2].knots, map[2].degree)[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[2].knots, map[2].degree)[0]
                    else:
                        sX           = sol_field_NURBS_2d(n_dir,  map[0] , map[2].omega, map[2].knots, map[2].degree)[0]
                        sY           = sol_field_NURBS_2d(n_dir,  map[1] , map[2].omega, map[2].knots, map[2].degree)[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_Bspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]]))
                else:
                    #3D mapping
                    if map[3].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[3].knots, map[3].degree)[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[3].knots, map[3].degree)[0]
                        sZ           = pyccel_sol_field_2d(n_dir,  map[2] , map[3].knots, map[3].degree)[0]
                    else:
                        sX           = sol_field_NURBS_2d(n_dir,  map[0] , map[3].omega, map[3].knots, map[3].degree)[0]
                        sY           = sol_field_NURBS_2d(n_dir,  map[1] , map[3].omega, map[3].knots, map[3].degree)[0]
                        sZ           = sol_field_NURBS_2d(n_dir,  map[2] , map[3].omega, map[3].knots, map[3].degree)[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_Bspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]], sZ[boundary_map[Boundaries[i]]]  ))
            else :
                #-----------------------------------------------
                #.. In the phyisacl domain with adaptive mapping               
                if admap[2].omega[0] is None:
                    Xmae         = pyccel_sol_field_2d(n_dir,  admap[0] , admap[2].knots, admap[2].degree)[0]
                    Ymae         = pyccel_sol_field_2d(n_dir,  admap[1] , admap[3].knots, admap[3].degree)[0]
                else:
                    Xmae         = sol_field_NURBS_2d(n_dir,  admap[0] , admap[2].omega, admap[2].knots, admap[2].degree)[0]
                    Ymae         = sol_field_NURBS_2d(n_dir,  admap[1] , admap[3].omega, admap[3].knots, admap[3].degree)[0]
                if len(map) == 3:
                    #planar mapping
                    if map[2].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]
                    else :
                        sX           = sol_field_NURBS_2d( None, map[0], map[2].omega, map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]
                        sY           = sol_field_NURBS_2d( None, map[1], map[2].omega, map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_Bspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]]))
                else:
                    #3D mapping
                    if map[3].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sZ           = pyccel_sol_field_2d(n_dir,  map[2] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                    else :
                        sX           = sol_field_NURBS_2d( None, map[0], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sY           = sol_field_NURBS_2d( None, map[1], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sZ           = sol_field_NURBS_2d( None, map[2], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_Bspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]], sZ[boundary_map[Boundaries[i]]]))
        if V.dim == 3 :
            raise NotImplementedError("3D Dirichlet boundary conditions are not yet implemented. nd: Use L2 projection using fast diagonalization.")
    else :
        if V.dim == 2:
            if map is None:
                #------------------------------
                #.. In the parametric domain
                for i in range(len(Boundaries)):
                    x_d[boundary_map[Boundaries[i]]] = least_square_NURBspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                            V.omega[Space_map[Boundaries[i]]],
                                                                            f_dir[Boundaries[i]-1])

            elif admap is None :
                #-------------------------------------------------
                #.. In the phyisacl domain without adaptive mapping
                if len(map) == 3:
                    #planar mapping
                    if map[2].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[2].knots, map[2].degree)[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[2].knots, map[2].degree)[0]
                    else:
                        sX           = sol_field_NURBS_2d(n_dir,  map[0] , map[2].omega, map[2].knots, map[2].degree)[0]
                        sY           = sol_field_NURBS_2d(n_dir,  map[1] , map[2].omega, map[2].knots, map[2].degree)[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_NURBspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                V.omega[Space_map[Boundaries[i]]],
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]]))
                else:
                    #3D mapping
                    if map[3].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[3].knots, map[3].degree)[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[3].knots, map[3].degree)[0]
                        sZ           = pyccel_sol_field_2d(n_dir,  map[2] , map[3].knots, map[3].degree)[0]
                    else:
                        sX           = sol_field_NURBS_2d(n_dir,  map[0] , map[3].omega, map[3].knots, map[3].degree)[0]
                        sY           = sol_field_NURBS_2d(n_dir,  map[1] , map[3].omega, map[3].knots, map[3].degree)[0]
                        sZ           = sol_field_NURBS_2d(n_dir,  map[2] , map[3].omega, map[3].knots, map[3].degree)[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_NURBspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                V.omega[Space_map[Boundaries[i]]],
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]], sZ[boundary_map[Boundaries[i]]]))

            else :
                #-----------------------------------------------
                #.. In the phyisacl domain with adaptive mapping               
                if admap[2].omega[0] is None:
                    Xmae         = pyccel_sol_field_2d(n_dir,  admap[0] , admap[2].knots, admap[2].degree)[0]
                    Ymae         = pyccel_sol_field_2d(n_dir,  admap[1] , admap[3].knots, admap[3].degree)[0]
                else:
                    Xmae         = sol_field_NURBS_2d(n_dir,  admap[0] , admap[2].omega, admap[2].knots, admap[2].degree)[0]
                    Ymae         = sol_field_NURBS_2d(n_dir,  admap[1] , admap[3].omega, admap[3].knots, admap[3].degree)[0]
                if len(map) == 3:
                    if map[2].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[2].knots, map[2].degree, mesh=(Xmae, Ymae))[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[2].knots, map[2].degree, mesh=(Xmae, Ymae))[0]
                    else :
                        sX           = sol_field_NURBS_2d( None, map[0], map[2].omega, map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]
                        sY           = sol_field_NURBS_2d( None, map[1], map[2].omega, map[2].knots, map[2].degree, mesh = (Xmae, Ymae))[0]

                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_NURBspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                V.omega[Space_map[Boundaries[i]]],
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]]))
                else:
                    #3D mapping
                    if map[3].omega[0] is None :
                        sX           = pyccel_sol_field_2d(n_dir,  map[0] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sY           = pyccel_sol_field_2d(n_dir,  map[1] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sZ           = pyccel_sol_field_2d(n_dir,  map[2] , map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                    else :
                        sX           = sol_field_NURBS_2d( None, map[0], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sY           = sol_field_NURBS_2d( None, map[1], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                        sZ           = sol_field_NURBS_2d( None, map[2], map[3].omega, map[3].knots, map[3].degree, mesh = (Xmae, Ymae))[0]
                    for i in range(len(Boundaries)):
                        x_d[boundary_map[Boundaries[i]]] = least_square_NURBspline(V.degree[Space_map[Boundaries[i]]], V.knots[Space_map[Boundaries[i]]], 
                                                                                V.omega[Space_map[Boundaries[i]]],
                                                                                f_dir[Boundaries[i]-1](sX[boundary_map[Boundaries[i]]], sY[boundary_map[Boundaries[i]]], sZ[boundary_map[Boundaries[i]]]))

        if V.dim == 3 :
            raise NotImplementedError("3D Dirichlet boundary conditions are not yet implemented. nd: Use L2 projection using fast diagonalization.")
    u_d.from_array(V, x_d)
    return x_d, u_d

#========================================================================
# ... save geometry patch from a given mapping into cml file
#========================================================================
import xml.etree.ElementTree as ET
import numpy as np
def save_geometry_to_xml(V, Gmap, name = 'Geometry', locname = None):
    """
    save_geometry_to_xml : save the coefficients table, knots table, and degree in an XML file.
    """
    if locname is None :
        filename = f'./figs/'+name+'.xml'
    else :
        filename = locname+'.xml'
    # Root element
    root = ET.Element('xml')
    root.text  = '\n'
    if isinstance(Gmap, (list, tuple)):
        for pid, gmap in enumerate(Gmap):
            geom = ET.SubElement(root, 'Geometry', type='TensorNurbs2', id=str(pid))
            geom.text = '\n'

            outer = ET.SubElement(geom, 'Basis', type='TensorNurbsBasis2'); outer.text = '\n'
            inner = ET.SubElement(outer, 'Basis', type='TensorBSplineBasis2'); inner.text = '\n'

            # mêmes nœuds / degrés pour chaque patch
            for d in range(2):
                b = ET.SubElement(inner, 'Basis', type='BSplineBasis', index=str(d)); b.text = '\n'
                kv = ET.SubElement(b, 'KnotVector', degree=str(V.degree[d]))
                kv.text = '\n' + ' '.join(map(str, V.knots[d])) + '\n'
                b.tail = '\n'

            co = ET.SubElement(inner, 'coefs', geoDim='2')
            co.text = '\n' + '\n'.join(
                ' '.join(f'{float(v):.6f}' for v in row) for row in gmap
            ) + '\n'
            co.tail = '\n'            

            inner.tail = '\n'; outer.tail = '\n'; geom.tail = '\n'

        # ---------- MultiPatch ----------
        mp = ET.SubElement(root, 'MultiPatch', parDim='2', id=str(len(Gmap))); mp.text = '\n'
        pr = ET.SubElement(mp, 'patches', type='id_range')
        pr.text = f'\n0 {len(Gmap)-1}\n'
        bd = ET.SubElement(mp, 'boundary')
        bd.text = '\n' + ''.join(f'  {i} 1\n  {i} 2\n  {i} 3\n  {i} 4\n' for i in range(len(Gmap)))
        bd.tail = '\n'

    else :
        # Geometry element
        geometry    = ET.SubElement(root, 'Geometry', type='TensorNurbs2', id='0')
        geometry.text = '\n'
        basis_outer = ET.SubElement(geometry, 'Basis', type='TensorNurbsBasis2')
        basis_outer.text = '\n'
        basis_inner = ET.SubElement(basis_outer, 'Basis', type='TensorBSplineBasis2')
        basis_inner.text = '\n'

        # Add basis elements
        for i in range(2):
            basis            = ET.SubElement(basis_inner, 'Basis', type='BSplineBasis', index=str(i))
            basis.text       = '\n'
            knot_vector      = ET.SubElement(basis, 'KnotVector', degree=str(V.degree[0]))
            knot_vector.text = '\n' + ' '.join(map(str, V.knots[i])) + '\n'
        
        # Add coefficients (control points)
        coefs      = ET.SubElement(basis_inner, 'coefs', geoDim='2')
        coefs.text = '\n' + '\n'.join(' '.join(f'{v:.6f}' for v in row) for row in Gmap) + '\n'

        # Close inner Basis element properly
        basis_inner.tail = '\n'
        basis_outer.tail = '\n'

        # MultiPatch element
        multipatch   = ET.SubElement(root, 'MultiPatch', parDim='2', id='1')
        multipatch.text = '\n'
        patches      = ET.SubElement(multipatch, 'patches', type='id_range')
        patches.text = '\n0 0\n'
        
        # Boundary conditions
        boundary      = ET.SubElement(multipatch, 'boundary')
        boundary.text = '\n  0 1\n  0 2\n  0 3\n  0 4\n '
        boundary.tail = '\n'
    
    # Convert to XML string with declaration
    xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')
    xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' \
                 '<!--This file was created by bahari95/pyrefiga -->\n' \
                 '<!--Geometry in two dimensions -->\n' + xml_string
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    
    print(f"File saved as {filename}")

#========================================================================
#... class patches from xml files : one patch
# TODO doesn"t suuport multipatches 
#========================================================================
class pyref_patch:
    """
    pyref_patch : extracts the coefficients table, knots table, and degree from an XML file based on a given id.
    Parameters
    ----------
    filename : str
        The path to the XML file containing the geometry data.
    element_id : str
        The id of the geometry element to extract.
    nurbs : bool, optional
        If True, weights will be extracted; otherwise, weights default to ones. Default is False.
    Returns
    -------
    An object with attributes:
    - knots_data : list of lists
        The knot vectors for each dimension.
    - degree_data : list
        The degrees for each dimension.
    - _coefs : np.ndarray
        The control points reshaped according to the number of basis functions.
    - _grids : list of lists
        The grid points for each dimension. 
    """
    def __init__(self, filename, element_id, nurbs = True, Dirichlet_all = True):
      #print("""Initialize with the XML filename.""", filename)
      root            = ET.parse(filename).getroot()
      """Retrieve coefs table, knots table, and degree for a given id."""
      # Find the Geometry element by id
      pyref_patch = root.find(f".//*[@id='{element_id}']")        
      if pyref_patch is None:
         raise RuntimeError(f"No element found with id {element_id}")

      # Extract knots data and degree
      knots_data  = []
      degree_data = []
      for basis in pyref_patch.findall(".//Basis[@type='BSplineBasis']"):
         knot_vector = basis.find("KnotVector")
         if knot_vector is not None:
               degree_data.append(int(knot_vector.get("degree", -1)))  # Default to -1 if not found
               knots = list(map(float, knot_vector.text.strip().split()))
               knots_data.append(knots)
      #....dimension
      dim              = len(knots_data)
      #....number of basis functions
      _nbasis           = [len(knots_data[n]) - degree_data[n]-1 for n in range(dim)]
      # Extract coefs data
      coefs_element = pyref_patch.find(".//coefs")
      geo_dim       = int(coefs_element.attrib.get("geoDim")) if coefs_element is not None else None
      coefs_data    = None
      if coefs_element is not None:
         coefs_text = coefs_element.text.strip()
         coefs_data = np.array([
               list(map(float, line.split())) for line in coefs_text.split("\n")
         ])
      # Extract weights data or default to ones
      weights_element = pyref_patch.find(".//weights")
      if weights_element is not None:
          nurbs  = True
          weights_text = weights_element.text.strip()
          weights_data = np.array([
              float(w) for line in weights_text.split("\n") for w in line.strip().split() if w
          ])
      else:
          weights_data = np.ones(len(coefs_data[:,0]))
      # ...
      boundaries  = [1,2,3,4]
      if Dirichlet_all is False:
          boundaries = False
      elif Dirichlet_all is list:
          if Dirichlet_all[0][0]:
              boundaries.remove(1)
          if Dirichlet_all[0][1]:
              boundaries.remove(2)
          if Dirichlet_all[1][0]:
              boundaries.remove(3)
          if Dirichlet_all[1][0]:
              boundaries.remove(4)
      else:
          assert TypeError('Only [[False/Dirichlet on x=0, True/x=1],[False/y=0, True/y=1]] as example or True/False')
      # ...
      self.root        = root
      self.pyref_patch = pyref_patch
      self.knots_data  = knots_data
      self._degree     = degree_data
      self._coefs      = np.asarray([coefs_data[:,n].reshape(_nbasis) for n in range(geo_dim)])
      self._grids      = [knots_data[n][degree_data[n]:-degree_data[n]] for n in range(dim)]
      self._weights    = weights_data
      self._dim        = dim
      self._nurbs      = nurbs
      self._geo_dim    = geo_dim
      self._nbasis     = _nbasis
      self._nelements  = [_nbasis[n]-degree_data[n] for n in range(dim)]
      self.filename    = filename        
      self.element_id  = element_id
      self.boundaries  = boundaries
      self.dirichmlet  = Dirichlet_all

    @property
    def nbasis(self):
        return self._nbasis
    @property
    def dim(self):
        return self._dim
    @property
    def geo_dim(self):
        return self._geo_dim
    @property
    def knots(self):
        return [np.asarray(self.knots_data[i]) for i in range(self.dim)]
    @property
    def degree(self):
        return self._degree
    @property
    def nelements(self):
        return self._nelements
    @property
    def grids(self):
        return self._grids
    @property
    def nurbs(self):
        return self._nurbs
    @property
    def nb_patches(self):
        return 1
    @property
    def weights(self):
        Omega = self._weights.reshape(self._nbasis)
        if self.dim == 2:
            return Omega[:,0], Omega[0,:]
        else:
            return Omega[:,0,0], Omega[0,:,0], Omega[0,0,:]
    @property
    def coefs(self):    
        return [self._coefs[i].reshape(self._nbasis) for i in range( self.geo_dim)]
    #.. get tensor space
    @property
    def space(self):
        if self.dim == 2:
            # ... space of a B-spline patches
            Vmp1 = SplineSpace(degree=self.degree[0], grid = self.grids[0], omega = self.weights[0])
            Vmp2 = SplineSpace(degree=self.degree[1], grid = self.grids[1], omega = self.weights[1])
            return TensorSpace(Vmp1, Vmp2)
        else:
            # ... space of a B-spline patches
            Vmp1 = SplineSpace(degree=self.degree[0], grid = self.grids[0], omega = self.weights[0])
            Vmp2 = SplineSpace(degree=self.degree[1], grid = self.grids[1], omega = self.weights[1])
            Vmp3 = SplineSpace(degree=self.degree[2], grid = self.grids[2], omega = self.weights[2])
            return TensorSpace(Vmp1, Vmp2, Vmp3)        
    #.. get mapping in stencil vector format
    @property
    def stencil_mapping(self):
        # ... space of a B-spline patches
        Vh  = self.space
        if self.geo_dim == 2:
            #... create stencil vector for mapping
            u1  = StencilVector(Vh.vector_space)
            u2  = StencilVector(Vh.vector_space)
            #... get coefs
            xmp, ymp = self.coefs
            #... fill the stencil vector
            u1.from_array(Vh, xmp)
            u2.from_array(Vh, ymp)        
            return u1, u2
        elif self.geo_dim == 3:
            #... create stencil vector for mapping
            u1  = StencilVector(Vh.vector_space)
            u2  = StencilVector(Vh.vector_space)
            u3  = StencilVector(Vh.vector_space)
            #... get coefs
            xmp, ymp, zmp = self.coefs
            #... fill the stencil vector
            u1.from_array(Vh, xmp)
            u2.from_array(Vh, ymp)        
            u3.from_array(Vh, zmp)        
            return u1, u2, u3
        else:
            raise TypeError('dimension mismatch')
    def clone(self, Dirichlet_all = None):
        if Dirichlet_all is None:
            Dirichlet_all = self.Dirichlet_all
        return pyref_patch(self.filename, self.element_id, nurbs = self.nurbs, Dirichlet_all = Dirichlet_all)
    #.. get spline space on uniform mesh
    def getspace(self, V):
        '''
        Docstring pour getspace
        Compues basis in mesh using quad_degree quadrature points and unified integral discretization
        
        :param self: Description
        :param V: TensorSpace from finite element
        '''
        assert isinstance(V, TensorSpace), "Expecting TensorSpace"

        if self._weights is None:
            raise TypeError('Geometry is not NURBs please use same basis as for FE')

        if self.dim == 2:
            # ... space of a B-spline patches on uniform mesh
            Vmp1 = SplineSpace(degree=self.degree[0], grid = self.grids[0], omega = self.weights[0], nderiv= V.nderiv[0], mesh= V.mesh[0], quad_degree= V.weights[0].shape[1]-1)
            Vmp2 = SplineSpace(degree=self.degree[1], grid = self.grids[1], omega = self.weights[1], nderiv= V.nderiv[1], mesh= V.mesh[1], quad_degree= V.weights[1].shape[1]-1)
            return TensorSpace(V.spaces[0], V.spaces[1], Vmp1, Vmp2) if V.dim == 2 else TensorSpace(V.spaces[0], V.spaces[1], V.spaces[2], V.spaces[3], Vmp1, Vmp2)
        else:
            # ... space of a B-spline patches on uniform mesh
            Vmp1 = SplineSpace(degree=self.degree[0], grid = self.grids[0], omega = self.weights[0], nderiv= V.nderiv[0], mesh= V.mesh[0], quad_degree= V.weights[0].shape[1]-1)
            Vmp2 = SplineSpace(degree=self.degree[1], grid = self.grids[1], omega = self.weights[1], nderiv= V.nderiv[1], mesh= V.mesh[1], quad_degree= V.weights[1].shape[1]-1)
            Vmp3 = SplineSpace(degree=self.degree[2], grid = self.grids[2], omega = self.weights[2], nderiv= V.nderiv[2], mesh= V.mesh[2], quad_degree= V.weights[2].shape[1]-1)
            return TensorSpace(V.spaces[0], V.spaces[1], V.spaces[2], Vmp1, Vmp2, Vmp3)
    def assemble_dirichlet(self, V, g, boundaries = None):
        '''
        Docstring pour assemble_dirichlet
            computes StencilVector Dirichlet Solution zero inside the domaine
        :param self: Description
        :param V: Description
        :param g: Description
        :param boundaries: Description
        '''
        if boundaries is None:
           boundaries = self.boundaries
        if isinstance(V, TensorSpace) and V.dim == 2:
            #... mapping as arrays
            xmp, ymp         = self.coefs
            # Assemble Dirichlet boundary conditions
            u_d = build_dirichlet(V, g, map = (xmp, ymp, self.space), Boundaries  = boundaries)[1]
            return u_d
             
        else:
            raise TypeError('Expecting two dimensions TensorSpace')
    def rotated_2d(self, theta):
        '''
        Docstring pour rotated_2d
        
        :param self: Description
        :param theta: angle of rotation in radians
        :return: rotated coordinates (xmp, ymp)
        '''
        assert self.dim == 2, "Rotation is only implemented for 2D geometries."

        xmp, ymp = self._coefs[0].reshape(self._nbasis), self._coefs[1].reshape(self._nbasis)
        #... rotation loop
        for x ,y in zip(xmp.flatten(), ymp.flatten()):
            R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
            xy_rot = R @ np.array([x, y])
            xmp[np.where(xmp == x)] = xy_rot[0]
            ymp[np.where(ymp == y)] = xy_rot[1]
        return xmp, ymp    
    
    def Refinegrid(self, j_direct, numElevate = 1):
        """
        Refine a grid multiple times by inserting midpoints between unique knots.
        
        Parameters
        ----------
        numElevate : int
            Number of refinement iterations.
        Returns
        -------
        Grid_refined : ndarray
            Refined grid with original multiplicities preserved.
        """
        if numElevate ==0:
            return self._grids[j_direct]

        #... refine the grid numElevate times
        Grid_refined = np.array(self._grids[j_direct])        
        for _ in range(numElevate):
            # extract unique values
            unique_knots = np.unique(Grid_refined)
            # compute midpoints between consecutive unique knots
            midpoints = 0.5 * (unique_knots[:-1] + unique_knots[1:])
            # merge original grid and midpoints
            Grid_refined = np.sort(np.concatenate([Grid_refined, midpoints]))
        
        return Grid_refined
    
    def RefineGeometryMap(self, numElevate=1):
        """
        pyref_patch :  Refine the geometry by elevating the DoFs numElevate times.
        """
        assert(numElevate >= 1)
        #... refine the grid numElevate times
        #... refine the space
        if self.dim == 2:
            Vh1       = SplineSpace(degree=self.degree[0], grid= self.Refinegrid(0, numElevate))
            Vh2       = SplineSpace(degree=self.degree[1], grid= self.Refinegrid(1, numElevate))
            Vh        = TensorSpace(Vh1, Vh2)# after refinement
            #... initial
            nbasis_tot = self._nbasis[0]*self._nbasis[1]
            VH         = self.space
        else:
            Vh1        = SplineSpace(degree=self.degree[0], grid= self.Refinegrid(0, numElevate))
            Vh2        = SplineSpace(degree=self.degree[1], grid= self.Refinegrid(1, numElevate))
            Vh3        = SplineSpace(degree=self.degree[2], grid= self.Refinegrid(2, numElevate))            
            Vh         = TensorSpace(Vh1, Vh2, Vh3)# after refinement
            #... initial
            VH         = self.space
            nbasis_tot = self._nbasis[0]*self._nbasis[1]*self._nbasis[2]

        # Extract coefs data
        coefs_data = []
        # Refine the coefs
        M_mp      = prolongation_matrix(VH, Vh)
        if self.nurbs:
            coefs_data.append( (M_mp.dot(self._weights)).reshape(Vh.nbasis))
            for i in range(self.geo_dim):
                coefs_data.append( (M_mp.dot(self._weights*self._coefs[i].reshape(nbasis_tot))).reshape(Vh.nbasis) / coefs_data[0])
            return coefs_data
        else:
            for i in range(self.geo_dim):
                coefs_data.append( (M_mp.dot(self._coefs[i].reshape(nbasis_tot))).reshape(Vh.nbasis) )
            return coefs_data
    def Refinesolution(self, solution, VH, Vh):
        """
        weights_h : Weights for the NURBS geometry, if None, it is assumed to be ones.
        the user should provide the weights if the geometry is NURBS already in uniform mesh.
        Refine the solution by elevating the DoFs numElevate times.
        """
        if isinstance(solution, StencilVector):
            # Refine the coefs
            M_mp      = prolongation_matrix(VH, Vh)
            return (M_mp.dot(solution.toarray())).reshape(Vh.nbasis)
        else:
            raise AssertionError("Solution must be a StencilVector")
    # ...
    def eval_mesh(self, mesh, i_dir):
        '''
        Docstring pour eval : evaluate at a given point
        
        :param self: 
        #:param mesh: given mesh
        #:param i_dir: i direction (0/x,1/y,2/z)
        '''
        from   .  import nurbs_utilities_core as core
        uh  = self.coefs[i_dir]
        if self.dim == 2:
            w1, w2 = self.weights
            Tu, Tv = self.knots
            pu, pv = self.degree
            # ...
            nx, ny   = mesh[0].shape
            Q        = np.zeros((nx, ny, 5))
            Q[:,:,3] = mesh[0][:,:]
            Q[:,:,4] = mesh[1][:,:] 
            core.sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, w1, w2, Q)       
            return Q[:,:,0], Q[:,:,1], Q[:,:,2]
        elif self.dim == 3:
            nx, ny, nz = mesh[0].shape
            # ...
            w1, w2, w3 = self.weights
            Tu, Tv, Tz = self.knots
            pu, pv, pz = self.degree
            # ...
            Q          = np.zeros((nx, ny, nz, 7))
            Q[:,:,:,4] = mesh[0][:,:,:]
            Q[:,:,:,5] = mesh[1][:,:,:]
            Q[:,:,:,6] = mesh[2][:,:,:]
            core.sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tz, pu, pv, pz, w1, w2, w3, Q)
            return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3]
        
    def eval(self, patch_nb = 1, mesh = None, nbpts = None):
        '''
        Docstring pour eval computes mapping in a given mesh
        :param self: 
        :param mesh: given mesh
        :param nbpts: given nymber of meshgrid
        '''
        from . import fast_diag_core as fs
        if mesh is None:
            if self.dim == 2:
                xs, ys = self.grids
                if nbpts is None:
                    nbpts = [len(self.grids[i]) for i in range(self.dim)]
                else:
                    if isinstance(nbpts, int):
                        nbpts = (nbpts,nbpts)
                    nx, ny = nbpts
                    Tu, Tv = self.knots
                    pu, pv = self.degree
                    xs     = np.linspace(Tu[pu], Tu[-pu-1], nx)
                    ys     = np.linspace(Tv[pv], Tv[-pv-1], ny)
                # ...
                meshx = np.zeros(nbpts)
                meshy = np.zeros(nbpts)
                fs.build_identity_mapping_2d(xs, ys, meshx, meshy)                         
                return [self.eval_mesh((meshx, meshy), i)[0] for i in range(self.geo_dim)]
            elif self.dim == 3:
                # ...
                xs, ys, zs = self.grids 
                if nbpts is None:
                    nbpts = [len(self.grids[i]) for i in range(self.dim)]
                else:
                    if isinstance(nbpts, int):
                        nbpts = (nbpts,nbpts,nbpts)
                    nx, ny, nz = nbpts
                    Tu, Tv, Tz = self.knots
                    pu, pv, pz = self.degree
                    xs         = np.linspace(Tu[pu], Tu[-pu-1], nx)
                    ys         = np.linspace(Tv[pv], Tv[-pv-1], ny)
                    zs         = np.linspace(Tz[pz], Tz[-pz-1], nz)
                # ...
                meshx  = np.zeros(nbpts)
                meshy  = np.zeros(nbpts)
                meshz  = np.zeros(nbpts)
                fs.build_identity_mapping_3d(xs,ys,zs, meshx, meshy, meshz)
                return [self.eval_mesh((meshx, meshy, meshz),0)[0], 
                        self.eval_mesh((meshx, meshy, meshz),1)[0], 
                        self.eval_mesh((meshx, meshy, meshz),2)[0]]
            else:
                raise TypeError('Expect two or three dimensions')
        else:
            if self.dim == 2:
                return [self.eval_mesh(mesh,i)[0] for i in range(self.geo_dim)]
            elif self.dim == 3:
                return [self.eval_mesh(mesh,0)[0], 
                        self.eval_mesh(mesh,1)[0],
                        self.eval_mesh(mesh,2)[0]]
            else:
                raise TypeError('Expect two or three dimensions')            
    def gradient(self, patch_nb =1, mesh = None, nbpts = None):
        '''
        Docstring pour gradient computes gradient of the solution in a given mesh
        :param self: 
        :param mesh: given mesh
        :param nbpts: given nymber of meshgrid
        '''
        from . import fast_diag_core as fs
        if mesh is None:
            if self.dim == 2:
                xs, ys = self.grids
                if nbpts is None:
                    nbpts = [len(self.grids[i]) for i in range(self.dim)]
                else:
                    if isinstance(nbpts, int):
                        nbpts = (nbpts,nbpts)
                    nx, ny = nbpts
                    Tu, Tv = self.knots
                    pu, pv = self.degree
                    xs = np.linspace(Tu[pu], Tu[-pu-1], nx)
                    ys = np.linspace(Tv[pv], Tv[-pv-1], ny)
                # ...
                meshx = np.zeros(nbpts)
                meshy = np.zeros(nbpts)
                fs.build_identity_mapping_2d(xs, ys, meshx, meshy) 
                if self.geo_dim == 2:          
                    return [[self.eval_mesh((meshx, meshy), 0)[1], 
                            self.eval_mesh((meshx, meshy), 0)[2]], 
                            [self.eval_mesh((meshx, meshy), 1)[1], 
                            self.eval_mesh((meshx, meshy), 1)[2]]]
                else:
                    return [[self.eval_mesh((meshx, meshy), 0)[1], 
                            self.eval_mesh((meshx, meshy), 0)[2]], 
                            [self.eval_mesh((meshx, meshy), 1)[1], 
                            self.eval_mesh((meshx, meshy), 1)[2]],
                            [self.eval_mesh((meshx, meshy), 2)[1], 
                            self.eval_mesh((meshx, meshy), 2)[2]]]
            elif self.dim == 3:
                # ...
                xs, ys, zs = self.grids 
                if nbpts is None:
                    nbpts = [len(self.grids[i]) for i in range(self.dim)] 
                else:
                    if isinstance(nbpts, int):
                        nbpts = (nbpts,nbpts,nbpts)
                    nx, ny, nz = nbpts
                    Tu, Tv, Tz = self.knots
                    pu, pv, pz = self.degree
                    xs = np.linspace(Tu[pu], Tu[-pu-1], nx)
                    ys = np.linspace(Tv[pv], Tv[-pv-1], ny)
                    zs = np.linspace(Tz[pz], Tz[-pz-1], nz)
                # ...
                meshx  = np.zeros(nbpts)
                meshy  = np.zeros(nbpts)
                meshz  = np.zeros(nbpts)
                fs.build_identity_mapping_3d(xs,ys,zs, meshx, meshy, meshz)
                return [[self.eval_mesh((meshx, meshy, meshz),0)[1], 
                        self.eval_mesh((meshx, meshy, meshz),0)[2], 
                        self.eval_mesh((meshx, meshy, meshz),0)[3]],
                        [self.eval_mesh((meshx, meshy, meshz),1)[1], 
                        self.eval_mesh((meshx, meshy, meshz),1)[2], 
                        self.eval_mesh((meshx, meshy, meshz),1)[3]],
                        [self.eval_mesh((meshx, meshy, meshz),2)[1], 
                        self.eval_mesh((meshx, meshy, meshz),2)[2], 
                        self.eval_mesh((meshx, meshy, meshz),2)[3]]]
            else:
                raise TypeError('Expect two or three dimensions')
        else:
            if self.dim == 2 and self.geo_dim == 2:
                return [[self.eval_mesh(mesh,0)[1],self.eval_mesh(mesh,0)[2]],
                       [self.eval_mesh(mesh,1)[1],self.eval_mesh(mesh,1)[2]]]
            if self.dim == 2 and self.geo_dim == 3:
                return [[self.eval_mesh(mesh,0)[1],self.eval_mesh(mesh,0)[2]],
                       [self.eval_mesh(mesh,1)[1],self.eval_mesh(mesh,1)[2]],
                       [self.eval_mesh(mesh,2)[1],self.eval_mesh(mesh,2)[2]]]
            elif self.dim == 3:
                return [[self.eval_mesh(mesh,0)[1], 
                        self.eval_mesh(mesh,0)[2],
                        self.eval_mesh(mesh,0)[3]],
                        [self.eval_mesh(mesh,1)[1], 
                        self.eval_mesh(mesh,1)[2],
                        self.eval_mesh(mesh,1)[3]],
                        [self.eval_mesh(mesh,2)[1], 
                        self.eval_mesh(mesh,2)[2],
                        self.eval_mesh(mesh,2)[3]]]
            else:
                raise TypeError('Expect two or three dimensions')

#========================================================================
#... construct connectivity between patches: 
# TODO doesn't support different oriontation
#========================================================================
class pyref_multipatch(object):
    """
    Detect connectivity between patches.
    Returns the list of interfaces and Dirichlet BCs to be applied on each patch.
    The convention for the patches is as follows:
#            2D CASE                        3D CASE
#    Edge 1, {(u,v) : u = 0}        Face 1, {(u,v,w) : u = 0}
#    Edge 2, {(u,v) : u = 1}        Face 2, {(u,v,w) : u = 1}
#    Edge 3, {(u,v) : v = 0}        Face 3, {(u,v,w) : v = 0}
#    Edge 4, {(u,v) : v = 1}        Face 4, {(u,v,w) : v = 1}
#                                   Face 5, {(u,v,w) : w = 0}
#                                   Face 6, {(u,v,w) : w = 1}
                                                 ___4___
                                                |       |
                                              1 |       | 2
                                                |_______|
                                                    3
                                                z     ______
                                                |    / 6   /|
                                                |   /     / |
                                                |  /_____/  |  
                                                | |   1 |   |
                                                | |     | 4 |
                                               3| |_____|___|
                                                |/      |  /
                                                /   5   | / 
                                               /        |/
                                              x---------y
                                                  2

    The interface is defined as the common edge between two patches.
    The Dirichlet BCs are defined as follows:
        [True, False] : Dirichlet BC on the left edge
        [False, True] : Dirichlet BC on the right edge
    The input are the control points of all patches.
    """
    def __init__(self, geometryname, id_list, nurbs = True, Dirichlet_all = True):
        #.. TODO FIND ID automatically from the XML file
        mp        = []
        for i in id_list:
            mp.append( pyref_patch(geometryname, i, nurbs = nurbs) )
        self._dim     = mp[0].dim
        self._geo_dim = mp[0].geo_dim
        #... edge id mapping
        idmapping = {
            1: (0, 0),
            2: (0, 1),
            3: (1, 0),
            4: (1, 1)
        }
        #... number of patches
        num_patches         = len(mp)
        #... list of interfaces (patch1, patch2, [edge_patch1, edge_patch2])
        interfaces          = []
        #... list of patches connection (interface, pach next)
        # patch_connection = {} TODO L shape
        #... First we assume all boundaries are Dirichlet
        dirichlet           = np.zeros((num_patches, self._dim, 2), dtype = bool)+Dirichlet_all
        # ...
        patch_has_interface = [False] * num_patches
        #... loop over all patches
        for i in range(num_patches):
            for j in range(i+1, num_patches):
                #... test if they share an interface
                interface_obj = pyrefInterface(mp[i], mp[j])
                if interface_obj.interface is False :
                    continue
                # ...
                interfaces.append( (i+1, j+1, interface_obj.interface) )
                #... set the Dirichlet BCs False for the interface edges
                dirichlet[i,idmapping[interface_obj.interface[0]][0],idmapping[interface_obj.interface[0]][1]] = False
                dirichlet[j,idmapping[interface_obj.interface[1]][0],idmapping[interface_obj.interface[1]][1]] = False
                #...
                patch_has_interface[i] = True
                patch_has_interface[j] = True
        # ...
        assert num_patches <= 1 or all(patch_has_interface), \
            "Invalid geometry: at least one patch has no interface"
        # ...
        self.num_patches      = num_patches
        self.interfaces       = interfaces
        self.dirichlet        = dirichlet.tolist()
        self.geometryname     = geometryname
        self.id_list          = id_list
        self.patches          = mp
        self.idmapping        = idmapping
        self.nurbs            = nurbs
        # ...
        self._degree          = mp[0].degree
        self._knots           = mp[0].knots
        self._grids           = mp[0].grids
        self._weights         = mp[0].weights
        self.Ninterfaces      = len(self.interfaces) 
        self._space           = mp[0].space
        # self.patch_connection = patch_connection
    @property
    def NURBS(self):
        return self.nurbs
    @property
    def dim(self):
        return self._dim
    @property
    def geo_dim(self):
        return self._geo_dim
    #.. get degree
    @property
    def degree(self, num_patch=1):
        return self._degree 
    #.. get knots
    @property
    def knots(self):
        return [np.asarray(self._knots[i]) for i in range(self.dim)]  
    #.. get grids
    @property
    def grids(self):
        return self._grids
    #.. get weights: same for all patches
    @property
    def weights(self):
        return [self._weights[i] for i in range(self.dim)]  
    #... get number of Interfaces
    @property
    def nb_interfaces(self):
        return self.Ninterfaces
    #.. get id list
    @property
    def getidlist(self):
        return self.id_list
    #.. get number of patches
    @property
    def nb_patches(self):
        return self.num_patches
    #.. get geometry name    
    @property
    def getGeometryname(self):
        return self.geometryname
    #.. get tensor space
    @property
    def space(self):
        return self._space
    @property
    def setFalseDirichlet(self):
        self.dirichlet = np.zeros((self.num_patches, self._dim, 2), dtype = bool)+False
    # ...
    def eval(self, patch_nb, mesh=None, nbpts=None):
        '''
        Docstring pour eval
        
        :param self: Description
        :param patch_nb: patch number
        :param mesh: collection of meshgrid poins 
        :param nbpts: number of meshgrid points in each direction
        '''
        patch = self.patches[patch_nb-1]
        assert isinstance(patch, pyref_patch)
        return patch.eval(patch_nb, mesh=mesh, nbpts=nbpts)
    # ...
    def gradient(self, patch_nb, mesh=None, nbpts=None):
        '''
        Docstring pour gradient 
        
        :param self: Description
        :param patch_nb: patch number
        :param mesh: collection of meshgrid poins 
        :param nbpts: number of meshgrid points in each direction
        '''
        patch = self.patches[patch_nb-1]
        assert isinstance(patch, pyref_patch)
        return patch.gradient(patch_nb, mesh=mesh, nbpts=nbpts)    
    def clone(self, Dirichlet_all = True):
        return pyref_multipatch(self.geometryname, self.id_list, nurbs = self.nurbs, Dirichlet_all = Dirichlet_all)
    #.. get coefs
    def getcoefs(self, num_patch=1):
        return self.patches[num_patch-1].coefs
    #.. get coefs
    def getAllcoefs(self, direct='x'):
        map_xy = {
            'x':0,
            'y':1, 
            'z':2
        }
        return [self.patches[np].coefs[map_xy[direct]] for np in range(self.num_patches)]
            
    #.. get spline space on uniform mesh
    def getspace(self, V):
        '''
        Docstring pour getspace
        Computes basis in mesh using quad_degree quadrature points and smae integral discretization
        
        :param V: TensorSpace
        '''
        if isinstance(V, TensorSpace):
            return self.patches[0].getspace(V)
        else :
            raise TypeError('Expect only TensorSpace')

    #.. get mapping in stencil vector format
    def stencil_mapping(self, num_patch=1):
        # ... space of a B-spline patches
        if num_patch > 0:
            return self.patches[num_patch-1].stencil_mapping
        else:
            raise TypeError('three dimension is not yet updatedn or maybe num_patch = 1 not 0')
    #.. get refined geometry map
    def getRefineGeometryMap(self, num_patch=1, numElevate=1):
        return self.patches[num_patch-1].RefineGeometryMap(numElevate=numElevate)
    
    #.. refined grid
    def Refinegrid(self, j_direct=0, numElevate=1):
        return self.patches[0].Refinegrid(j_direct, numElevate=numElevate)

    #.. get interfaces
    def getInterfaces(self):
        return self.interfaces
    
    #.. get dirichlet BCs for a given patch
    def getDirPatch(self, num_patch):
        return self.dirichlet[num_patch-1]
    # .. get singularity points L shape TODO
    # def getSingularityPatch(self, num_patch):
    #     interfaceP = self.getInterfacePatch(num_patch)
    #     if 2 in interfaceP and 4 in interfaceP:            
    #         patch_tmp = self.patch_connection(num_patch)
    #     return 0

    #.. get interfaces for a given patch
    def getInterfacePatch(self, num_patch):
        interface_patch = []
        for interface in self.interfaces:
            if interface[0] == num_patch:
                interface_patch.append(interface[2][0])
            if interface[1] == num_patch:
                interface_patch.append(interface[2][1])
        return np.asarray(interface_patch)

    #.. get dirichlet boundaries for a given patch
    def getDirichletBoundaries(self, num_patch):
        boundary_dirichlet = []
        for edge in range(1,5):
            if self.dirichlet[num_patch-1][self.idmapping[edge][0]][self.idmapping[edge][1]] == True:
                boundary_dirichlet.append(edge)
        # print(f"Patch {num_patch} Dirichlet boundaries : {boundary_dirichlet}")
        return np.asarray(boundary_dirichlet)
    
    def assemble_dirichlet(self, V, g):
        '''
        Docstring pour assemble_dirichlet
            computes StencilVector Dirichlet Solution zero inside the domaine
        :param self: Description
        :param V: Description
        :param g: Description
        :param boundaries: Description
        '''
        if isinstance(V, TensorSpace) and V.dim == 2:
            u_d           = []
            for patch_nb in range(1, self.nb_patches+1):
                #... mapping as arrays
                xmp, ymp         = self.getcoefs(patch_nb)
                # Assemble Dirichlet boundary conditions
                u_d1 = build_dirichlet(V, g, map = (xmp, ymp, self.space), Boundaries  = self.getDirichletBoundaries(patch_nb))[1]
                u_d.append(u_d1)
            return u_d
        else:
            raise TypeError('Expecting two dimensions TensorSpace')
    #.. print multipatch info
    def detail(self):
        print(f"Number of patches : {self.num_patches}")
        print("Interfaces between patches :")
        for interface in self.interfaces:
            print(f" Patches {interface[0]} and {interface[1]} share edges {interface[2]}")
        print("Dirichlet BCs for each patch :")
        for i in range(self.num_patches):
            print(f" Patch {i+1} : {self.dirichlet[i]}")
    def __repr__(self):
        return f"pyref_multipatch(num_patches={self.num_patches}, interfaces={self.interfaces}, dirichlet={self.dirichlet})"
    
    def __str__(self):
        return f"pyref_multipatch with {self.num_patches} patches"
    def __len__(self):
        return self.num_patches
    def __getitem__(self, index):
        if index < 0 or index >= self.num_patches:
            raise IndexError("Index out of range")
        return {
            'interfaces': self.getInterfacePatch(index+1),
            'dirichlet': self.getDirPatch(index)
        }
    def __iter__(self):
        for i in range(self.num_patches):
            yield {
                'interfaces': self.getInterfacePatch(i+1),
                'dirichlet': self.getDirPatch(i)
            }
    def __contains__(self, item):
        for i in range(self.num_patches):
            if item == {
                'interfaces': self.getInterfacePatch(i+1),
                'dirichlet': self.getDirPatch(i)
            }:
                return True
        return False
#========================================================================
#... construct connectivity between two patches: 
# TODO doesn't support different oriontation
#========================================================================
class pyrefInterface(object):
    """
    Detect interface between two patches.
    Returns the list of interfaces and Dirichlet BCs to be applied on each patch.
    The convention for the patches is as follows:
#            2D CASE                        3D CASE
#    Edge 1, {(u,v) : u = 0}        Face 1, {(u,v,w) : u = 0}
#    Edge 2, {(u,v) : u = 1}        Face 2, {(u,v,w) : u = 1}
#    Edge 3, {(u,v) : v = 0}        Face 3, {(u,v,w) : v = 0}
#    Edge 4, {(u,v) : v = 1}        Face 4, {(u,v,w) : v = 1}
#                                   Face 5, {(u,v,w) : w = 0}
#                                   Face 6, {(u,v,w) : w = 1}
                                                 ___4___
                                                |       |
                                              1 |       | 2
                                                |_______|
                                                    3
                                                z     ______
                                                |    / 6   /|
                                                |   /     / |
                                                |  /_____/  |  
                                                | |   1 |   |
                                                | |     | 4 |
                                               3| |_____|___|
                                                |/      |  /
                                                /   5   | / 
                                               /        |/
                                              x---------y
                                                  2

    The interface is defined as the common edge between two patches.
    The Dirichlet BCs are defined as follows:
        [True, False] : Dirichlet BC on the left edge
        [False, True] : Dirichlet BC on the right edge
    The input are the control points of the two patches.
    """
    def __init__(self, mp, mp_next):
        assert isinstance(mp, pyref_patch)
        assert isinstance(mp_next, pyref_patch)
        if mp.geo_dim == mp_next.geo_dim == 2 and mp.dim == mp_next.dim == 2:
            xmp, ymp     = mp.coefs
            xmp1, ymp1   = mp_next.coefs
            #...
            self.interface   = False
            self.dirichlet_1 = False
            self.dirichlet_2 = False
            if np.max(np.absolute(xmp[-1,:] - xmp1[0,:])+np.absolute(ymp[-1,:] - ymp1[0,:])) <= 1e-15 :
                self.interface   = [2,1]
                self.dirichlet_1 = [[True, False],[True, True]]
                self.dirichlet_2 = [[False, True],[True, True]]
            elif np.max(np.absolute(xmp[0,:] - xmp1[-1,:])+np.absolute(ymp[0,:] - ymp1[-1,:])) <= 1e-15 :
                self.interface   = [1,2]
                self.dirichlet_1 = [[False, True], [True, True]]
                self.dirichlet_2 = [[True, False], [True, True]]
            elif np.max(np.absolute(xmp[:,0] - xmp1[:,-1])+np.absolute(ymp[:,0] - ymp1[:,-1])) <= 1e-15 :
                self.interface   = [3,4]
                self.dirichlet_1 = [[True, True], [False, True]]
                self.dirichlet_2 = [[True, True], [True, False]]
            elif np.max(np.absolute(xmp[:,-1] - xmp1[:,0])+np.absolute(ymp[:,-1] - ymp1[:,0])) <= 1e-15 :
                self.interface   = [4,3]
                self.dirichlet_1 = [[True, True], [True, False]]
                self.dirichlet_2 = [[True, True], [False, True]]
        elif mp.geo_dim == mp_next.geo_dim == 3 and mp.dim == mp_next.dim == 2:
            xmp, ymp, zmp    = mp.coefs
            xmp1, ymp1, zmp1 = mp_next.coefs
            #...
            self.interface   = False
            self.dirichlet_1 = False
            self.dirichlet_2 = False
            if np.max(np.absolute(xmp[-1,:] - xmp1[0,:])+np.absolute(ymp[-1,:] - ymp1[0,:])+np.absolute(zmp[-1,:] - zmp1[0,:])) <= 1e-15 :
                self.interface   = [2,1]
                self.dirichlet_1 = [[True, False],[True, True]]
                self.dirichlet_2 = [[False, True],[True, True]]
            elif np.max(np.absolute(xmp[0,:] - xmp1[-1,:])+np.absolute(ymp[0,:] - ymp1[-1,:])+np.absolute(zmp[0,:] - zmp1[-1,:])) <= 1e-15 :
                self.interface   = [1,2]
                self.dirichlet_1 = [[False, True], [True, True]]
                self.dirichlet_2 = [[True, False], [True, True]]
            elif np.max(np.absolute(xmp[:,0] - xmp1[:,-1])+np.absolute(ymp[:,0] - ymp1[:,-1])+np.absolute(zmp[:,0] - zmp1[:,-1])) <= 1e-15 :
                self.interface   = [3,4]
                self.dirichlet_1 = [[True, True], [False, True]]
                self.dirichlet_2 = [[True, True], [True, False]]
            elif np.max(np.absolute(xmp[:,-1] - xmp1[:,0])+np.absolute(ymp[:,-1] - ymp1[:,0])+np.absolute(zmp[:,-1] - zmp1[:,0])) <= 1e-15 :
                self.interface   = [4,3]
                self.dirichlet_1 = [[True, True], [True, False]]
                self.dirichlet_2 = [[True, True], [False, True]]
        elif mp.geo_dim == mp_next.geo_dim == 3 and mp.dim == mp_next.dim == 3:
            xmp, ymp, zmp    = mp.coefs
            xmp1, ymp1, zmp1 = mp_next.coefs
            #...
            self.interface   = False
            self.dirichlet_1 = False
            self.dirichlet_2 = False
            if np.max(np.absolute(xmp[-1,:,:] - xmp1[0,:,:])+np.absolute(ymp[-1,:,:] - ymp1[0,:,:])+np.absolute(zmp[-1,:,:] - zmp1[0,:,:])) <= 1e-15 :
                self.interface   = [2,1]
                self.dirichlet_1 = [[True, False],[True, True],[True, True]]
                self.dirichlet_2 = [[False, True],[True, True],[True, True]]
            elif np.max(np.absolute(xmp[0,:,:] - xmp1[-1,:,:])+np.absolute(ymp[0,:,:] - ymp1[-1,:,:])+np.absolute(zmp[0,:,:] - zmp1[-1,:,:])) <= 1e-15 :
                self.interface   = [1,2]
                self.dirichlet_1 = [[False, True], [True, True],[True, True]]
                self.dirichlet_2 = [[True, False], [True, True],[True, True]]
            elif np.max(np.absolute(xmp[:,0,:] - xmp1[:,-1,:])+np.absolute(ymp[:,0,:] - ymp1[:,-1,:])+np.absolute(zmp[:,0,:] - zmp1[:,-1,:])) <= 1e-15 :
                self.interface   = [3,4]
                self.dirichlet_1 = [[True, True], [False, True],[True, True]]
                self.dirichlet_2 = [[True, True], [True, False],[True, True]]
            elif np.max(np.absolute(xmp[:,-1,:] - xmp1[:,0,:])+np.absolute(ymp[:,-1,:] - ymp1[:,0,:])+np.absolute(zmp[:,-1,:] - zmp1[:,0,:])) <= 1e-15 :
                self.interface   = [4,3]
                self.dirichlet_1 = [[True, True], [True, False],[True, True]]
                self.dirichlet_2 = [[True, True], [False, True],[True, True]]
            elif np.max(np.absolute(xmp[:,:,0] - xmp1[:,:,-1])+np.absolute(ymp[:,:,0] - ymp1[:,:,-1])+np.absolute(zmp[:,:,0] - zmp1[:,:,-1])) <= 1e-15 :
                self.interface   = [5,6]
                self.dirichlet_1 = [[True, True],[True, True], [False, True]]
                self.dirichlet_2 = [[True, True],[True, True], [True, False]]
            elif np.max(np.absolute(xmp[:,:,-1] - xmp1[:,:,0])+np.absolute(ymp[:,:,-1] - ymp1[:,:,0])+np.absolute(zmp[:,:,-1] - zmp1[:,:,0])) <= 1e-15 :
                self.interface   = [6,5]
                self.dirichlet_1 = [[True, True],[True, True], [True, False]]
                self.dirichlet_2 = [[True, True],[True, True], [False, True]]
        else:
            raise TypeError('only two or three dimensions')

    def interface(self):
        return self.interface
    def dirichlet_1(self):
        return self.dirichlet_1
    def dirichlet_2(self):
        return self.dirichlet_2
    
    def printInterface(self):
        print(f"Interfaces ({self.interface})")
        print(f"Dirichlet BCs for patch {1} : {self.dirichlet_1}")
        print(f"Dirichlet BCs for patch {2} : {self.dirichlet_2}")

    def setInterface(self, xd1, xd2):
        if self.interface[0] == 2 and self.interface[1] == 1:
            xd1[-1,1:-1]   = 0.0 # Reset xd to zero
            xd2[0,1:-1]    = 0.0 # Reset xd to zero
        elif self.interface[0] == 1 and self.interface[1] == 2:
            xd1[0,1:-1]    = 0.0 # Reset xd to zero
            xd2[-1,1:-1]   = 0.0 # Reset xd to zero
        elif self.interface[0] == 3 and self.interface[1] == 4 :
            xd1[1:-1,0]    = 0.0 # Reset xd to zero
            xd2[1:-1,-1]   = 0.0 # Reset xd to zero
        elif self.interface[0] == 4 and self.interface[1] == 3 :
            xd1[1:-1,-1]   = 0.0
            xd2[1:-1,0]    = 0.0 # Reset xd to zero
        else:
            raise ValueError("Invalid interface configuration")
        return xd1, xd2

#========================================================================
# ... loadf xml file from pyrefiga
#========================================================================    
import pyrefiga
from pathlib import Path

def load_xml(name: str) -> str:
    """
    Return the full path to a data file inside 'fields/' as a string.

    Parameters
    ----------
    name : str
        Filename of the XML file (e.g., "annulus.xml").

    Returns
    -------
    str
        Full path to the requested file as a string.
    """
    # Installed location
    base_path = Path(pyrefiga.__file__).parent.parent
    xml_path = base_path / "fields" / name
    if xml_path.exists():
        return str(xml_path)

    # Development fallback
    dev_path = Path(__file__).parent.parent.parent / "fields" / name
    if dev_path.exists():
        return str(dev_path)

    raise FileNotFoundError(f"Cannot find XML file '{name}' in fields/")