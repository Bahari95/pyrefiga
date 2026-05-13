"""
interpolation_mapping_composition.py

interpolation for a given composition in two dimensions

Author: M. Bahari
"""
from pyrefiga import collocation_2dNURBspline
from pyrefiga import least_square_2dNURBspline
from pyrefiga import cubic_bspline_interpolation_2D
from pyrefiga import pyref_patch
from pyrefiga import TensorSpace
from pyrefiga import pyccel_sol_field_2d
from pyrefiga import sol_field_NURBS_2d
from pyrefiga import StencilVector
from pyrefiga import StencilVectorSpace
#...
import numpy as np

def collocation_solve(V, mp, u11, u12, nbpts):
        
        assert isinstance( mp, pyref_patch)
        assert isinstance( V,  TensorSpace)
        v11   = StencilVector(V.vector_space)
        v12   = StencilVector(V.vector_space)
        # Computes each component of the image by collocation
        f = lambda x,y : x
        vx11 = collocation_2dNURBspline(V, f, xmp = mp, adxmp = ( u11.tensor,  u12.tensor) )
        f = lambda x,y : y
        vx12 = collocation_2dNURBspline(V, f, xmp = mp, adxmp = ( u11.tensor, u12.tensor))
        #... store the solution in the form of control points
        v11.from_array(V, vx11)
        v12.from_array(V, vx12)
        # ...
        return v11, v12 
def LS_solve(V, mp, u11, u12, nbpts):
        
        assert isinstance( mp, pyref_patch)
        assert isinstance( V,  TensorSpace)
        v11   = StencilVector(V.vector_space)
        v12   = StencilVector(V.vector_space)
        # Computes data for the least square problem
        sx    = pyccel_sol_field_2d((nbpts, nbpts), u11.tensor, V.knots, V.degree)[0]
        sy    = pyccel_sol_field_2d((nbpts, nbpts), u12.tensor, V.knots, V.degree)[0]
        # #---Compute a image by initial mapping
        x,y = mp.eval(mesh = (sx, sy))
        vx11 = least_square_2dNURBspline(V, x)
        vx12 = least_square_2dNURBspline(V, y)
        #... store the solution in the form of control points
        v11.from_array(V, vx11)
        v12.from_array(V, vx12)
        # ...
        return v11, v12

def cubic_bspline_solve(V, mp, u11, u12, nbpts):
        
        assert isinstance( mp, pyref_patch)
        assert isinstance( V,  TensorSpace)
        # uniform grid
        xgrid  = V.grid[0]
        ygrid  = V.grid[1]

        X,Y = np.meshgrid(xgrid,ygrid)
        # function and derivative at boundaries
        f       = lambda x,y : np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
        dxf     = lambda x,y : 2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
        dyf     = lambda x,y : 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        dxyf    = lambda x,y : 2*np.pi*2*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


        v11   = StencilVector(V.vector_space)
        v12   = StencilVector(V.vector_space)
        # Computes data for the least square problem
        sx    = pyccel_sol_field_2d((nbpts, nbpts), u11.tensor, V.knots, V.degree)[0]
        sy    = pyccel_sol_field_2d((nbpts, nbpts), u12.tensor, V.knots, V.degree)[0]
        # #---Compute a image by initial mapping
        x,y = mp.eval(mesh = (sx, sy))

        # .. first component
        g       = x
        gprimex = [dxf(xgrid[0],xgrid),dxf(xgrid[-1],xgrid)]
        gprimey = [dxf(xgrid, xgrid[-1]), dxf(xgrid, xgrid[-1])]

        corners = [dxyf(xgrid[0], xgrid[0]), 
                dxyf(xgrid[0], xgrid[-1]), 
                dxyf(xgrid[-1], xgrid[0]), 
                dxyf(xgrid[-1], xgrid[-1])]
        vx11 = cubic_bspline_interpolation_2D(xgrid, xgrid, g, gprimex, gprimey, corners)
        vx12 = cubic_bspline_interpolation_2D(xgrid, xgrid, g, gprimex, gprimey, corners)
        #... store the solution in the form of control points
        v11.from_array(V, vx11)
        v12.from_array(V, vx12)
        # ...
        return v11, v12


