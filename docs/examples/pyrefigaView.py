"""
pyrefigaView.py

# Example on how one can show a geometry discrebed by nurbs or b-spline

@author : M. BAHARI
"""

from pyrefiga import ViewGeo
from pyrefiga import load_xml
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
import argparse
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
parser.add_argument("--name", type=str, default='cylinder.xml', help="Name of geometry (default: 'cylinder.xml')")
parser.add_argument("--f", type=str, default='1./(2.+np.cos(2.*np.pi*np.sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2+(z-0.5)**2)))', 
                    help="Function expression (default: '1./(2.+np.cos(2.*np.pi*np.sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2+(z-0.5)**2)))')")
parser.add_argument("--mp", type=int, nargs="+",help="Number of patches (default: [0])", default=[0])
parser.add_argument("--nbpts", type=int, default=100, help="Number of elements used for plot(default: 100)")
args = parser.parse_args()

#==============================================
#==============================================
nbpts    = args.nbpts
Nump     = args.mp
func     = args.f
# geometry = 'quart_annulus.xml'
# geometry = 'circle.xml'
#geometry  = 'egg.xml'
geometry  = load_xml(args.name)

functions = [
    {"name": "density", "expression": func},
]

ViewGeo(geometry, Nump, nbpts = nbpts, functions= functions, filename="figs/multipatch_geometry", plot = args.plot)