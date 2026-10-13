"""Export one 2D spline/NURBS basis function on its physical geometry.

PYTHONPATH=. python docs/examples/iga_basis_in_generalgeo.py \
    --name butterfly.xml --mp 0 --basis 2 3 --plot

Basis indices are zero-based: --basis 2 3 selects coefficient [2, 3].
The geometry's native degree, knot vectors and NURBS weights are used.
Without --plot, the ParaView files are saved without opening ParaView.
Author: M. BAHARI
"""

import argparse
from pathlib import Path

import numpy as np

from pyrefiga import load_xml, pyref_patch, paraview_nurbsSolutionMultipatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='butterfly.xml', help='Geometry XML filename or path')
    parser.add_argument('--mp', type=int, default=0, help='Geometry patch ID (default: 0)')
    parser.add_argument('--basis', '--index', type=int, nargs=2, default=[2, 3],
                        metavar=('I', 'J'), help='Zero-based basis index [I, J] (default: 2 3)')
    parser.add_argument('--nbpts', type=int, default=100, help='Sampling points per parametric direction')
    parser.add_argument('--output', default='figs/iga_basis', help='Output filename prefix')
    parser.add_argument('--plot', action='store_true', help='Open the exported basis in ParaView')
    args = parser.parse_args(argv)
    if args.nbpts < 2:
        parser.error('nbpts must be at least 2')

    try:
        filename = str(Path(args.name)) if Path(args.name).is_file() else load_xml(args.name)
        geometry = pyref_patch(filename, args.mp)
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if geometry.dim != 2:
        parser.error('Choose a 2D parametric geometry (planar patch or surface in 3D)')

    i, j = args.basis
    if not (0 <= i < geometry.nbasis[0] and 0 <= j < geometry.nbasis[1]):
        parser.error(f'Basis index [{i}, {j}] is outside shape {geometry.nbasis}; '
                     f'use 0 <= I < {geometry.nbasis[0]} and 0 <= J < {geometry.nbasis[1]}')
    space = geometry.space
    coefficients = np.zeros(space.nbasis)
    coefficients[i, j] = 1.0
    solutions = [dict(name=f'basis_{i}_{j}', data=[coefficients], space=space)]
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f'Patch {args.mp}: degree {space.degree}, basis shape {space.nbasis}')
    print(f'Selected basis [{i}, {j}]; in ParaView, color by basis_{i}_{j}.')
    paraview_nurbsSolutionMultipatch(args.nbpts, geometry, solution=solutions,
                                    filename=str(prefix), plot=args.plot)
    return geometry, space, coefficients


if __name__ == '__main__':
    main()
