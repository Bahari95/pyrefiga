"""Plate-with-hole validation from Blanchard et al., RR-8111, section 4.

Reference: https://inria.hal.science/hal-00743199/document
Weak form: integral 2*mu*eps(u):eps(v) + lambda*div(u)*div(v) = integral traction.v.
The first-quadrant plate has hole radius 1, outer size 4, E=1e5, nu=0.3.
Plane stress and remote x-directed tension 10 are the example's conventions.

PYTHONPATH=. python docs/examples/elasticity2d_example.py --export
PYTHONPATH=. python docs/examples/elasticity2d_example.py --degree-study --levels 4
Author: M. Bahari
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from pyrefiga import load_xml
from pyrefiga.elasticity import NurbsElasticity2D, kirsch_stress, principal_stress


def export_solution(operator, coefficients, exact, nbpts, prefix, plot=False):
    """Export samples using the full rational geometry and physical gradients."""
    import pyvista as pv

    points = [np.linspace(knots[p], knots[-p-1], nbpts)
              for knots, p in zip(operator.knots, operator.degree)]
    sampled = operator.sample(coefficients, *points)
    x, y = sampled['coordinates']
    ux, uy = sampled['displacement']
    stress = sampled['stress']
    reference = exact(x, y)
    grid = pv.StructuredGrid(x, y, np.zeros_like(x))
    fields = dict(displacement_x=ux, displacement_y=uy, sigma_xx=stress[0],
                  sigma_yy=stress[1], sigma_xy=stress[2], exact_sigma_xx=reference[0],
                  exact_sigma_yy=reference[1], exact_sigma_xy=reference[2],
                  principal_stress=principal_stress(stress),
                  exact_principal_stress=principal_stress(reference),
                  error_sigma_xx=np.abs(stress[0]-reference[0]),
                  Jacobian=sampled['determinant'])
    for name, values in fields.items():
        grid[name] = values.ravel(order='F')
    grid['displacement'] = np.column_stack((ux.ravel(order='F'), uy.ravel(order='F'),
                                           np.zeros(ux.size)))
    filename = str(prefix)+'.vtm'
    pv.MultiBlock({'plate': grid}).save(filename)
    print(f'ParaView solution saved to {filename}')
    if plot:
        import subprocess
        subprocess.run(['paraview', str(Path(filename).resolve())], check=True)
    return filename


def plot_convergence(rows, prefix):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(10, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for degree in sorted({row['degree'] for row in rows}):
        group = [row for row in rows if row['degree'] == degree]
        for ax, name in zip(axes, ('L2_sigma_xx', 'L2_principal')):
            ax.loglog([r['free_dofs'] for r in group], [r[name] for r in group],
                      marker='o', label=f'p={degree}')
    for ax, title in zip(axes, (r'Cartesian stress $\sigma_{xx}$', r'Largest principal stress $\sigma_1$')):
        ax.set(xlabel='Free displacement DOFs', ylabel=r'$L^2(\Omega)$ stress error', title=title)
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
    figure.suptitle('Plate with a circular hole: stress convergence')
    for extension in ('png', 'pdf'):
        figure.savefig(str(prefix)+'_convergence.'+extension, dpi=180)
    print(f'Convergence plots saved to {prefix}_convergence.png and .pdf')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', default='elasticity.xml', help='XML filename in fields/ or a local path')
    parser.add_argument('--patch-id', type=int, default=0)
    parser.add_argument('--degree', type=int, default=2, help='Spline degree in both directions')
    parser.add_argument('--degree-study', action='store_true', help='Compare degrees 2, 3 and 4, and save convergence plots')
    parser.add_argument('--refinement', type=int, default=2, help='Initial dyadic knot-refinement level')
    parser.add_argument('--levels', type=int, default=3, help='Number of successive refinement levels')
    parser.add_argument('--young', type=float, default=1e5)
    parser.add_argument('--poisson', type=float, default=.3)
    parser.add_argument('--tension', type=float, default=10.)
    parser.add_argument('--plane', choices=('stress', 'strain'), default='stress')
    parser.add_argument('--convergence-plot', action='store_true')
    parser.add_argument('--export', action='store_true', help='Export the finest solution for each degree to ParaView')
    parser.add_argument('--plot', action='store_true', help='Export and open the last solution in ParaView')
    parser.add_argument('--nbpts', type=int, default=80)
    parser.add_argument('--output', default='figs/elasticity')
    args = parser.parse_args(argv)
    if args.degree < 2 or args.refinement < 0 or args.levels < 1 or args.nbpts < 2:
        parser.error('Use degree >= 2, refinement >= 0, levels >= 1 and nbpts >= 2')
    if not np.isfinite(args.tension):
        parser.error('tension must be finite')
    try:
        filename = str(Path(args.geometry)) if Path(args.geometry).is_file() else load_xml(args.geometry)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    degrees = (2, 3, 4) if args.degree_study else (args.degree,)

    def exact(x, y):
        return kirsch_stress(x, y, tension=args.tension)

    def traction(x, y, nx, ny):
        xx, yy, xy = exact(x, y)
        return xx*nx+xy*ny, xy*nx+yy*ny

    print(f'Plate with a unit hole; E={args.young:g}, nu={args.poisson:g}, '
          f'tension={args.tension:g}; plane {args.plane}')
    print(' p   elements    free DOFs    L2(sigma_xx)  order    L2(sigma_1)   order    ||u_h||_E^2')
    rows = []
    for degree in degrees:
        previous = None
        for level in range(args.refinement, args.refinement+args.levels):
            start = time.perf_counter()
            try:
                operator = NurbsElasticity2D(filename, patch_id=args.patch_id, degree=degree,
                                              refinement=level, young=args.young,
                                              poisson=args.poisson, plane=args.plane)
                # This benchmark's symmetry edges must be the coordinate axes;
                # the unloaded inner boundary must be the unit circle.
                bottom = operator.edge_data(1, 0)[1]
                left = operator.edge_data(1, 1)[1]
                inner = operator.edge_data(0, 0)[1]
                if (not np.allclose(bottom[:, 1], 0, atol=1e-10)
                        or not np.allclose(left[:, 0], 0, atol=1e-10)
                        or not np.allclose(np.sum(inner*inner, axis=1), 1, atol=1e-10)):
                    raise ValueError('Use a first-quadrant plate with a unit circular inner edge '
                                     'and symmetry edges eta=0 on y=0, eta=1 on x=0')
                coefficients = operator.solve(traction)
            except (ValueError, RuntimeError) as exc:
                parser.error(str(exc))
            error = operator.errors(coefficients, exact)
            h = max(np.max(np.diff(np.unique(knots))) for knots in operator.knots)
            orders = {}
            for name in ('sigma_xx', 'principal'):
                value = error['L2_'+name]
                orders['order_'+name] = (np.log(previous['L2_'+name]/value)/np.log(previous['h']/h)
                                         if previous is not None and value > 0 and previous['L2_'+name] > 0
                                         else np.nan)
            row = dict(degree=degree, refinement=level, elements_x=operator.nelements[0],
                       elements_y=operator.nelements[1], free_dofs=len(operator.free), h=h,
                       **error, **orders, energy_squared=float(coefficients @ (operator.stiffness @ coefficients)),
                       compliance=float(coefficients @ operator.load),
                       relative_residual=operator.relative_residual, seconds=time.perf_counter()-start)
            rows.append(row)
            previous = row
            nx, ny = operator.nelements
            print(f'{degree:2d}  {nx:3d} x {ny:<3d}   {row["free_dofs"]:7d}    '
                  f'{row["L2_sigma_xx"]:12.5e} {row["order_sigma_xx"]:6.2f}   '
                  f'{row["L2_principal"]:12.5e} {row["order_principal"]:6.2f}   '
                  f'{row["energy_squared"]:12.5e}')
        if args.export or args.plot:
            export_prefix = Path(str(prefix)+f'_p{degree}') if args.degree_study else prefix
            export_solution(operator, coefficients, exact, args.nbpts, export_prefix,
                            plot=args.plot and degree == degrees[-1])
    csv_path = str(prefix)+'_convergence.csv'
    with open(csv_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Convergence data saved to {csv_path}')
    if args.convergence_plot or args.degree_study:
        plot_convergence(rows, prefix)
    return operator, coefficients, rows


if __name__ == '__main__':
    main()
