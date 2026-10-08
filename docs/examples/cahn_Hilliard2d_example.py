"""Periodic 2D Cahn–Hilliard solver with ParaView time-series output.

Example:
    PYTHONPATH=. python docs/examples/cahn_Hilliard2d_example.py \
        --nelements 8 --steps 10 --export --save-every 2

Use --plot to export and open the resulting .pvd file in ParaView.
Periodic reduction retains stencil storage; convert explicitly at the SciPy
solver boundary. Exported coefficients include their periodic copies.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import kron, linalg as sla

from pyrefiga import (
    SplineSpace, TensorSpace, StencilVector, apply_periodic,
    assemble_mass1D, compile_kernel, load_xml, pyref_multipatch,
    paraview_TimeSolutionMultipatch,
)
from gallery.gallery_section_09 import (
    assemble_matrix_ex03, assemble_vector_ex03, assemble_norm_ex01,
)

assemble2_stiffness = compile_kernel(assemble_matrix_ex03, arity=2)
assemble2_rhs = compile_kernel(assemble_vector_ex03, arity=1)
assemble_norm_l2 = compile_kernel(assemble_norm_ex01, arity=1)


def Proj_solve(V1, V2, V, alpha, rng=None):
    """Initialize the periodic field, mass matrix and GL free energy."""
    rng = np.random.default_rng() if rng is None else rng
    M1 = apply_periodic(V1, assemble_mass1D(V1))
    M2 = apply_periodic(V2, assemble_mass1D(V2))
    mass = kron(M1.tosparse(), M2.tosparse(), format='csr')
    independent_shape = (V1.nbasis-V1.degree, V2.nbasis-V2.degree)
    initial = (rng.random(independent_shape)-1.0)*0.05+0.63
    xh = apply_periodic(V, initial, [True, True], update=True)
    u = StencilVector(V.vector_space)
    u.from_array(V, xh)
    energy = assemble_norm_l2(V, fields=[u], value=[alpha]).toarray()[0]
    return u, xh, mass, energy


def Cahn_Hliard_solve(V1, V2, V, u, xh, dt, alpha, N_iter=None):
    """Perform one nonlinear step; return field, coefficients, residual, energy."""
    N_iter = 100 if N_iter is None else N_iter
    if N_iter < 1:
        raise ValueError('N_iter must be positive')
    tol = 1e-7
    periodic = [True, True]
    xu_n = xh.copy()
    u_f = StencilVector(V.vector_space)
    u_f.from_array(V, xu_n)
    for i in range(N_iter):
        stiffness = assemble2_stiffness(V, fields=[u_f], value=[dt, alpha])
        matrix = apply_periodic(V, stiffness, periodic)
        rhs = assemble2_rhs(V, fields=[u_f, u], value=[dt, alpha])
        rhs = apply_periodic(V, rhs, periodic)
        # Stencil objects are retained until the external solver call.
        sparse_matrix = matrix.tosparse().tocsr()
        b = -rhs.toarray()
        correction, info = sla.cgs(sparse_matrix, b, rtol=1e-30)
        if info != 0:
            # The very strict CGS tolerance can cause numerical breakdown.
            # Fall back to sparse LU rather than using an unconverged update.
            correction = sla.splu(sparse_matrix.tocsc()).solve(b)
        d_tx = apply_periodic(V, correction, periodic, update=True)
        xu_n += d_tx
        residual = np.max(np.abs(correction))
        if not np.isfinite(residual) or residual > 1e3:
            raise RuntimeError(f'Cahn–Hilliard nonlinear iteration diverged: {residual}')
        u_f.from_array(V, xu_n)
        if residual < tol:
            break
    else:
        raise RuntimeError(f'Cahn–Hilliard nonlinear solve did not converge in {N_iter} iterations')
    u.from_array(V, xu_n)
    print(f'Iterations: {i+1}; residual: {residual:.3e}')
    energy = assemble_norm_l2(V, fields=[u], value=[alpha]).toarray()[0]
    return u, xu_n, residual, energy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--nelements', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--dt', type=float, default=1e-8)
    parser.add_argument('--alpha', type=int, default=6000)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--nbpts', type=int, default=100, help='ParaView sampling points per direction')
    parser.add_argument('--save-every', type=int, default=100, help='Export interval; initial/final states are included')
    parser.add_argument('--export', action='store_true', help='Save a ParaView .pvd time series without opening it')
    parser.add_argument('--plot', action='store_true', help='Export and open the time series in ParaView')
    parser.add_argument('--output', default='figs/cahn_hilliard', help='Output path prefix')
    args = parser.parse_args(argv)
    if args.degree < 2 or args.nelements <= args.degree:
        parser.error('Use degree >= 2 and nelements > degree')
    if args.steps < 0 or args.dt <= 0 or args.max_iterations < 1:
        parser.error('Use steps >= 0, dt > 0 and max-iterations >= 1')
    if args.save_every < 1 or args.nbpts < 2:
        parser.error('Use save-every >= 1 and nbpts >= 2')

    grid = np.linspace(0, 1, args.nelements+1)
    V1 = SplineSpace(degree=args.degree, grid=grid, nderiv=2, periodic=True)
    V2 = SplineSpace(degree=args.degree, grid=grid, nderiv=2, periodic=True)
    V = TensorSpace(V1, V2)
    u, coefficients, mass, energy = Proj_solve(V1, V2, V, args.alpha, np.random.default_rng(args.seed))
    initial = coefficients.copy()
    independent = (slice(0, V1.nbasis-V1.degree), slice(0, V2.nbasis-V2.degree))
    times, energies, moments = [0.0], [energy], [0.0]
    export = args.export or args.plot
    saved_times = [0.0] if export else []
    saved_fields = [[coefficients.copy()]] if export else []
    print(f'Time: 0; GL energy: {energy:.8e}; L2 change squared: 0')
    for step in range(1, args.steps+1):
        u, coefficients, residual, energy = Cahn_Hliard_solve(
            V1, V2, V, u, coefficients, args.dt, args.alpha, args.max_iterations,
        )
        t = step*args.dt
        difference = (initial[independent]-coefficients[independent]).ravel()
        moment = float(difference @ (mass @ difference))
        times.append(t)
        energies.append(energy)
        moments.append(moment)
        print(f'Time: {t:.8e}; GL energy: {energy:.8e}; L2 change squared: {moment:.8e}')
        if export and (step % args.save_every == 0 or step == args.steps):
            saved_times.append(t)
            saved_fields.append([coefficients.copy()])

    if export:
        prefix = Path(args.output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        geometry = pyref_multipatch(load_xml('unitSquare.xml'), (0,))
        solutions = [{'name': 'Concentration', 'data': saved_fields, 'space': V}]
        paraview_TimeSolutionMultipatch(
            args.nbpts, geometry, LStime=saved_times, solution=solutions,
            filename=str(prefix), plot=args.plot,
        )
        np.savetxt(str(prefix)+'_history.csv', np.column_stack((times, energies, moments)),
                   delimiter=',', header='time,GL_free_energy,L2_change_squared', comments='')
    return u, coefficients, times, energies, moments


if __name__ == '__main__':
    main()
