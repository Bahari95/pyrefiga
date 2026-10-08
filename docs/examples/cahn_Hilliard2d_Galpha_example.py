"""Periodic Cahn–Hilliard with generalized-alpha and adaptive time steps.

PYTHONPATH=. python docs/examples/cahn_Hilliard2d_Galpha_example.py \
    --nelements 8 --steps 20 --export --save-every 5

G-alpha controls numerical damping; step doubling controls the time step.
Use --fixed-dt for one G-alpha solve per step. --plot exports and opens ParaView.
Author: M. BAHARI
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import kron, linalg as sla
from pyrefiga import (
    SplineSpace, TensorSpace, StencilVector, apply_periodic, assemble_mass1D,
    compile_kernel, load_xml, pyref_multipatch, paraview_TimeSolutionMultipatch,
)
from gallery.gallery_section_08 import (
    assemble_vector_ex02, assemble_matrix_ex03, assemble_vector_ex03, assemble_norm_ex01,
)

assemble_dtrhs = compile_kernel(assemble_vector_ex02, arity=1)
assemble2_stiffness = compile_kernel(assemble_matrix_ex03, arity=2)
assemble2_rhs = compile_kernel(assemble_vector_ex03, arity=1)
assemble_norm_l2 = compile_kernel(assemble_norm_ex01, arity=1)


class NonlinearConvergenceError(RuntimeError):
    """A trial may be retried with a smaller time step."""


def generalized_alpha_parameters(rho_inf=0.5):
    if not np.isfinite(rho_inf) or not 0 <= rho_inf <= 1:
        raise ValueError('rho_inf must lie in [0, 1]')
    alpha_m = 0.5*(3-rho_inf)/(1+rho_inf)
    alpha_f = 1/(1+rho_inf)
    gamma = 0.5+alpha_m-alpha_f
    return alpha_m, alpha_f, gamma


def Proj_solve(V1, V2, V, alpha, rng=None):
    """Initialize concentration and its consistent PDE time derivative."""
    rng = np.random.default_rng() if rng is None else rng
    m1 = apply_periodic(V1, assemble_mass1D(V1))
    m2 = apply_periodic(V2, assemble_mass1D(V2))
    mass = kron(m1.tosparse(), m2.tosparse(), format='csr')
    shape = (V1.nbasis-V1.degree, V2.nbasis-V2.degree)
    xh = apply_periodic(V, (rng.random(shape)-1)*.05+.63, update=True)
    u = StencilVector(V.vector_space)
    u.from_array(V, xh)
    rhs = apply_periodic(V, assemble_dtrhs(V, fields=[u], value=[alpha]))
    derivative = sla.splu(mass.tocsc()).solve(-rhs.toarray())
    txh = apply_periodic(V, derivative, update=True)
    ut = StencilVector(V.vector_space)
    ut.from_array(V, txh)
    energy = float(assemble_norm_l2(V, fields=[u], value=[alpha]).toarray()[0])
    return u, xh, ut, txh, mass, energy


def Cahn_Hliard_solve(V1, V2, V, u, ut, xh, txh, dt, alpha,
                      N_iter=None, rho_inf=0.5, nonlinear_tol=1e-9):
    """Return a G-alpha trial without modifying the input fields or arrays.

    Newton's unknown is the endpoint time derivative. Concentration and
    derivative satisfy x_new=x_old+dt*((1-gamma)*tx_old+gamma*tx_new).
    """
    N_iter = 100 if N_iter is None else N_iter
    if N_iter < 1 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(nonlinear_tol) or nonlinear_tol <= 0:
        raise ValueError('Use positive dt, nonlinear_tol and N_iter')
    alpha_m, alpha_f, gamma = generalized_alpha_parameters(rho_inf)
    xu = xh.copy()
    xtu = ((gamma-1)/gamma)*txh
    u_f, u_m = StencilVector(V.vector_space), StencilVector(V.vector_space)
    # Check the updated residual, including after the last permitted correction.
    for iteration in range(N_iter+1):
        u_f.from_array(V, xh+alpha_f*(xu-xh))
        u_m.from_array(V, txh+alpha_m*(xtu-txh))
        rhs = apply_periodic(V, assemble2_rhs(V, fields=[u_f, u_m], value=[alpha]))
        b = -rhs.toarray()
        residual = float(np.max(np.abs(b)))
        if not np.isfinite(residual):
            raise NonlinearConvergenceError('Nonfinite G-alpha residual')
        if residual <= nonlinear_tol:
            break
        if iteration == N_iter:
            raise NonlinearConvergenceError(f'G-alpha did not converge in {N_iter} iterations (residual {residual:.3e})')
        matrix = apply_periodic(V, assemble2_stiffness(V, fields=[u_f], value=[dt, alpha, rho_inf]))
        # Periodic assembly remains in stencil storage until the SciPy solve.
        a = matrix.tosparse().tocsr()
        correction, info = sla.cgs(a, b, rtol=1e-10, atol=1e-14)
        if info != 0 or not np.isfinite(correction).all():
            try:
                correction = sla.splu(a.tocsc()).solve(b)
            except RuntimeError as exc:
                raise NonlinearConvergenceError('G-alpha linear solve failed') from exc
        if not np.isfinite(correction).all():
            raise NonlinearConvergenceError('Nonfinite G-alpha correction')
        correction = apply_periodic(V, correction, update=True)
        xtu += correction
        xu += gamma*dt*correction
    candidate_u, candidate_ut = StencilVector(V.vector_space), StencilVector(V.vector_space)
    candidate_u.from_array(V, xu)
    candidate_ut.from_array(V, xtu)
    energy = float(assemble_norm_l2(V, fields=[candidate_u], value=[alpha]).toarray()[0])
    if not np.isfinite(energy):
        raise NonlinearConvergenceError('Nonfinite G-alpha energy')
    return candidate_u, xu, candidate_ut, xtu, residual, energy


def adaptive_step(V1, V2, V, u, ut, xh, txh, mass, dt, alpha,
                  dt_min=1e-12, dt_max=1e-4, atol=1e-6, rtol=1e-3,
                  N_iter=100, rho_inf=0.5, nonlinear_tol=1e-9, max_rejections=20):
    """Compare one full step with two half steps; accept the two-half result.

    Only concentration enters the L2 error estimate. The endpoint derivative
    from the accepted half steps is retained for the next G-alpha predictor.
    Returns (trial, accepted_dt, next_dt, normalized_error, rejected_trials).
    """
    if not all(np.isfinite(c) and c > 0 for c in (dt, dt_min, dt_max, atol)) or dt_min > dt_max:
        raise ValueError('Use 0 < dt_min <= dt_max, dt > 0 and atol > 0')
    if not np.isfinite(rtol) or rtol < 0 or max_rejections < 0:
        raise ValueError('Use rtol >= 0 and max_rejections >= 0')
    generalized_alpha_parameters(rho_inf)
    h = float(np.clip(dt, dt_min, dt_max))
    independent = (slice(0, V1.nbasis-V1.degree), slice(0, V2.nbasis-V2.degree))

    def norm(coefficients):
        c = coefficients[independent].ravel()
        return float(np.sqrt(max(float(c @ (mass @ c)), 0)))

    def trial(a, b, c, d, delta):
        return Cahn_Hliard_solve(V1, V2, V, a, b, c, d, delta, alpha,
                                 N_iter=N_iter, rho_inf=rho_inf, nonlinear_tol=nonlinear_tol)

    for rejected in range(max_rejections+1):
        try:
            full = trial(u, ut, xh, txh, h)
            half = trial(u, ut, xh, txh, h/2)
            fine = trial(half[0], half[2], half[1], half[3], h/2)
            error = norm(fine[1]-full[1])/(3*(atol+rtol*max(norm(xh), norm(fine[1]))))
            if not np.isfinite(error):
                raise NonlinearConvergenceError('Nonfinite step-doubling error')
            factor = 2.0 if error == 0 else float(np.clip(.9*error**(-1/3), .2, 2.0))
            if error <= 1:
                return fine, h, float(np.clip(h*factor, dt_min, dt_max)), error, rejected
            reason = f'normalized temporal error {error:.3e} exceeds one'
            factor = min(factor, .9)
        except NonlinearConvergenceError as exc:
            reason, factor = str(exc), .5
        if h <= dt_min or rejected == max_rejections:
            raise RuntimeError(f'Adaptive G-alpha step failed at dt={h:.3e}: {reason}')
        h = max(dt_min, h*factor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--nelements', type=int, default=64)
    parser.add_argument('--steps', type=int, default=1000, help='Maximum accepted steps')
    parser.add_argument('--t-end', type=float, default=None, help='Optional final physical time')
    parser.add_argument('--dt', type=float, default=1e-8, help='Initial time step')
    parser.add_argument('--dt-min', type=float, default=1e-12)
    parser.add_argument('--dt-max', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-6, help='Absolute concentration L2 error tolerance')
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--rho-inf', type=float, default=.5, help='G-alpha high-frequency spectral radius')
    parser.add_argument('--fixed-dt', action='store_true', help='Disable time-step adaptation')
    parser.add_argument('--alpha', type=int, default=6000)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--nonlinear-tol', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--nbpts', type=int, default=80)
    parser.add_argument('--save-every', type=int, default=10, help='Interval in accepted steps')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Export and open ParaView')
    parser.add_argument('--output', default='figs/cahn_hilliard_galpha')
    args = parser.parse_args(argv)
    if args.degree < 2 or args.nelements <= args.degree or args.alpha <= 0:
        parser.error('Use degree >= 2, nelements > degree and alpha > 0')
    if args.steps < 0 or args.max_iterations < 1 or args.save_every < 1 or args.nbpts < 2:
        parser.error('Use steps >= 0, max-iterations/save-every >= 1 and nbpts >= 2')
    if not all(np.isfinite(c) and c > 0 for c in (args.dt, args.dt_min, args.dt_max, args.atol, args.nonlinear_tol)):
        parser.error('Time-step bounds and absolute/nonlinear tolerances must be finite and positive')
    if not args.dt_min <= args.dt <= args.dt_max or not np.isfinite(args.rtol) or args.rtol < 0:
        parser.error('Use dt-min <= dt <= dt-max and rtol >= 0')
    if not np.isfinite(args.rho_inf) or not 0 <= args.rho_inf <= 1:
        parser.error('Use 0 <= rho-inf <= 1')
    if args.t_end is not None and (not np.isfinite(args.t_end) or args.t_end <= 0):
        parser.error('t-end must be finite and positive')

    grid = np.linspace(0, 1, args.nelements+1)
    V1 = SplineSpace(args.degree, grid=grid, nderiv=2, periodic=True)
    V2 = SplineSpace(args.degree, grid=grid, nderiv=2, periodic=True)
    V = TensorSpace(V1, V2)
    u, xh, ut, txh, mass, energy = Proj_solve(V1, V2, V, args.alpha, np.random.default_rng(args.seed))
    initial = xh.copy()
    independent = (slice(0, V1.nbasis-V1.degree), slice(0, V2.nbasis-V2.degree))
    t, dt = 0.0, args.dt
    history = [(t, 0., energy, 0., 0., 0., 0)]
    export = args.export or args.plot
    saved_times = [t] if export else []
    saved_fields = [[xh.copy()]] if export else []
    print(f'Time: 0; GL energy: {energy:.8e}')
    for step in range(1, args.steps+1):
        remaining = np.inf if args.t_end is None else args.t_end-t
        if remaining <= 0:
            break
        h = min(dt, remaining)
        if t+h == t:
            raise RuntimeError('Time step is too small to advance physical time')
        if args.fixed_dt:
            result = Cahn_Hliard_solve(V1, V2, V, u, ut, xh, txh, h, args.alpha,
                                      args.max_iterations, args.rho_inf, args.nonlinear_tol)
            used_dt, error, rejected = h, 0., 0
        else:
            result, used_dt, dt, error, rejected = adaptive_step(
                V1, V2, V, u, ut, xh, txh, mass, h, args.alpha,
                dt_min=min(args.dt_min, h), dt_max=args.dt_max, atol=args.atol, rtol=args.rtol,
                N_iter=args.max_iterations, rho_inf=args.rho_inf, nonlinear_tol=args.nonlinear_tol,
            )
        u, xh, ut, txh, residual, energy = result
        # Only accepted steps advance time and enter diagnostics/export.
        t = args.t_end if args.t_end is not None and used_dt == remaining else t+used_dt
        difference = (initial[independent]-xh[independent]).ravel()
        moment = float(difference @ (mass @ difference))
        history.append((t, used_dt, energy, moment, residual, error, rejected))
        print(f'Step: {step}; time: {t:.8e}; dt: {used_dt:.3e}; error: {error:.3e}; '
              f'rejected: {rejected}; GL energy: {energy:.8e}')
        final = step == args.steps or (args.t_end is not None and t >= args.t_end)
        if export and (step % args.save_every == 0 or final):
            saved_times.append(t)
            saved_fields.append([xh.copy()])
    history = np.asarray(history)
    if export:
        prefix = Path(args.output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        geometry = pyref_multipatch(load_xml('unitSquare.xml'), (0,))
        solutions = [dict(name='Concentration', data=saved_fields, space=V)]
        paraview_TimeSolutionMultipatch(args.nbpts, geometry, LStime=saved_times,
                                       solution=solutions, filename=str(prefix), plot=args.plot)
        np.savetxt(str(prefix)+'_history.csv', history, delimiter=',',
                   header='time,dt,GL_free_energy,L2_change_squared,nonlinear_residual,temporal_error,rejected_trials',
                   comments='')
    return u, xh, ut, txh, mass, history


if __name__ == '__main__':
    main()
