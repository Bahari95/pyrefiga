"""Periodic H(curl) IGA waves with leapfrog and optional ParaView output.

PYTHONPATH=. python docs/examples/curlcurl2d_example.py --nelements 8 --steps 20 --export
Use --frequency 3 to exercise a nonzero manufactured source J_e.
Author: M. BAHARI
"""

import argparse
from pathlib import Path
import numpy as np

from pyrefiga import apply_periodic, load_xml, pyref_multipatch, paraview_TimeSolutionMultipatch
from pyrefiga.maxwell import PeriodicCurlCurl2D, CurlCurlLeapfrog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--nelements', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--cfl', type=float, default=0.5, help='Fraction of the leapfrog time-step limit')
    parser.add_argument('--frequency', type=float, default=None, help='Manufactured temporal angular frequency')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Export and open ParaView')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--nbpts', type=int, default=80)
    parser.add_argument('--output', default='figs/curlcurl')
    args = parser.parse_args(argv)
    if args.steps < 0 or args.save_every < 1 or args.nbpts < 2 or not 0 < args.cfl < 1:
        parser.error('Use steps >= 0, save-every >= 1, nbpts >= 2 and 0 < cfl < 1')
    try:
        op = PeriodicCurlCurl2D(args.degree, args.nelements, args.epsilon, args.mu)
    except ValueError as exc:
        parser.error(str(exc))
    natural_frequency = 2*np.pi/np.sqrt(args.epsilon*args.mu)
    frequency = natural_frequency if args.frequency is None else args.frequency
    if not np.isfinite(frequency) or frequency < 0:
        parser.error('frequency must be finite and nonnegative')

    def exact(x, y, time):
        amplitude = np.cos(frequency*time)
        return amplitude*np.sin(2*np.pi*y), amplitude*np.sin(2*np.pi*x)

    def source(x, y, time):
        ux, uy = exact(x, y, time)
        scale = args.epsilon*(natural_frequency**2-frequency**2)
        return scale*ux, scale*uy

    try:
        state = CurlCurlLeapfrog(op, op.project(exact),
                                dt=args.cfl*op.dt_limit if args.dt is None else args.dt,
                                source=None if args.frequency is None else source)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'H(curl) degrees: {op.component_spaces[0].degree}, {op.component_spaces[1].degree}')
    print(f'Independent DOFs: {len(state.u)}; dt: {state.dt:.8e}; limit: {op.dt_limit:.8e}')
    export = args.export or args.plot
    times, ux_data, uy_data, curl_data, history = [], [], [], [], []

    def record(save):
        error = op.l2_error(state.u, exact, state.time)
        energy = state.energy()
        history.append((state.time, energy, error))
        if save:
            ux, uy = op.expand(state.u)
            times.append(state.time)
            ux_data.append([ux.tensor.copy()])
            uy_data.append([uy.tensor.copy()])
            curl_data.append([apply_periodic(op.curl_space, op.curl @ state.u, update=True)])
        return energy, error

    initial_energy, initial_error = record(export)
    for step in range(1, args.steps+1):
        state.step()
        energy, error = record(export and (step % args.save_every == 0 or step == args.steps))
    print(f'Time: {state.time:.8e}; L2 error: {history[-1][2]:.6e} (initial: {initial_error:.6e})')
    print(f'Discrete energy: {initial_energy:.8e} -> {history[-1][1]:.8e}')
    if export:
        prefix = Path(args.output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        geometry = pyref_multipatch(load_xml('unitSquare.xml'), (0,))
        solutions = [dict(name='u_x', data=ux_data, space=op.component_spaces[0]),
                     dict(name='u_y', data=uy_data, space=op.component_spaces[1]),
                     dict(name='curl_u', data=curl_data, space=op.curl_space)]
        paraview_TimeSolutionMultipatch(args.nbpts, geometry, LStime=times, solution=solutions,
                                       filename=str(prefix), plot=args.plot)
        np.savetxt(str(prefix)+'_history.csv', history, delimiter=',',
                   header='time,discrete_energy,L2_error', comments='')
    return op, state, np.asarray(history)


if __name__ == '__main__':
    main()
