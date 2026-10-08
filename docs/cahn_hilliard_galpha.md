# Adaptive generalized-alpha Cahn–Hilliard example

The [example](examples/cahn_Hilliard2d_Galpha_example.py) uses the existing
matrix, residual, initial-derivative and energy kernels in
`gallery/gallery_section_08.py`. Periodic folding returns stencil matrices
and vectors; conversion to SciPy sparse storage occurs at the linear solve.
`alpha` is the model parameter, passed explicitly to each trial. It is
different from the generalized-alpha integration parameters below.

## Generalized-alpha update

For the first-order semidiscrete equation `M c_dot + R(c) = 0`, the method
solves the residual at

$$c_{n+\alpha_f}=c_n+\alpha_f(c_{n+1}-c_n),$$
$$\dot c_{n+\alpha_m}=\dot c_n+\alpha_m(\dot c_{n+1}-\dot c_n),$$

with the kinematic update

$$c_{n+1}=c_n+\Delta t[(1-\gamma)\dot c_n+\gamma\dot c_{n+1}].$$

The parameters are

$$\alpha_m=\frac{3-\rho_\infty}{2(1+\rho_\infty)},\qquad
\alpha_f=\frac1{1+\rho_\infty},\qquad
\gamma=\frac12+\alpha_m-\alpha_f.$$

`--rho-inf` sets the high-frequency spectral radius between zero and one;
the default is 0.5. This is the first-order generalized-alpha formulation of
[Jansen, Whiting and Hulbert](https://www.sciencedirect.com/science/article/abs/pii/S0045782500002036).
The same value is passed to the gallery tangent and the predictor.
Newton corrects the endpoint derivative, so the tangent is
`alpha_m*M + dt*alpha_f*gamma*R'(c_stage)`.

## Adaptive time step

Generalized-alpha supplies damping; an additional step-doubling controller
selects the time step. A trial compares one step of length `h` with two
steps of length `h/2`, starting from the same accepted state. Its normalized
concentration error is

$$e=\frac{\|c_{\mathrm{half,half}}-c_{\mathrm{full}}\|_M}
{3[\mathrm{atol}+\mathrm{rtol}\max(\|c_n\|_M,\|c_{\mathrm{half,half}}\|_M)]},
\qquad \|c\|_M=\sqrt{c^T M c}.$$

The factor three is the step-doubling estimate for a second-order method.
Only independent periodic coefficients enter these norms. This controls a
local estimate in concentration, rather than a global error guarantee or
an estimate for the derivative. Accepted trials have `e <= 1`; the two-half
solution and its endpoint derivative become the next state. The next step
uses a safety factor of 0.9 and an `e^(-1/3)` multiplier bounded between
0.2 and 2, with user-defined minimum and maximum steps. Each normal adaptive
trial costs three G-alpha solves.

Nonlinear failures also reject a trial and halve its step. Rejected trials
leave the accepted concentration, derivative and physical time unchanged.
Failure at the minimum step or after twenty rejections raises an error.
Newton checks the residual after corrections and fails explicitly at its
iteration limit; CGS failure triggers a sparse LU fallback.

## Running and ParaView

```bash
PYTHONPATH=. python docs/examples/cahn_Hilliard2d_Galpha_example.py \
  --nelements 8 --steps 20 --seed 7 --export --save-every 5
```

Useful options:

- `--dt`: initial step, default `1e-8`.
- `--dt-min` / `--dt-max`: bounds, defaults `1e-12` and `1e-4`.
- `--atol` / `--rtol`: absolute L2 and relative temporal tolerances,
  defaults `1e-6` and `1e-3`.
- `--nonlinear-tol`: absolute maximum assembled residual, default `1e-9`.
- `--t-end`: optional physical stopping time; the last step is shortened
  to reach it, including when the remainder is smaller than `dt-min`.
  `--steps` still limits the number of accepted steps.
- `--fixed-dt`: disable step doubling and use one G-alpha solve per step.
- `--plot`: export and open ParaView.

The exporter uses the accepted, generally nonuniform physical times in its
`.pvd` collection. Initial and final snapshots are included. The CSV records
time, accepted `dt`, GL free energy, squared L2 change from the initial
concentration, nonlinear residual, normalized temporal error and rejection
count. The initial coefficients are copied, so the reference concentration
is preserved. The script has a `main` guard and no blocking Matplotlib plots
or intermediate image/GIF generation.
