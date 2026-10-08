# Periodic curl-curl waves with leapfrog

`pyrefiga.maxwell.PeriodicCurlCurl2D` assembles a 2D Cartesian curl-curl
operator. `CurlCurlLeapfrog` advances its coefficients in time. The initial
implementation supports periodic rectangles, nonuniform tensor grids and
positive **constant scalar** epsilon and mu. It does not yet support mapped
NURBS geometries, multipatch interfaces, spatial material variation or 3D.

## Curl convention and equation

Here `curl(u) = d_x u_y - d_y u_x`. The vector curl of a scalar q is
`(d_y q, -d_x q)`, the adjoint of scalar curl under periodic conditions.
With this convention the stable wave equation is

$$\epsilon u_{tt}+\operatorname{curl}(\mu^{-1}\operatorname{curl}u)=J_e,$$

and its weak form has the positive term
`integral (1/mu) curl(u) curl(v)`.
If your outer curl instead means `(-d_y q,d_x q)`, your displayed **minus**
sign gives precisely this same positive weak form. With the adjoint vector
curl defined above, a literal minus would give exponentially growing modes;
this solver uses the positive weak form.

## Periodicity and the de Rham spaces

Use the same Cartesian tangent when identifying opposite edges:

$$u_y(0,y)=u_y(1,y),\qquad u_x(x,0)=u_x(x,1).$$

Boundary traversal tangents can have opposite signs; these equations compare
components in the same Cartesian direction. The implementation constructs
the entire periodic de Rham complex, so both components are periodic in both
directions. This also supplies periodic material fluxes across the seams.

Starting with degrees `(px,py)` and simple interior knots, the spaces are

$$S^{p_x,p_y}\xrightarrow{\nabla}
S^{p_x-1,p_y}\times S^{p_x,p_y-1}
\xrightarrow{\mathrm{curl}} S^{p_x-1,p_y-1}.$$

Each degree reduction trims the first and last knot; continuity is reduced
in the differentiated direction. Degrees must be at least two in this
implementation, because the existing knot constructor does not handle the
degree-zero space required for degree-one H(curl).
The construction follows the tensor-product spline differential-form
approach described by [Buffa et al., Isogeometric Discrete Differential Forms](https://epubs.siam.org/doi/10.1137/100786708).

For independent periodic coefficients the one-dimensional derivative map is

$$ (D c)_j=\frac{p}{T_{j+p+1}-T_{j+1}}
(c_{j+1\bmod N}-c_j). $$

With C-order coefficient flattening,

$$G=\begin{bmatrix}D_x\otimes I_y\\I_x\otimes D_y\end{bmatrix},
\qquad C=\begin{bmatrix}-I_x\otimes D_y & D_x\otimes I_y\end{bmatrix}.$$

Thus `C @ G = 0`: gradients lie in the curl-curl nullspace. The periodic
rectangle also has constant harmonic fields. These modes are retained;
there is no artificial stiffness or gauge fixing for this time-dependent
problem. For source-free constant-material data, the weak divergence
constraint is preserved when it is satisfied by both initial displacement
and initial velocity.

The existing `assemble_mass1D` and `apply_periodic` retain stencil storage
for the univariate masses. At the block-operator boundary they are converted
to SciPy sparse matrices. This is necessary because the vector components
have different degrees and the curl mixes them; the assembled vector
operator is not a single scalar `StencilMatrix`.

$$M_\epsilon=\epsilon\,\mathrm{diag}
(M_x^-\otimes M_y^+,M_x^+\otimes M_y^-),
\qquad A=C^T\bigl((M_x^-\otimes M_y^-)/\mu\bigr)C.$$

The mass is consistent, not lumped. Its inverse is applied using cached
one-dimensional sparse factorizations and tensor solves.

## Time integration and source

Leapfrog stores `u` at time n*dt and `velocity_half` at time (n+1/2)*dt.
The first half-step includes half the initial acceleration, giving second
order accuracy for both nonzero initial velocity and a source:

$$v^{1/2}=v^0+\frac{\Delta t}{2}M^{-1}(F^0-Au^0),$$
$$u^{n+1}=u^n+\Delta t\,v^{n+1/2},$$
$$v^{n+3/2}=v^{n+1/2}+\Delta t\,M^{-1}(F^{n+1}-Au^{n+1}).$$

The solver rejects time steps at or above `dt_limit = 2/sqrt(lambda_max)`,
where `lambda_max` is the largest eigenvalue of `M^-1 A`, computed using
the tensor-product spectrum. `state.energy()` reports the conserved
half-time discrete energy for zero source, rather than a continuous energy
computed from values at different times. For nonzero source it need not be
constant. Leapfrog and spline complexes for Maxwell time integration are
also discussed in [Kapidani and Vázquez, High order geometric methods with splines](https://arxiv.org/abs/2302.04979).

```python
import numpy as np
from pyrefiga.maxwell import PeriodicCurlCurl2D, CurlCurlLeapfrog

op = PeriodicCurlCurl2D(degree=2, nelements=16, epsilon=2.0, mu=1.0)
u0 = op.project(lambda x, y, t: (np.sin(2*np.pi*y), 0.0))
v0 = np.zeros_like(u0)

# J_e is a physical vector field, not an array of coefficients.
def J_e(x, y, t):
    return np.sin(t)*np.sin(2*np.pi*y), 0.0

state = CurlCurlLeapfrog(op, u0, v0, dt=0.5*op.dt_limit, source=J_e)
for _ in range(100):
    state.step()

# Full coefficient copies for existing evaluation/export routines.
ux, uy = op.expand(state.u)  # two StencilVectors with different spaces
```

`op.load(J_e,t)` integrates the source against each component basis;
`op.project(field,t)` computes a componentwise L2 projection.
`op.component_spaces` contains the corresponding two `TensorSpace` objects.
`op.evaluate(u,x,y)` samples each component on a tensor grid.

## Run and export the example

```bash
PYTHONPATH=. python docs/examples/curlcurl2d_example.py \
  --degree 2 --nelements 8 --steps 20 --export --save-every 5
```

The example uses a divergence-free periodic Fourier wave with known exact
solution. `--epsilon` and `--mu` change the wave speed; `--dt` overrides the
automatic time step. `--frequency 3` changes its temporal frequency and
constructs the corresponding nonzero manufactured source. It reports the
L2 error and discrete energy. ParaView output uses the existing multipatch
exporter, with scalar arrays `u_x`, `u_y` and `curl_u`, each evaluated in its
own space. Use ParaView's Calculator to combine the component arrays into
a vector. Output includes physical time stamps in the `.pvd` collection and
a CSV history. `--plot` exports and opens ParaView.
