# Plate with a circular hole

The example follows the linear-elasticity validation in section 4 of
[Blanchard, Duvigneau, Vuong and Simeon, RR-8111 (2012)](https://inria.hal.science/hal-00743199/document),
using the weak form in equations (3) and (14). It solves the plate benchmark;
the shape-optimization algorithm in later sections is outside this example.

The default `fields/elasticity.xml` describes the first-quadrant portion of
a square plate of outer size 4 with a circular hole of radius 1. This is a
reflected version of the quarter plate in figure 2. The inner edge is free,
`u_y=0` on `y=0`, and `u_x=0` on `x=0`. The outer square boundary receives
`sigma_exact @ n`, where `sigma_exact` is the Kirsch stress for remote
x-directed tension. These exact tractions model an infinite plate on the
truncated domain; they are not constant tractions on the finite square.

Defaults retain the example's plane-stress convention, `E=1e5`, `nu=0.3`
and remote tension 10. The paper's figure specifies `E` and `nu`; plane
stress and traction magnitude are explicit choices in this implementation.
`--plane strain` selects the corresponding 2D constitutive law instead.

```bash
PYTHONPATH=. python docs/examples/elasticity2d_example.py --export
```

This solves three meshes, writes `figs/elasticity_convergence.csv`, and
exports the finest solution to `figs/elasticity.vtm`. Replace `--export`
with `--plot` to launch ParaView. Without either flag, only the convergence
CSV is written. `--nbpts` controls visualization resolution independently
of the analysis mesh.

For a degree comparison corresponding to the layout of figure 3:

```bash
PYTHONPATH=. python docs/examples/elasticity2d_example.py \
  --degree-study --refinement 2 --levels 4 --export \
  --output figs/elasticity_plate
```

The study uses degrees 2, 3 and 4, saves PNG/PDF convergence plots and a CSV,
and exports one finest-mesh solution per degree. `--degree` selects a single
degree when `--degree-study` is absent. `--refinement` is the initial number
of dyadic subdivisions of each original nonempty knot span; `--levels`
controls the number of successive refinements. The reported element counts
are therefore `(2**level, 2**(level+1))` for the supplied plate patch.

Both `L2_sigma_xx` (Cartesian stress) and `L2_principal` (largest in-plane
principal stress) are reported and plotted separately. This distinguishes
the old example's `sigma_xx` error from the principal-stress label in the
paper's figure 3. These are computed results, not digitized paper data.
The CSV also contains the stress-based energy error, `u.T @ K @ u`,
compliance `u.T @ F`, the free-system residual, timings and observed orders.
The energy norm squared should increase under nested refinement, as
discussed in section 4; compliance and energy norm squared should agree
for the homogeneous symmetry constraints.

`pyrefiga.elasticity.NurbsElasticity2D` assembles the sparse block matrix
directly from physical basis gradients:

$$K_{11}=(\lambda+2\mu)D_x^T W D_x+\mu D_y^T W D_y,$$
$$K_{22}=\mu D_x^T W D_x+(\lambda+2\mu)D_y^T W D_y,$$
$$K_{12}=\lambda D_x^T W D_y+\mu D_y^T W D_x,\qquad K_{21}=K_{12}^T.$$

Here `W` includes the physical Jacobian determinant and Gauss weights.
The traction integral includes outward physical normals and boundary arc
length. Dirichlet coefficients are eliminated component by component,
and the reduced sparse system is solved directly with a residual check.

The plate's weights are not separable. Consequently the implementation
uses the full rational basis `R_ij=w_ij B_i B_j / sum(w_kl B_k B_l)`,
including its quotient-rule derivatives. It does not approximate the weight
net by the product of its first row and column. Degree elevation and knot
insertion operate on homogeneous control points, preserving the exact
geometry and the original C1 knot line. Newly inserted knots are simple.
The square's outer corner is a singular parameter point: stresses sampled
exactly there are NaN; all integration points remain regular. The default
even visualization resolution avoids that point.
