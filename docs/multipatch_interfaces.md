# Multipatch interfaces and anisotropic diffusion

This guide describes the fixed 2D mapping path used by `StencilNitsche` in
[mulipatch2d_example.py](examples/mulipatch2d_example.py). It explains how patch
interfaces are detected, assembled and merged, and where to introduce a diffusion
tensor. The adaptive-mapping path uses separate legacy kernels.

## 1. Edge numbering and patch numbering

Every patch has its own parameters `(u, v)`. Edge numbers refer to these
parameters, rather than physical directions such as left or top.

| Edge | Parameter boundary | Normal parameter axis | Tangential parameter |
| --- | --- | --- | --- |
| 1 | `u = u_min` | `u` | `v` |
| 2 | `u = u_max` | `u` | `v` |
| 3 | `v = v_min` | `v` | `u` |
| 4 | `v = v_max` | `v` | `u` |

```text
                 edge 4
             +-----------+
      edge 1 |           | edge 2
             +-----------+
                 edge 3
```

`pyref_multipatch` detects common edges by comparing geometry control points,
including reversed ordering. It records each connection as:

```python
(source_patch, neighbor_patch, [source_edge, neighbor_edge])
```

Patch numbers in this tuple are **one-based positions in the selected patch
list**, not XML geometry IDs. For example:

```python
from pyrefiga import load_xml, pyref_multipatch

mp = pyref_multipatch(load_xml('triangle.xml'), (1, 2))
interface = mp.getInterfaces()[0]
print(interface)                         # (1, 2, [4, 2])
print(mp.isInterfaceReversed(interface)) # False
```

Here edge 4 of XML patch 1 meets edge 2 of XML patch 2. Their tangential
coordinates belong to different parameter axes, even though they describe the
same physical curve.

## 2. Matching points and physical normals

The assembler uses a normalized interface coordinate `s` in `[0, 1]`:

- The source patch is evaluated at `s` along its edge.
- The neighbor is evaluated at `s` for matching orientation, or `1-s` for
  reversed orientation.

The actual knot domains are used to convert `s` into each patch's parameter.
At every interface quadrature point, both sides must describe the same physical
point, opposite outward unit normals and the same line measure.

For a geometry mapping `F(u, v)` with Jacobian `J = dF/d(u,v)`, a reference
gradient becomes a physical gradient through

\[
\nabla_x N = J^{-T}\nabla_{u,v}N.
\]

The code stores gradients as rows, so the corresponding expression in
[interfaces.py](../pyrefiga/interfaces.py) is:

```python
gradient = derivatives @ inverse  # inverse = inv(J)
```

If `axis` is the edge's normal parameter axis, the physical outward unit normal
is proportional to `sign * J^{-T} e_axis`. The sign is negative for edges 1 and
3, and positive for edges 2 and 4. The physical line measure comes from the
geometry derivative along the tangential parameter.

The quadrature partition includes the union of both solution trace knot breaks
and both geometry trace knot breaks, reflecting the neighbor partition when
needed. Values, gradients, geometry and fluxes on both sides are evaluated at
the corresponding points of this common interface rule. The current Gauss
order is the maximum relevant solution/geometry degree plus one; rational or
rapidly varying coefficients may require more quadrature.
The current interface routine constructs this boundary rule independently of
the volume rule, rather than directly reusing `V.points`. Both interface sides
still use one common set of physical samples. See
[integration_rules.md](integration_rules.md) for the general quadrature conventions.

## 3. The Nitsche interface form

For scalar diffusion with coefficient one, let `n` be the source outward normal.
Define the jump and average normal derivative by

\[
[u]=u_P-u_Q,\qquad
\{\partial_n u\}=\tfrac12
\left(\nabla u_P\cdot n+\nabla u_Q\cdot n\right).
\]

The symmetric interface contribution is

\[
a_\Gamma(u,v)=\int_\Gamma
\left(
\gamma[u][v]
-\{\partial_n u\}[v]
-\{\partial_n v\}[u]
\right)\,ds.
\]

The neighbor evaluation initially uses its own outward normal, `n_Q = -n`.
Consequently, the arrays passed to the kernel are:

```python
jumps[q] = concatenate((trace_P, -trace_Q))
fluxes[q] = concatenate((outward_flux_P, -outward_flux_Q))
```

With `normalS = 0.5`,
`assemble_interface_quadrature` in [nitsche_core.py](../pyrefiga/nitsche_core.py)
evaluates, for each local trial/test pair:

```python
weight * (
    Kappa * jump_i * jump_j
    - normalS * (jump_i * flux_j + flux_i * jump_j)
)
```

This produces both patch diagonal blocks and the cross block. The same formula
handles all sixteen edge pairings; the geometry and orientation work happens
before the kernel is called.

The fixed-mapping path therefore uses:

```text
StencilNitsche._prepare_uniform_interfaces()
    -> interfaces.assemble_interface()
        -> interfaces._trace() on both patches
        -> nitsche_core.assemble_interface_quadrature()
```

It does **not** call the older `assemble_matrix_DiffSpacediagnitsche` and
`assemble_matrix_DiffSpaceoffdiagnitsche` functions. Adaptive mappings still use
the corresponding legacy functions in `adnitsche_core.py`.

## 4. Assembly, boundary lifting and DOF merging

The example follows this sequence:

1. Build the solution space `V` and geometry evaluation space `W`.
2. Assemble the prescribed boundary coefficient vectors `u_d`.
3. Construct `StencilNitsche(V, W, mp, u_d)` and call `assemble_nitsche()`.
4. Assemble each patch's volume stiffness and RHS, eliminate its Dirichlet
   edges and append the resulting blocks.
5. Call `nitsche_merge()` and `nitsche_merge_rhs()`, solve, then use
   `extract_sol()` to recover each patch solution.

The volume RHS must include the contribution from the prescribed coefficient
vector. `StencilNitsche` adds the corresponding interface contribution
`-A_interface @ u_d`. The solved unknown is a correction to the prescribed
vector.

`nitsche_merge()` identifies matching trace coefficients, reversing their
indices where necessary. This requires matching trace degrees, normalized
knots and rational weights. A shared tensor space can have different degrees
in its two directions, but a cross-axis interface must still have equal degrees
and matching bases in the two tangential directions being joined.

The result is a conforming system: trial and test traces are continuous, so the
Nitsche jump terms cancel after merging. The tests check this cancellation and
also verify flux consistency before merging. Independently refined,
nonmatching trace spaces need a separate weak-coupling workflow without this
DOF merge; the current path rejects them.

### Shared Dirichlet corners

A patch can have two interface edges at a corner that lies on the domain's
Dirichlet boundary through its neighboring patches. Edge flags alone cannot
express this isolated constraint.

For example, selecting `(0, 1, 2)` in `unitSquare.xml` gives an L-shaped union.
The corner `(1, 1)` belongs to all three patches. On XML patch 0, both incident
edges are interfaces; on the other two patches it touches exterior Dirichlet
edges.

`pyref_multipatch.getCornerGroups()` connects copies of vertices through
interface endpoints. `getDirichletCornerGroups()` marks a group prescribed if
any copy touches a Dirichlet edge. `propagate_dirichlet_corners()` then copies
the boundary value into every incident patch, checking that the donor values
agree.

`StencilNitsche` excludes isolated prescribed corner corrections from the
merged system (`new_id[dof] == -1`) and restores their prescribed values during
solution extraction. It leaves the rest of each interface free. A four-patch
interior junction remains free if none of its incident edges is Dirichlet.

## 5. Example: anisotropic diffusion

Consider

\[
-\nabla\cdot(K\nabla u)=f,\qquad
K=\begin{pmatrix}4&1\\1&2\end{pmatrix}.
\]

This constant tensor is symmetric positive definite. The volume form becomes

\[
a_K(u,v)=\int_\Omega \nabla v\cdot K\nabla u\,dx.
\]

The current generic fixed-mapping interface API implements `K = I`; it does
not yet accept a diffusion argument. The following snippets show the code
changes needed to prototype this constant tensor. They are not a runnable
`StencilNitsche(..., diffusion=K)` API.

### A. Change the volume stiffness

In [gallery_section_06.py](examples/gallery/gallery_section_06.py), the example
currently accumulates:

```python
v += (bi_x * bj_x + bi_y * bj_y) * wvol / J_mat[g1, g2]
```

Replace that accumulation with:

```python
K11, K12 = 4.0, 1.0
K21, K22 = 1.0, 2.0

Kbj_x = K11 * bj_x + K12 * bj_y
Kbj_y = K21 * bj_x + K22 * bj_y

v += (bi_x * Kbj_x + bi_y * Kbj_y) * wvol / J_mat[g1, g2]
```

In this particular kernel, `bi_x`, `bi_y`, `bj_x`, and `bj_y` are cofactor
gradient numerators, before division by the Jacobian determinant. The existing
`wvol / J_mat` factor accounts for the two gradient transformations and the
volume measure. Keep that factor; do not divide these gradients a second time.

For a spatially varying tensor, evaluate its entries at the physical volume
quadrature point. The existing
[anisotropic gallery](anisotropic_diffusion/gallery/gallery_section_04.py)
already contains volume tensor contractions that can serve as a reference.

### B. Change the interface flux on both patches

In `interfaces._trace()`, replace the isotropic flux calculation with:

```python
K = np.array([[4.0, 1.0], [1.0, 2.0]])
gradient = derivatives @ inverse
flux = (gradient @ K.T) @ normal
```

This computes each basis function's outward conormal flux

\[
q_n(N)=n\cdot K\nabla N.
\]

The transpose is required by the row-gradient convention. For this symmetric
tensor `K.T == K`, but writing the transpose makes the convention explicit.
For a vertical boundary with `n = (1, 0)`, the flux is `4*N_x + N_y`; for a
horizontal boundary with `n = (0, 1)`, it is `N_x + 2*N_y`.

For variable or patch-dependent diffusion, evaluate the appropriate tensor at
the physical `point` already computed in `_trace()`. Retain the neighbor flux
sign reversal in `assemble_interface()`. That sign converts the neighbor's
outward flux to the common source-normal convention.

`assemble_interface_quadrature` needs no change to its consistency-term
formula: it receives these conormal fluxes instead of ordinary normal
derivatives. For discontinuous coefficients, flux continuity means

\[
n_P\cdot K_P\nabla u_P+n_Q\cdot K_Q\nabla u_Q=0.
\]

### C. Scale the penalty consistently

The diffusion tensor `K` and the scalar Nitsche penalty `Kappa` are different
quantities. Changing `Kappa` alone does not introduce anisotropic diffusion.

The current penalty is a global scalar based on spline degrees and basis
counts. For a simple constant-tensor prototype, a conservative diffusion scaling
is to multiply that scalar by the largest eigenvalue of `K`, **before any
interface assembly**, for example:

```python
Ni = StencilNitsche(V, W, mp, u_d)
K = np.array([[4.0, 1.0], [1.0, 2.0]])
Ni.Kappa *= np.linalg.eigvalsh(K).max()
Ni.assemble_nitsche()
```

This scales the existing penalty; it is not a replacement for local
stabilization on strongly graded meshes. A suitable local design is

\[
\gamma(s)=C\max\left(
\frac{p_P^2}{h_{n,P}}n^TK_Pn,
\frac{p_Q^2}{h_{n,Q}}n^TK_Qn
\right),
\]

where `h_n` is a physical element size normal to the interface and `p` is the
normal-direction solution degree. The constant must be large enough for the
applicable trace bound. Implementing this design requires passing a penalty
array indexed by quadrature point into the kernel instead of its scalar
`Kappa`. Large coefficient contrasts may also benefit from weighted flux
averages; the current kernel uses the arithmetic average.

### D. Update the RHS and boundary lifting

For the manufactured solution

\[
u=\sin(\pi x)\sin(\pi y),
\]

the tensor above gives

\[
f=-\nabla\cdot(K\nabla u)
=6\pi^2\sin(\pi x)\sin(\pi y)
-2\pi^2\cos(\pi x)\cos(\pi y).
\]

Use this forcing in the RHS assembler, while retaining `u` as the Dirichlet
data and exact solution. Any volume boundary-lifting contraction must use the
same tensor `K`; changing only the forcing or stiffness leaves the assembled
problem inconsistent. The generic interface lifting automatically uses the
updated interface blocks. For variable `K`, the divergence also includes
derivatives of its entries.

For example, with a true physical row gradient `grad_ud` of the prescribed
field, a volume RHS contribution has the form:

```python
rhs_i += (
    f * N_i - grad_N_i @ K @ grad_ud
) * physical_volume_weight
```

For this symmetric tensor, this is the lifted form
`f*N_i - grad_N_i · (K*grad_ud)`. Use the Jacobian conventions of the selected
volume kernel when translating this expression into its local variables.

## 6. Checks after changing the operator

- With `K = I`, recover the existing isotropic matrices and solutions.
- Verify symmetry for symmetric `K`.
- Verify that interface terms cancel after merging matching continuous traces.
- Check a linear manufactured solution before merging as well; cancellation
  alone can hide incorrect fluxes.
- Check the anisotropic manufactured forcing above and measure convergence.
- Include reversed interfaces and nonzero prescribed shared corners.

Relevant regression tests are
[test_nitsche_all_interfaces.py](../pyrefiga/tests/test_nitsche_all_interfaces.py),
[test_nitsche_triangle_interfaces.py](../pyrefiga/tests/test_nitsche_triangle_interfaces.py)
and [test_nitsche_corner_dirichlet.py](../pyrefiga/tests/test_nitsche_corner_dirichlet.py).
They currently cover isotropic assembly; anisotropic checks must be added when
the diffusion extension is implemented.
