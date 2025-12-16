Integration Rules in PyRef IGA
===============================

This document specifies the integration (quadrature) rules used in the **pyrefiga** library.

Rule 1: Same Quadrature Grid per Parametric Direction
-----------------------------------------------------

For a given patch, in each parametric direction (ξ, η, ζ), **all operators must be evaluated on the same quadrature grid and with the same weights**, independent of:

- the geometry representation (B-spline, NURBS, etc.)
- the FE space used for the solution (B-spline, NURBS, or hierarchical splines)

Objects evaluated on the grid include:

- Matrices (mass, stiffness, etc.)
- Geometry Jacobians
- Basis functions and derivatives
- Weak interface terms (e.g., Nitsche enforcement)

.. note::
    The quadrature points and weights are independent of the geometry and FE basis.
    The integration grid cannot change during the evaluation of a single operator.

Importance
----------
In the spline space, \texttt{nelements} refers to \(\texttt{len(mesh)}\), where the mesh is used for numerical integration rather than for defining the knot grid. 

If the user provides mesh, spans are automatically extended from a shape of \((n_e)\) to \((n_e, n_q)\), where \(n_q\) denotes the number of quadrature points per element. see poisson2d_example.py

Using a consistent quadrature grid ensures:

- Correct evaluation of geometry and FE quantities
- Stable and accurate weak enforcement (e.g., Nitsche coupling, DDM)
- Optimal convergence rates (exact quadrature for polynomial basis)

Additional implementation benefits:

- Less memory usage: one grid per direction reused everywhere
- Simpler and safer assembly: no grid conversion or mapping bookkeeping
- Easier extension to multipatch and adaptive refinement: rule applies independently per patch

Rule 2: Quadrature Order Depends on Spline Degree
-------------------------------------------------

The **minimum quadrature order** should be chosen based on the **maximum degree of the FE solution and the geometry mapping**:

.. code-block:: text

    quadrature_order ≥ max(p_solution, p_geometry) + 1
