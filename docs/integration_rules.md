# Integration Rules in pyrefiga

This document specifies the integration (quadrature) rules used in the pyrefiga library.

## 📌 Rule 1: Same Quadrature Grid per Parametric Direction

For one patch, in each parametric direction (ξ, η, ζ):
- All matrices (mass, stiffness)
- Geometry Jacobians
- Basis and derivative evaluations
- Weak interface terms (Nitsche)
  
🔹 **must use the exact same integration points and weights**.

⚠️ The integration grid cannot change between terms of the same operator.

## 🎯 Importance

Using the same grid ensures:
- Symmetric matrices
- Stable Nitsche coupling
- Correct geometry evaluation
- Optimal convergence

Mixing grids leads to instability and incorrect results.

> NURBS weights do **not** modify the quadrature rule.

## 📌 Rule 2: Grid Depends on Degree

Minimum quadrature order:
