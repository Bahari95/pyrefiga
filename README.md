# pyrefiga: python + refinement + Isogeometric analysis

**pyrefiga** is a CAD and simulation library tailored for Isogeometric Analysis (IGA). It offers advanced capabilities, including r-adaptive mesh refinement, enabling accurate and efficient computations on complex geometries.

## License

This project is licensed under the GNU GPL v3.0, as it includes and modifies code from the original [simplines](https://github.com/ratnania/simplines) library.

© 2025 Mustapha Bahari. See the LICENSE file for full licensing details.



**Modifications in this version were made, including:**

    Support for NURBS basis functions in CAD and IGA

    Fast solvers

    Examples on Poisson, Cahn-Hilliard, and Elasticity problems ...

    It supports mappings defined by composition.

    Adaptation to complex 2D and 3D geometries

    Enhancements for multipatch domains (coming soon)

    Improvements to solver stability and adaptive mesh refinement

    Extended support for visualization (matplotlib and paraview) and performance diagnostics
    
## Clone the Repository

```bash
git clone https://github.com/Bahari95/pyrefiga.git
cd pyrefiga
```

## Install

**Standard mode**

```shell

    python3 -m pip install .

```

**Development mode**

```shell
    python3 -m pip install --user -e .
```
Please pyccelize the specific files and run them afterward to accelerate the computation of results and mesh adaptation using the following command:
**Pyccel**
```shell
    python3 setup.py run_pyccel
```
You can explore and run some tests in the "tests" folder or review the content in the "examples" folder.

You may work in the **newFolder** for your tasks, as it has been created automatically for you.

## Example

![PNG](https://github.com/Bahari95/pyrefiga/blob/main/r_adaptive_refinement/adaptive_meshes.png)

## Application 

    <img src="https://raw.githubusercontent.com/Bahari95/pyrefiga/main/r_adaptive_refinement/admesh_sphere.png" width="240" alt="mesh" />
    <img src="https://raw.githubusercontent.com/Bahari95/pyrefiga/main/r_adaptive_refinement/admesh_sol_sphere.png" width="240" alt="solution" />