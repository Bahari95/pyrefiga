# pyrefiga

**pyrefiga** is a simulation and r-adaptive mesh refinement library based on Isogeometric Analysis (IGA).  


**Modifications in this version were made by Mustapha Bahari (2025), including:**

    Support for NURBS basis functions in CAD and IGA

    Fast solvers

    Examples on Poisson, Cahn-Hilliard, and Elasticity problems ...

    r-refinement (coming soon)

    Adaptation to complex 2D and 3D geometries

    Enhancements for multipatch domains (coming soon)

    Improvements to solver stability and adaptive mesh refinement

    Extended support for visualization and performance diagnostics

## License

This project is licensed under the [GNU GPL v3.0](LICENSE), as it includes and modifies code from the original [simplines](https://github.com/ratnania/simplines) library.

© 2025 Mustapha Bahari. See the LICENSE file for more details.

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
