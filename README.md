# pyrefiga: python + refinement + Isogeometric analysis

**pyrefiga** is a CAD and simulation library tailored for Isogeometric Analysis (IGA). It offers advanced capabilities, including r-adaptive mesh refinement, enabling accurate and efficient computations on complex geometries.

## License

This project is licensed under the GNU GPL v3.0, as it includes and modifies code from the original [simplines](https://github.com/ratnania/simplines) library.

© 2025 Mustapha Bahari. See the LICENSE file for full licensing details.



**Modifications in this version were made, including:**

    Support for NURBS basis functions in CAD and IGA

    Fast solvers

    Examples on Poisson, Cahn-Hilliard, and Elasticity problems

    It supports mappings defined by composition.

    Adaptation to complex 2D and 3D geometries

    Enhancements for multipatch domains (2D)

    Improvements to solver stability and adaptive mesh refinement

    Extended support for visualization (matplotlib and paraview) and performance diagnostics
    
## Clone the Repository

```bash
git clone https://github.com/Bahari95/pyrefiga.git
cd pyrefiga
```

## Install

Python 3.10 or newer is required. The core library uses NumPy and SciPy and
runs with Python kernels; plotting packages and a compiler are optional.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e '.[dev]'
```

Install optional features as needed:

```bash
python -m pip install '.[visualization]'  # Matplotlib plots and PyVista/VTK export
python -m pip install '.[accelerate]'     # Pyccel acceleration
```

`compile_kernel(core, arity=...)` uses Pyccel when it is installed and falls
back to Python when it is absent. Pass `pyccel=False` to always use Python,
or `pyccel=True` to require Pyccel. Compilation failures are reported rather
than silently switching backends. Acceleration also requires a compiler
supported by Pyccel, such as gfortran for its default Fortran backend.

To compile the bundled kernels from a source checkout:

```bash
python setup.py run_pyccel
```

The XML geometries are included in installed packages and can be located with
`pyrefiga.load_xml('elasticity.xml')` from any working directory.

## Tests

Run the numerical regression suite from the repository root:

```bash
python -m unittest discover -s pyrefiga/tests -p 'test_*.py' -v
```

The PyVista export test is skipped when visualization support is absent.
Install `.[visualization,accelerate]` to run the full suite with acceleration
and export support. CI runs both the core suite and the optional features,
and checks a wheel built from the source distribution outside the checkout.

To reproduce the wheel installation check in a separate environment:

```bash
python -m build
python3 -m venv /tmp/pyrefiga-wheel-check
/tmp/pyrefiga-wheel-check/bin/python -m pip install dist/*.whl
/tmp/pyrefiga-wheel-check/bin/python -I tests/test_installed_package.py
```

## Example

The [periodic curl-curl example](docs/examples/curlcurl2d_example.py) uses
compatible IGA H(curl) spaces and leapfrog time stepping. See
[the solver documentation](docs/curlcurl.md) for periodic boundary conditions,
source terms and ParaView output.

The [generalized-alpha Cahn–Hilliard example](docs/examples/cahn_Hilliard2d_Galpha_example.py)
adds adaptive time stepping and ParaView export. See
[its documentation](docs/cahn_hilliard_galpha.md) for time-step tolerances and controls.

```bash
PYTHONPATH=. python docs/examples/curlcurl2d_example.py --nelements 8 --steps 20 --export
```

![PNG](https://github.com/Bahari95/pyrefiga/blob/main/r_adaptive_refinement/adaptive_meshes.png)
