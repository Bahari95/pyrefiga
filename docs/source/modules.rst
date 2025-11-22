PyRefiga Package
================

Overview
--------
This document automatically documents all modules, classes, and functions
inside the `pyrefiga` package using Sphinx's autodoc and autosummary features.

Auto-generated modules index
----------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    pyrefiga
    pyrefiga.linalg.py
    pyrefiga.ad_mesh_tools.py
    pyrefiga.nurbs_utilities.py
    pyrefiga.nurbs_utilities_core.py
    pyrefiga.fast_diag_core.py
    pyrefiga.api.py
    pyrefiga.ad_mesh_core.py
    pyrefiga.fast_diag.py
    pyrefiga.results_f90_core.py
    pyrefiga.utilities.py
    pyrefiga.results_f90.py
    pyrefiga.quadratures.py
    pyrefiga.results.py
    pyrefiga.version.py
    pyrefiga.bsplines.py
    pyrefiga.nurbs_core.py
    pyrefiga.cad.py
    pyrefiga.spaces.py
    pyrefiga.gallery.py

Per-module documentation template
---------------------------------
Each module listed above will have its own page generated. Example for one module:

MODULE: pyrefiga.linalg
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pyrefiga.linalg
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: __all__, __author__, __version__

Practical Notes
---------------
- Ensure `conf.py` has these extensions enabled:

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
autosummary_generate = True