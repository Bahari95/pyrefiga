"""Load feature dependencies without making them core import requirements."""

from importlib import import_module


def import_optional(module, extra, *, required=True):
    """Load a dependency, preserving errors caused by a broken installation."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split('.')[0]:
            raise
        if not required:
            return None
        raise ImportError(
            f"{module} is required for this feature. "
            f"Install it with: python -m pip install 'pyrefiga[{extra}]'"
        ) from exc
