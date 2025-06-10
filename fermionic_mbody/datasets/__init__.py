# fermionic_mbody/datasets/__init__.py
from importlib import import_module
from typing import Any

__all__ = ["h2o"]           # list every dataset module here

def __getattr__(name: str) -> Any:      # pragma: no cover
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
