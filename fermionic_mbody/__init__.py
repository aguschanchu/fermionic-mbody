"""
Top-level namespace for the fermionic_mbody package.
"""
from importlib import import_module
from types import ModuleType
from typing import Any

from .basis import FixedBasis                      
from .rho import (
    rho_m_gen,
    rho_2_block_gen,
    rho_2_kkbar_gen,
    rho_m,
    rho_m_direct
)

__all__ = [
    "FixedBasis",
    "rho_m_gen",
    "rho_2_block_gen",
    "rho_2_kkbar_gen",
    "rho_m",
    "rho_m_direct",
    "datasets",           
]

def __getattr__(name: str) -> Any:                         
    if name == "datasets":
        return import_module(f"{__name__}.datasets")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")