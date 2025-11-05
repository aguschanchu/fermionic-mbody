"""
Top-level namespace for the fermionic_mbody package.
"""
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np

from .basis import FixedBasis
from .rho import (
    rho_m_gen,
    rho_2_block_gen,
    rho_2_kkbar_gen,
    rho_m,
    rho_m_direct,
    rho_m_gen_legacy,
)

__all__ = [
    "FixedBasis",
    "rho_m_gen",
    "rho_2_block_gen",
    "rho_2_kkbar_gen",
    "rho_m",
    "rho_m_direct",
    "rho_m_gen_legacy"
]
