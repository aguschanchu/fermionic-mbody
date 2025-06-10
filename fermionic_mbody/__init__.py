"""
Top-level namespace for the *fermionic_mbody* package.

We re-export the public API that users expect to import directly.
"""

from .basis import FixedBasis                       # noqa: F401
from .rho import (
    rho_m_gen,
    rho_2_block_gen,
    rho_2_kkbar_gen,
    rho_m,
)

__all__: list[str] = [
    "FixedBasis",
    "rho_m_gen",
    "rho_2_block_gen",
    "rho_2_kkbar_gen",
    "rho_m",
]
