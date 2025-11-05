"""
Pytest fixtures & helpers shared by all test modules.
"""
from __future__ import annotations
import pytest

import numpy as np
import pytest

from fermionic_mbody import FixedBasis, rho_m_gen


@pytest.fixture
def random_state():
    """Factory → random normalised complex state for a given basis."""
    def _make(basis: FixedBasis, seed: int = 2025, use_complex=True):
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=basis.size)
        if use_complex:
            vec = vec + 1j * rng.normal(size=basis.size)
        else:
            vec = vec + 0j
            
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    return _make


# --------------------------- fixtures ----------------------------------------
@pytest.fixture(scope="module")
def basis_4_2() -> FixedBasis:
    """d = 4, N = 2  without pair-compression (dim = 6)."""
    return FixedBasis(d=4, num=2, pairs=False)

@pytest.fixture(scope="module")
def rho1_tensor(basis_4_2):
    """ρ(1) generator tensor for the (4, 2) basis."""
    return rho_m_gen(basis_4_2, m=1)

@pytest.fixture(scope="module")
def rho2_tensor(basis_4_2):
    """ρ(2) generator tensor for the (4, 2) basis."""
    return rho_m_gen(basis_4_2, m=2)