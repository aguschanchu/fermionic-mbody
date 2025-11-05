"""
Pytest helpers shared by all test modules.
"""
from __future__ import annotations

from typing import Tuple
from itertools import combinations # Import combinations
import pytest

import numpy as np

from fermionic_mbody import FixedBasis

def get_slater_determinant(basis: FixedBasis, occupied: Tuple[int, ...]) -> np.ndarray:
    """
    Build a Slater determinant |n₀ n₁ …⟩ in the *canonical* Fock basis.
    """
    bitmask = sum(1 << i for i in occupied)
    
    idx_arr = np.where(basis.bitmasks == np.int64(bitmask))[0]
    
    if idx_arr.size == 0:
        raise ValueError(f"Configuration {occupied} (mask {bitmask}) not found in basis.")

    idx = int(idx_arr[0])
    vec = np.zeros(basis.size, complex)
    vec[idx] = 1.0
    return vec

def get_pair_condensate(basis: FixedBasis, m_pairs: int) -> np.ndarray:
    """
    Uniform pair condensate |Ψ_m⟩ ∝ Σ_{|S|=m} ∏_{k∈S} c†_k c†_{k̄} |0⟩
    """
    d = basis.d
    assert d % 2 == 0, "d must be even (k / k̄ pairs)"
    n_pairs = d // 2
    assert 0 < m_pairs <= n_pairs

    vec = np.zeros(basis.size, complex)

    # iterate over every subset of m time-reversed pairs
    for combo in combinations(range(n_pairs), m_pairs):
        bitmask = 0
        for k in combo:
            bitmask |= 1 << (2 * k)      # c†_k
            bitmask |= 1 << (2 * k + 1)  # c†_{k̄}
        
        idx_arr = np.where(basis.bitmasks == np.int64(bitmask))[0]
        
        if idx_arr.size > 0:
            idx = int(idx_arr[0])
            vec[idx] = 1.0

    norm = np.linalg.norm(vec)
    if norm == 0:
         raise ValueError("Pair condensate state has zero norm in the given basis.")
    return vec / norm


# --------------------------- helpers -----------------------------------------
def dense(arr):
    """Return a NumPy ndarray from either dense or sparse input."""
    return np.asarray(arr.todense()) if hasattr(arr, "todense") else np.asarray(arr)
