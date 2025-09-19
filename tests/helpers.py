"""
Pytest fixtures & helpers shared by all test modules.
"""
from __future__ import annotations

import math
from typing import Generator, Tuple

import numpy as np
import pytest

from fermionic_mbody import FixedBasis, rho_m, rho_m_gen


def slater_state(basis: FixedBasis, occupied: Tuple[int, ...]) -> np.ndarray:
    """
    Build a Slater determinant |n₀ n₁ …⟩ in the *canonical* Fock basis.

    `occupied` lists the single-particle indices that contain one
    particle each.
    """
    bitmask = sum(1 << i for i in occupied)
    idx = int(np.where(basis.num_ele == bitmask)[0])
    vec = np.zeros(basis.size, complex)
    vec[idx] = 1.0
    return vec


def pair_condensate_state(basis: FixedBasis, m_pairs: int) -> np.ndarray:
    """
    Uniform pair condensate

        |Ψ_m⟩ ∝ Σ_{|S|=m} ∏_{k∈S} c†_k c†_{k̄} |0⟩

    for *small* `d` (sufficient for unit-tests).
    """
    d = basis.d
    assert d % 2 == 0, "d must be even (k / k̄ pairs)"
    n_pairs = d // 2
    assert 0 < m_pairs <= n_pairs

    vec = np.zeros(basis.size, complex)

    # iterate over every subset of m time-reversed pairs
    from itertools import combinations

    for combo in combinations(range(n_pairs), m_pairs):
        bitmask = 0
        for k in combo:
            bitmask |= 1 << (2 * k)      # c†_k
            bitmask |= 1 << (2 * k + 1)  # c†_{k̄}
        idx = int(np.where(basis.num_ele == bitmask)[0])
        vec[idx] = 1.0

    vec /= np.linalg.norm(vec)
    return vec


# --------------------------- helpers -----------------------------------------
def dense(arr):
    """Return a NumPy ndarray from either dense or sparse input."""
    return np.asarray(arr.todense()) if hasattr(arr, "todense") else np.asarray(arr)

