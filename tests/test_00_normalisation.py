"""
Trace and particle-number identities that hold for **any** N-particle state.
"""
import math
import numpy as np
import pytest

from fermionic_mbody import rho_m
from .helpers import dense, get_slater_determinant

def test_rho_m_trace(basis_4_2, rho1_tensor, rho2_tensor):
    N = basis_4_2.num

    # random pure state
    psi = random_state(basis_4_2, seed=42)

    # ρ₁
    rho1 = dense(rho_m(psi, rho1_tensor))
    assert np.isclose(np.trace(rho1), N)         # Tr ρ₁ = N

    # ρ₂
    rho2 = dense(rho_m(psi, rho2_tensor))
    assert np.isclose(np.trace(rho2), math.comb(N, 2))  # Tr ρ₂ = C(N,2)


def test_number_operator_identity(basis_4_2, rho1_tensor):
    N = basis_4_2.num
    psi = get_slater_determinant(basis_4_2, (0, 1))

    rho1 = dense(rho_m(psi, rho1_tensor))
    # Σ_i ρ₁[i,i]  equals N
    assert np.isclose(rho1.diagonal().sum(), N)
