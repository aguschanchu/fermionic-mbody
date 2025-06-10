"""
Pair-space sanity check for rho_2_kkbar_gen (pairs=True).

For d = 4 (two time-reversed pairs) we prepare the equal superposition

    |Ψ⟩ = (|11110000⟩ + |00001111⟩) / √2

and verify that the **row-sums** of ρ₂_kk̄ equal the pair occupancies
⟨n_k n_{k̄}⟩ = diag ρ₁[even indices].  This identity holds for every
state because Σ_j c†_j c†_{j̄} is a completeness relation in the paired
subspace.
"""
import numpy as np

from fermionic_mbody import FixedBasis, rho_m, rho_m_gen, rho_2_kkbar_gen
from tests.conftest import dense, slater_state


def test_rho2_kkbar_row_sum_matches_rho1_diag():
    # Basis with pair compression ON
    basis = FixedBasis(d=8, num=4, pairs=True)  # size = 2

    # |Ψ⟩  =  (|1100⟩ + |0011⟩) / √2   in *paired* space
    vec_a = slater_state(basis, (0, 1, 2, 3))  # pair-0 occupied
    vec_b = slater_state(basis, (4, 5, 6, 7))  # pair-1 occupied
    psi = (vec_a + vec_b) / np.sqrt(2)

    # tensors
    rho1_t   = rho_m_gen(basis, m=1)
    rho2kk_t = rho_2_kkbar_gen(basis)

    rho1   = dense(rho_m(psi, rho1_t))        # shape (4, 4)
    rho2kk = dense(rho_m(psi, rho2kk_t))      # shape (2, 2)

    # Expected pair occupancies  ⟨n_k n_{k̄}⟩  (even indices 0,2)
    expected = rho1[::2, ::2].diagonal()      # array([0.5, 0.5])

    # Row sums must match expected occupancies
    row_sum = rho2kk.sum(axis=1)
    col_sum = rho2kk.sum(axis=0)

    assert np.allclose(row_sum, expected, atol=1e-8)
    assert np.allclose(col_sum, expected, atol=1e-8)

    # Matrix is Hermitian by construction
    assert np.allclose(rho2kk, rho2kk.conj().T, atol=1e-8)
