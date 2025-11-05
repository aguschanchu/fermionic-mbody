"""
Pair-space sanity check for rho_2_kkbar_gen (pairs=True).

We verify that the DIAGONAL elements of ρ₂_kk̄ equal the pair occupancies
⟨n_k n_{k̄}⟩.
"""
import numpy as np
import pytest

from fermionic_mbody import FixedBasis, rho_m, rho_2_kkbar_gen
# Import updated helpers
from .helpers import dense, get_slater_determinant

def test_rho2_kkbar_diagonal_matches_occupancies():
    # Basis with pair compression ON (d=8, N=4 -> 4 pairs total, 2 occupied. Size=C(4,2)=6)
    basis = FixedBasis(d=8, num=4, pairs=True)

    # |Ψ⟩ = (|11110000⟩ + |00001111⟩) / √2
    # Pairs 0, 1 occupied (modes 0, 1, 2, 3)
    vec_a = get_slater_determinant(basis, (0, 1, 2, 3))
    # Pairs 2, 3 occupied (modes 4, 5, 6, 7)
    vec_b = get_slater_determinant(basis, (4, 5, 6, 7))
    psi = (vec_a + vec_b) / np.sqrt(2)

    # tensors
    rho2kk_t = rho_2_kkbar_gen(basis)

    rho2kk = dense(rho_m(psi, rho2kk_t))      # shape (4, 4)

    # Expected pair occupancies ⟨n_k n_{k̄}⟩.
    # Each pair (0-3) has 0.5 probability of being occupied in this state.
    expected = np.array([0.5, 0.5, 0.5, 0.5])

    diag = np.diag(rho2kk)

    assert np.allclose(diag, expected, atol=1e-8)

    # Matrix is Hermitian by construction
    assert np.allclose(rho2kk, rho2kk.conj().T, atol=1e-8)