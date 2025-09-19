"""
Projector-like properties that characterise Slater determinants.
"""
import numpy as np

from fermionic_mbody import rho_m
from .helpers import dense

def test_rho1_is_projector(basis_4_2, rho1_tensor):
    # Slater determinant |1100⟩
    psi = np.eye(basis_4_2.size)[3]

    rho1 = dense(rho_m(psi, rho1_tensor))
    assert np.allclose(rho1 @ rho1, rho1)


def test_rho2_eigenvalues_slater(basis_4_2, rho2_tensor):
    # same Slater determinant
    psi = np.eye(basis_4_2.size)[3]
    rho2 = dense(rho_m(psi, rho2_tensor))

    w = np.linalg.eigvalsh(rho2)
    # non-zero eigenvalues should all be ≈ 1
    non_zero = w[w > 1e-10]
    assert np.allclose(non_zero, 1.0, atol=1e-8)
