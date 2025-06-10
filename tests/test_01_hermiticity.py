"""
Every reduced density matrix must be Hermitian.
"""
import numpy as np

from fermionic_mbody import rho_m
from tests.conftest import dense

def test_rho1_hermitian(basis_4_2, rho1_tensor):
    psi = np.random.default_rng(1).normal(size=basis_4_2.size) + 0j
    psi /= np.linalg.norm(psi)

    rho1 = dense(rho_m(psi, rho1_tensor))
    assert np.allclose(rho1, rho1.conj().T)


def test_rho2_hermitian(basis_4_2, rho2_tensor):
    psi = np.random.default_rng(7).normal(size=basis_4_2.size) + 0j
    psi /= np.linalg.norm(psi)

    rho2 = dense(rho_m(psi, rho2_tensor))
    assert np.allclose(rho2, rho2.conj().T)
