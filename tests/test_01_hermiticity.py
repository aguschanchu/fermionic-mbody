"""
Every reduced density matrix must be Hermitian.
"""
import numpy as np
import pytest

from fermionic_mbody import rho_m
from .helpers import dense


def test_rho1_hermitian(basis_4_2, rho1_tensor):
    # Use complex state to rigorously test R = R†
    psi = random_state(basis_4_2, seed=1, use_complex=True)

    rho1 = dense(rho_m(psi, rho1_tensor))
    assert np.allclose(rho1, rho1.conj().T)


def test_rho2_hermitian(basis_4_2, rho2_tensor):
    psi = random_state(basis_4_2, seed=1, use_complex=True)

    rho2 = dense(rho_m(psi, rho2_tensor))
    assert np.allclose(rho2, rho2.conj().T)
