"""
Spectral “twin” identity  λ(ρ_M) = λ(ρ_{N-M})  for pure states.
Here we take N = 2, so M = N-M = 1  (trivial), but the test shows
the API works for different d.
"""
import numpy as np
import pytest

from fermionic_mbody import FixedBasis, rho_m, rho_m_gen
from .helpers import dense

def test_spectral_twin_identity(random_state):
    basis = FixedBasis(d=6, num=3)      # N = 3
    rho1_t = rho_m_gen(basis, m=1)
    rho2_t = rho_m_gen(basis, m=2)

    # random pure state
    vec = random_state(basis, seed=99)

    rho1 = dense(rho_m(vec, rho1_t))
    rho2 = dense(rho_m(vec, rho2_t))

    # eigenvalues (sorted)
    w1 = np.sort(np.linalg.eigvalsh(rho1))
    w2 = np.sort(np.linalg.eigvalsh(rho2))

    tol = 1e-10
    nz1 = w1[w1 > tol]          # six non-zero eigenvalues
    nz2 = w2[w2 > tol]          # same six in ρ₂

    assert nz1.size == nz2.size == 6
    assert np.allclose(nz1, nz2, atol=1e-8)
