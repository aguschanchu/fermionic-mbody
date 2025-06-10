"""
Spectral “twin” identity  λ(ρ_M) = λ(ρ_{N-M})  for pure states.
Here we take N = 2, so M = N-M = 1  (trivial), but the test shows
the API works for different d.
"""
import numpy as np

from fermibasis import FixedBasis, rho_m, rho_m_gen


def test_spectral_twin_identity():
    basis = FixedBasis(d=6, num=3)      # N = 3
    rho1_t = rho_m_gen(basis, m=1)
    rho2_t = rho_m_gen(basis, m=2)

    # random pure state
    vec = np.random.default_rng(99).normal(size=basis.size) + 0j
    vec /= np.linalg.norm(vec)

    rho1 = rho_m(vec, rho1_t)
    rho2 = rho_m(vec, rho2_t)

    # eigenvalues sorted
    w1 = np.sort(np.linalg.eigvalsh(rho1))
    w2 = np.sort(np.linalg.eigvalsh(rho2))

    # λ(ρ₁)  ==  λ(ρ₂)   for N = 3 pure state
    assert np.allclose(w1, w2, atol=1e-8)
