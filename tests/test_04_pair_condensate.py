"""
Uniform pair-condensate identities (m = 1, d = 4 ⇒ n = 2 pairs).
"""
import math
import numpy as np

from fermionic_mbody import rho_m
from .helpers import dense


def test_pair_condensate_occupancies(basis_4_2, rho1_tensor, random_state):
    n_pairs = basis_4_2.d // 2
    m_pairs = 1
    expected_occupancy = m_pairs / n_pairs     # = 0.5

    from tests.conftest import pair_condensate_state

    psi = pair_condensate_state(basis_4_2, m_pairs)
    rho1 = dense(rho_m(psi, rho1_tensor))

    diag = rho1.diagonal()
    assert np.allclose(diag, expected_occupancy, atol=1e-8)


def test_pair_condensate_lambda_max(basis_4_2, rho2_tensor):
    from tests.conftest import pair_condensate_state

    m_pairs = 1
    n_pairs = basis_4_2.d // 2
    expected = m_pairs * (1 - (m_pairs - 1) / n_pairs)   # Eq. (69) ⇒ 1.0

    psi = pair_condensate_state(basis_4_2, m_pairs)
    rho2 = dense(rho_m(psi, rho2_tensor))

    lam_max = np.max(np.linalg.eigvalsh(rho2))
    assert np.isclose(lam_max, expected, atol=1e-8)
