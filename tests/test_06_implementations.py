"""
Tests comparing the different implementations (Direct vs Generator, Optimized vs Legacy) 
and testing contraction inputs (Density Matrix).
"""
import numpy as np
import pytest
import openfermion as of
import pytest

from fermionic_mbody import FixedBasis, rho_m, rho_m_gen, rho_m_direct, rho_m_gen_legacy
from .helpers import dense

# Parameterize tests over different (d, N, m) combinations
@pytest.mark.parametrize("d, N, m", [
    (4, 2, 1), (6, 3, 2), (8, 4, 2),
])
def test_rho_m_direct_vs_generator(d, N, m, random_state):
    """Compares rho_m_direct with the standard rho_m_gen + rho_m pipeline."""
    basis = FixedBasis(d=d, num=N)
    psi = random_state(basis, seed=101)

    # Direct calculation
    rdm_direct = rho_m_direct(basis, m=m, psi=psi)

    # Generator calculation
    tensor_gen = rho_m_gen(basis, m=m)
    rdm_gen = dense(rho_m(psi, tensor_gen))

    # Assert equality
    assert np.allclose(rdm_direct, rdm_gen, atol=1e-9)

@pytest.mark.parametrize("d, N, m", [(6, 3, 1), (6, 3, 2)])
def test_rho_m_optimized_vs_legacy(d, N, m):
    """Compares rho_m_gen (optimized) with rho_m_gen_legacy."""
    basis = FixedBasis(d=d, num=N)

    tensor_opt = rho_m_gen(basis, m=m)
    tensor_leg = rho_m_gen_legacy(basis, m=m)

    # Compare dense representations
    assert np.allclose(tensor_opt.todense(), tensor_leg.todense(), atol=1e-9)

def test_rho_m_contraction_density_matrix(basis_4_2, random_state):
    """
    Rigorously tests the rho_m contraction helper using a density matrix input.
    Verifies ⟨O⟩ = Tr[O_Nbody ρ] = Σᵢⱼ Oᵢⱼ ρ₁ᵢⱼ.
    """
    d, N = basis_4_2.d, basis_4_2.num
    
    # 1. Define a mixed state (Density Matrix ρ)
    psi1 = random_state(basis_4_2, seed=1, use_complex=True)
    psi2 = random_state(basis_4_2, seed=2, use_complex=True)
    # ρ = 0.7|ψ1⟩⟨ψ1| + 0.3|ψ2⟩⟨ψ2|. Use np.outer(ket, bra.conj())
    rho = 0.7 * np.outer(psi1, psi1.conj()) + 0.3 * np.outer(psi2, psi2.conj())

    # 2. Define an arbitrary 1-body Hermitian operator O = Σᵢⱼ Oᵢⱼ c†ⱼ cᵢ
    rng = np.random.default_rng(42)
    O_matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    O_matrix = O_matrix + O_matrix.conj().T
    
    O_op = of.FermionOperator()
    for i in range(d):
        for j in range(d):
            # Indexing O_ij c†_j c_i matches library convention RDM_ij = <c†_j c_i>
            O_op += O_matrix[i, j] * of.FermionOperator(((j, 1), (i, 0)))

    # 3. Calculate ⟨O⟩ explicitly: Tr[O_Nbody ρ]
    O_Nbody = basis_4_2.get_operator_matrix(O_op).toarray()
    expected_value = np.trace(O_Nbody @ rho)

    # 4. Calculate ⟨O⟩ using RDM: Σᵢⱼ Oᵢⱼ ρ₁ᵢⱼ
    T1 = rho_m_gen(basis_4_2, m=1)
    # Calculate ρ₁ = Tr[T1 ρ] (Relies on the fix in rho_m)
    rho1 = dense(rho_m(rho, T1))
    
    # CRITICAL FIX: Calculate the expectation value using element-wise product and sum.
    # value_from_rdm = np.trace(O_matrix @ rho1) # INCORRECT: Calculates Tr[O ρ₁ᵀ]
    value_from_rdm = np.sum(O_matrix * rho1)

    # Verify consistency
    assert np.isclose(value_from_rdm, expected_value)