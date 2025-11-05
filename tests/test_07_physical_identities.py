"""
Test fundamental physical constraints and theoretical identities.
"""
import numpy as np
import pytest
from fermionic_mbody import FixedBasis, rho_m_direct

@pytest.mark.parametrize("d, N", [(6, 3), (8, 4)])
def test_pauli_exclusion_principle(d, N, random_state):
    """Eigenvalues of the 1-RDM must be in [0, 1] for fermions."""
    basis = FixedBasis(d=d, num=N)
    psi = random_state(basis)

    rdm1 = rho_m_direct(basis, m=1, psi=psi)
    
    # Calculate eigenvalues (use eigvalsh for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(rdm1)

    # Check bounds [0, 1] (within numerical tolerance)
    assert np.all(eigenvalues >= -1e-10)
    assert np.all(eigenvalues <= 1.0 + 1e-10)

def test_contraction_identity_rho2_to_rho1(random_state):
    """
    Verify the theoretical contraction identity (Eq. 29 in Ref [1]):
    ρ⁽¹⁾ = Tr₂[ρ⁽²⁾] / (N-1).
    """
    D, N = 6, 3
    basis_N = FixedBasis(d=D, num=N)
    psi = random_state(basis_N, seed=202)

    # Calculate ρ⁽¹⁾ and ρ⁽²⁾
    RDM1 = rho_m_direct(basis_N, m=1, psi=psi)
    RDM2 = rho_m_direct(basis_N, m=2, psi=psi)

    # Define the M=1 and M=2 bases for index mapping
    m1_basis = FixedBasis(D, num=1)
    m2_basis = FixedBasis(D, num=2)
    D1 = m1_basis.size

    # Perform the trace contraction Tr₂[ρ⁽²⁾] manually
    RDM1_contracted = np.zeros((D1, D1), dtype=RDM2.dtype)

    # Iterate over 1-body indices i, j and the contraction index k
    for i_idx in range(D1):
        for j_idx in range(D1):
            # Convert indices to standard Python int for robust access
            mask_i = int(m1_basis.bitmasks[i_idx])
            mask_j = int(m1_basis.bitmasks[j_idx])
            
            for k_idx in range(D1):
                mask_k = int(m1_basis.bitmasks[k_idx])
                
                # Pauli principle check (i!=k and j!=k)
                if (mask_i & mask_k) or (mask_j & mask_k):
                    continue

                # Form 2-body masks I=(i,k) and J=(j,k)
                mask_I = mask_i | mask_k
                mask_J = mask_j | mask_k
                
                # Find indices in the 2-body basis (using internal cache)
                # Ensure keys are standard Python int
                I_idx = m2_basis._mask2idx_cache.get(mask_I)
                J_idx = m2_basis._mask2idx_cache.get(mask_J)
                
                # The signs are implicitly handled by the FixedBasis ordering 
                # consistency between M=1 and M=2 bases.
                # Contribution: RDM1[i, j] += RDM2[I, J]
                s_i = +1 if i_idx < k_idx else -1
                s_j = +1 if j_idx < k_idx else -1
                RDM1_contracted[i_idx, j_idx] += (s_i * s_j) * RDM2[I_idx, J_idx]

    # Apply the normalization factor (N-1)
    RDM1_contracted /= (N - 1)

    # Assert equality
    assert np.allclose(RDM1, RDM1_contracted, atol=1e-9)