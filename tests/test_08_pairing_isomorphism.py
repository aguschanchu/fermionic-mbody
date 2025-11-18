"""
Test suite for the optimization of RDM calculations when pairs=True.

This optimization leverages the mathematical isomorphism between:
1. A paired fermionic system (D orbitals, N=2P electrons, pairs=True) calculating an m-pair RDM.
2. A reduced standard fermionic system (D/2 orbitals, P electrons, pairs=False) calculating an m-fermion RDM.

The optimization ensures that calls for (1) are internally redirected to perform the calculation for (2). 
These tests verify that the results are identical.
"""

import numpy as np
import pytest
from fermionic_mbody import FixedBasis, rho_m_gen, rho_m_direct, rho_m, rho_2_kkbar_gen

# Import the dense helper from the test suite's helpers module
try:
    from .helpers import dense, random_state, get_pair_condensate
except ImportError:
    # Fallback definition if helpers cannot be imported (e.g., if running the file in isolation)
    def dense(arr):
        return np.asarray(arr.todense()) if hasattr(arr, "todense") else np.asarray(arr)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Define the parameters for the isomorphic systems
# System 1: Paired (Optimization Target)
D_PAIRED = 8
N_PAIRED = 4  # P=2 pairs
# System 2: Reduced (Standard Calculation)
D_REDUCED = D_PAIRED // 2
N_REDUCED = N_PAIRED // 2

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def basis_paired():
    """Basis for the paired system (D=8, N=4, pairs=True)."""
    return FixedBasis(d=D_PAIRED, num=N_PAIRED, pairs=True)

@pytest.fixture(scope="module")
def basis_reduced():
    """Basis for the isomorphic reduced system (D=4, N=2, pairs=False)."""
    return FixedBasis(d=D_REDUCED, num=N_REDUCED, pairs=False)

@pytest.fixture
def isomorphic_state(basis_paired, basis_reduced, random_state):
    """A random state vector, valid for both isomorphic bases."""
    # Ensure the fundamental assumption holds before generating the state
    if basis_paired.size != basis_reduced.size:
        pytest.fail(f"Isomorphism prerequisite failed: Basis sizes differ ({basis_paired.size} != {basis_reduced.size})")
    
    # Generate the state. The 'random_state' fixture is defined in conftest.py
    return random_state(basis_paired, seed=20251118, use_complex=True)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_basis_isomorphism_properties(basis_paired, basis_reduced):
    """Verify that the basis definitions meet the isomorphism criteria."""
    assert basis_paired.d == D_PAIRED
    assert basis_paired.num == N_PAIRED
    assert basis_paired.pairs is True
    
    assert basis_reduced.d == D_REDUCED
    assert basis_reduced.num == N_REDUCED
    assert basis_reduced.pairs is False
    
    # The core requirement for the optimization
    assert basis_paired.size > 0
    assert basis_paired.size == basis_reduced.size

@pytest.mark.parametrize("m", [0, 1, 2])
def test_rho_m_direct_isomorphism(basis_paired, basis_reduced, isomorphic_state, m):
    """Test that rho_m_direct yields identical results for paired (optimized) 
    and reduced (explicitly bosonic) systems."""
    psi = isomorphic_state
    
    # Calculate using the optimized path (pairs=True). Internally uses 'bosonic'.
    rdm_paired = rho_m_direct(basis_paired, m=m, psi=psi, n_workers=1)
    
    # Calculate the correct reference: Explicitly use bosonic statistics on the reduced basis.
    rdm_reduced_bosonic = rho_m_direct(basis_reduced, m=m, psi=psi, n_workers=1, _statistics='bosonic')
    
    # The results must be numerically identical
    np.testing.assert_allclose(rdm_paired, rdm_reduced_bosonic, atol=1e-12)
    assert rdm_paired.shape == rdm_reduced_bosonic.shape

@pytest.mark.parametrize("m", [0, 1, 2])
def test_rho_m_gen_isomorphism(basis_paired, basis_reduced, isomorphic_state, m):
    """Test that rho_m_gen yields identical tensors for paired (optimized) 
    and reduced (explicitly bosonic) systems."""
    
    # Generate tensor using the optimized path (pairs=True). Internally 'bosonic'.
    tensor_paired = rho_m_gen(basis_paired, m=m, n_workers=1)
    
    # Generate the correct reference tensor: Explicitly use bosonic statistics.
    tensor_reduced_bosonic = rho_m_gen(basis_reduced, m=m, n_workers=1, _statistics='bosonic')
    
    # The tensors must have the same shape and data
    assert tensor_paired.shape == tensor_reduced_bosonic.shape
    np.testing.assert_allclose(dense(tensor_paired), dense(tensor_reduced_bosonic), atol=1e-12)

    # Additionally, verify the contraction yields the same RDM
    psi = isomorphic_state
    rdm_from_paired = rho_m(psi, tensor_paired)
    rdm_from_reduced = rho_m(psi, tensor_reduced_bosonic)
    
    np.testing.assert_allclose(dense(rdm_from_paired), dense(rdm_from_reduced), atol=1e-12)

@pytest.mark.parametrize("m", [0, 1, 2])
def test_cross_check_direct_vs_gen(basis_paired, isomorphic_state, m):
    """Cross-check consistency between direct and generator methods within the optimized paired basis."""
    
    psi = isomorphic_state

    # Direct calculation
    rdm_direct = rho_m_direct(basis_paired, m, psi, n_workers=1)

    # Generator calculation and contraction
    tensor_gen = rho_m_gen(basis_paired, m, n_workers=1)
    rdm_gen = dense(rho_m(psi, tensor_gen))

    # Assert consistency (ensures both optimized paths work correctly and match)
    np.testing.assert_allclose(rdm_direct, rdm_gen, rtol=1e-9, atol=1e-9)

def test_pair_condensate_analytical_rdm(basis_paired):
    """Verify the 1-pair RDM for a uniform pair condensate state against analytical results."""
    
    # System parameters (M=D/2 levels, P=N/2 pairs)
    M = D_REDUCED # Number of levels (M=4)
    P = N_REDUCED # Number of pairs (P=2)
    
    # 1. Generate the state
    try:
        psi_condensate = get_pair_condensate(basis_paired, P)
    except (ValueError, NameError) as e:
        pytest.skip(f"Skipping test: get_pair_condensate helper not available or failed: {e}")

    # 2. Calculate the 1-pair RDM (m=1) using the optimized function
    rdm1 = rho_m_direct(basis_paired, m=1, psi=psi_condensate, n_workers=1)
    
    # 3. Define the expected analytical RDM (derived using bosonic statistics)
    # Diagonal: <N_k> = P/M
    diag_val = P / M
    
    # Off-diagonal: <C†_j C_k> = P(M-P) / (M(M-1))
    if M > 1 and P >= 1:
        off_diag_val = (P * (M - P)) / (M * (M - 1))
    else:
        off_diag_val = 0

    expected_rdm = np.full((M, M), off_diag_val, dtype=float)
    np.fill_diagonal(expected_rdm, diag_val)
    
    # 4. Assert correctness
    np.testing.assert_allclose(rdm1.real, expected_rdm, atol=1e-12)
    
    # Ensure imaginary part is zero (as the state is real)
    if np.iscomplexobj(rdm1):
        np.testing.assert_allclose(rdm1.imag, np.zeros_like(expected_rdm), atol=1e-12)

    # 5. Check normalization Tr(ρ(1)) = P
    np.testing.assert_allclose(np.trace(rdm1).real, P, atol=1e-12)

def test_physical_cross_validation(basis_paired, isomorphic_state):
    """
    Validate the optimized calculation (pairs=True) against a standard fermionic 
    calculation (pairs=False) by mapping the state and comparing the physical RDM blocks.
    """
    # Use the random state provided by the fixture
    psi_paired = isomorphic_state

    # 1. Calculate RDM using the optimized path (System 1)
    # We test the 1-pair RDM (m=1).
    m = 1
    rdm_optimized = rho_m_direct(basis_paired, m=m, psi=psi_paired, n_workers=1)

    # 2. Define the standard fermionic basis (System 2, without optimization)
    basis_standard = FixedBasis(d=D_PAIRED, num=N_PAIRED, pairs=False)
    
    # 3. Map the state vector from the paired basis to the standard basis.
    psi_standard = np.zeros(basis_standard.size, dtype=psi_paired.dtype)
    
    # We rely on FixedBasis(pairs=False) sorting bitmasks in ascending order.
    # We can efficiently find the indices using np.searchsorted.
    standard_indices = np.searchsorted(basis_standard.bitmasks, basis_paired.bitmasks)
    
    # Verify the mapping is correct (masks should match)
    if not np.array_equal(basis_standard.bitmasks[standard_indices], basis_paired.bitmasks):
        pytest.fail("Bitmask mismatch during state mapping.")

    psi_standard[standard_indices] = psi_paired

    # 4. Calculate RDM using the standard fermionic path
    # The 1-pair RDM corresponds physically to the rho_2_kkbar block in the standard system.
    # rho_2_kkbar_gen correctly handles the standard fermionic case when pairs=False.
    tensor_kkbar_standard = rho_2_kkbar_gen(basis_standard, n_workers=1)
    rdm_standard = dense(rho_m(psi_standard, tensor_kkbar_standard))

    # 5. Compare the results
    # The optimized 1-RDM must match the standard rho_2_kkbar.
    np.testing.assert_allclose(rdm_optimized, rdm_standard, atol=1e-12)