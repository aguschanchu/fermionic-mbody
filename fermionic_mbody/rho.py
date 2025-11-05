"""
Generators for **m-body reduced density-matrix tensors** ρ(m), specialised
pairing-blocks, and a convenience contraction helper :func:`rho_m`.

All heavy lifting is off-loaded to
:pyfunc:`fermibasis._parallel.chunked`, which wraps a
:multiprocessing:`multiprocessing.Pool`.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
import openfermion as of
import sparse
from multiprocessing import cpu_count
from math import comb
from functools import wraps
import numba
from numba import njit

from ._parallel import chunked
from .basis import FixedBasis
from ._ofsparse import number_preserving_matrix, restrict_sector_matrix, mask_to_index_map


__all__ = [
    "rho_m_gen",
    "rho_m_gen_legacy",   
    "rho_2_block_gen",
    "antisymmetrise_block",
    "rho_2_kkbar_gen",
    "rho_m",
]

USE_NUMBA = True

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _ensure_workers(n_workers: int | None) -> int:
    """Return a strictly positive worker count (defaults to all CPUs)."""
    return max(1, n_workers or cpu_count())


def _restrict_last2(tensor, subset):
    """
    Keep rows/cols whose *full-sector* positions are listed in `subset`
    (1-D int array).  Works for any tensor whose last two axes are the
    N-body basis.

    Implementation detail: slice **one axis at a time** so we never pass
    more than one advanced index to sparse.COO.
    """
    subset = np.asarray(subset)

    # 1) cut the **last** axis: …, full_dim  ->  …, k
    tensor = tensor[(Ellipsis, subset)]             # ⇢ shape (..., full, k)

    # 2) cut what is now the **penultimate** axis (was full_dim)
    #    build an index tuple like  (..., subset, :)
    nd  = tensor.ndim
    idx = [slice(None)] * (nd - 2) + [subset, slice(None)]
    return tensor[tuple(idx)]                       # ⇢ shape (..., k, k)
    

def subspace_aware(gen_fn):
    """Make a ρ-generator work with restricted FixedBasis objects."""
    @wraps(gen_fn)
    def wrapper(basis, *args, **kw):
        full_dim = comb(basis.d, basis.num)
        if basis.size == full_dim:                 # full sector → original path
            return gen_fn(basis, *args, **kw)

        # --- expand to full sector -------------------------------------------------
        from .basis import FixedBasis             # lazy import to avoid cycles
        tensor_full = gen_fn(FixedBasis(basis.d, basis.num), *args, **kw)

        # --- slice it back ---------------------------------------------------------
        mapping = mask_to_index_map(basis.d, basis.num)
        subset  = np.fromiter((mapping[m] for m in basis.num_ele), int)
        return _restrict_last2(tensor_full, subset)
    return wrapper



# =====================================================================
# OPTIMIZED RHO_M_GEN IMPLEMENTATION (Numba acceleration)
# =====================================================================

# ---------------------------------------------------------------------
# Numba Optimized Helper Functions
# ---------------------------------------------------------------------

@njit(cache=True)
def popcount(n):
    """Fast population count (number of set bits) using Numba."""
    count = 0
    n = int(n) # Ensure integer type
    while n > 0:
        n &= (n - 1) 
        count += 1
    return count

@njit(cache=True)
def calculate_fermionic_sign_numba(mask_c, indices_i: np.ndarray):
    """
    Calculates the fermionic sign (Jordan-Wigner, LE convention) using Numba.
    indices_i must be a sorted numpy array.
    """
    sign = 1
    current_mask = mask_c
    
    # Apply operators sequentially (lowest index first)
    for idx in indices_i:
        # 1. Calculate sign based on parity of electrons before idx
        mask_before = (1 << idx) - 1
        
        # Use optimized popcount
        count = popcount(current_mask & mask_before)
            
        if (count & 1) == 1: # Check parity (if odd)
            sign = -sign
            
        # 2. Update the mask (annihilate electron at idx)
        current_mask ^= (1 << idx)
        
    return sign

@njit(cache=True)
def compute_rdm_contributions_numba(ii_arr, c_idx_arr, sign_arr, coord_dtype_np):
    """
    Numba-accelerated core loop using pre-allocation.
    """
    num_conn = len(ii_arr)
    
    if num_conn == 0:
        return np.empty((0, 4), dtype=coord_dtype_np), np.empty(0, dtype=np.float64)

    # 1. Calculate required size and pre-allocate.
    # The total number of entries generated when covering jj >= ii and adding symmetry (jj, ii)
    # is exactly num_conn * num_conn.
    total_size = num_conn * num_conn
    
    indices_np = np.empty((total_size, 4), dtype=coord_dtype_np)
    values_np = np.empty(total_size, dtype=np.float64) # Values are always float

    k = 0 # Counter for the current position in the output arrays

    # 2. Fill the arrays efficiently
    for i in range(num_conn):
        ii = ii_arr[i]
        c_idx_i = c_idx_arr[i]
        sign_i = sign_arr[i]
        
        # Iterate starting from i to only cover jj >= ii
        for j in range(i, num_conn):
            jj = ii_arr[j]
            c_idx_j = c_idx_arr[j]
            sign_j = sign_arr[j]

            value = float(sign_i * sign_j)
            
            # Store (ii, jj) part: [ii, jj, c_idx_j, c_idx_i]
            indices_np[k, 0] = ii
            indices_np[k, 1] = jj
            indices_np[k, 2] = c_idx_j
            indices_np[k, 3] = c_idx_i
            values_np[k] = value
            k += 1

            # Add the symmetric part (jj, ii) if i != j
            if i != j:
                # Store (jj, ii) part: [jj, ii, c_idx_i, c_idx_j]
                indices_np[k, 0] = jj
                indices_np[k, 1] = ii
                indices_np[k, 2] = c_idx_i
                indices_np[k, 3] = c_idx_j
                values_np[k] = value # Value is real
                k += 1

    # Return the filled arrays.
    return indices_np, values_np

# ---------------------------------------------------------------------
# Python Helper Functions (Used if Numba is disabled or for setup)
# ---------------------------------------------------------------------

def calculate_fermionic_sign_python(mask_c: int, indices_i: Sequence[int]) -> int:
    """Pure Python fermionic sign calculation (Fallback)."""
    sign = 1
    current_mask = mask_c
    for idx in indices_i:
        mask_before = (1 << idx) - 1
        # Use optimized population count if available (Python 3.10+)
        if hasattr(int, 'bit_count'):
            count = (current_mask & mask_before).bit_count()
        else:
            count = bin(current_mask & mask_before).count('1')

        if (count & 1) == 1:
            sign = -sign
        current_mask ^= (1 << idx)
    return sign

def calculate_connections_v3(basis_N: FixedBasis, m_basis: FixedBasis, d: int, basis_N_indices: Dict[int, List[int]]):
    """
    Calculates the connection map using optimized subset iteration.
    Uses Numba signs if available.
    """
    N = basis_N.num
    if N is None: return {}, {}, []
    m = m_basis.num
    mask2idx_m = m_basis._mask2idx

    # Pass 1: Identify Nm_masks (Python loop, optimized by subset iteration)
    Nm_masks = set()
    for mask_c in basis_N.num_ele:
        indices_c = basis_N_indices[mask_c]
        # Iterate over all combinations of size m (subsets)
        for indices_i in itertools.combinations(indices_c, m):
            mask_i = 0
            for idx in indices_i:
                mask_i |= (1 << idx)
            
            if mask_i in mask2idx_m:
                Nm_masks.add(mask_c ^ mask_i)

    # Create mapping for (N-m) subspace (LE ascending order)
    sorted_Nm_masks = sorted(list(Nm_masks))
    mask2idx_Nm = {mask: idx for idx, mask in enumerate(sorted_Nm_masks)}
    
    # Pass 2: Calculate connections
    connections = {}
    for c_idx, mask_c in enumerate(basis_N.num_ele):
        indices_c = basis_N_indices[mask_c]
        
        for indices_i_tuple in itertools.combinations(indices_c, m):
            mask_i = 0
            for idx in indices_i_tuple:
                mask_i |= (1 << idx)
            
            if mask_i in mask2idx_m:
                ii = mask2idx_m[mask_i]
                mask_r = mask_c ^ mask_i
                r_idx = mask2idx_Nm[mask_r]
                
                # Calculate sign (Use Numba if available)
                if USE_NUMBA:
                    # Numba function expects numpy array (use int64 for safety)
                    indices_i_np = np.array(indices_i_tuple, dtype=np.int64)
                    sign = calculate_fermionic_sign_numba(mask_c, indices_i_np)
                else:
                    sign = calculate_fermionic_sign_python(mask_c, indices_i_tuple)
                
                connections[(r_idx, ii)] = (c_idx, sign)
    
    return connections, mask2idx_Nm, sorted_Nm_masks

def _process_chunk_connections_v3(args):
    """
    Worker utilizing Numba for the core computation loop (if available).
    """
    r_indices_chunk, connections, D_m, coord_dtype = args
    
    # Get the actual numpy type object for Numba compatibility
    coord_dtype_np = np.dtype(coord_dtype).type

    # Accumulate results locally (Python level, outside JIT scope)
    indices_list = []
    values_list = []

    # Iterate over intermediate states r
    for r_idx in r_indices_chunk:
        # Collect connections passing through this r_idx
        # Convert data structure from Dict to NumPy arrays for Numba/Optimized Python
        ii_list, c_idx_list, sign_list = [], [], []
        
        # This iteration is O(D_m)
        for ii in range(D_m):
            key = (r_idx, ii)
            if key in connections:
                c_idx, sign = connections[key]
                ii_list.append(ii)
                c_idx_list.append(c_idx)
                sign_list.append(sign)
        
        if not ii_list:
            continue
            
        # Prepare arrays
        ii_arr = np.array(ii_list, dtype=coord_dtype_np)
        c_idx_arr = np.array(c_idx_list, dtype=coord_dtype_np)
        sign_arr = np.array(sign_list, dtype=np.int8) # Signs are +1/-1

        # Call optimized function
        if USE_NUMBA:
            # Call the corrected Numba function (using pre-allocation)
            indices_np, values_np = compute_rdm_contributions_numba(
                ii_arr, c_idx_arr, sign_arr, coord_dtype_np)
        else:
            # Fallback to Python version (which now also uses pre-allocation)
            # We access the original Python function underlying the Numba wrapper.
            indices_np, values_np = compute_rdm_contributions_numba.py_func(
                 ii_arr, c_idx_arr, sign_arr, coord_dtype_np)

        
        if indices_np.size > 0:
            indices_list.append(indices_np)
            values_list.append(values_np)

    # Concatenate results before returning (This happens outside Numba JIT scope)
    if not indices_list:
        return np.empty((0, 4), dtype=coord_dtype), np.empty(0, dtype=float)
        
    indices_np = np.concatenate(indices_list, axis=0)
    values_np = np.concatenate(values_list, axis=0)
    
    return indices_np, values_np

# ---------------------------------------------------------------------
# generic ρ(m) tensor (Main function)
# ---------------------------------------------------------------------

def rho_m_gen(
    basis: FixedBasis, m: int, *, n_workers: int | None = None
) -> sparse.COO:
    """
    Build the sparse ρ(m) tensor (V3.1: Direct Connection Mapping with Numba Acceleration).
    """
    if basis.num is None:
         raise ValueError("basis.num must be set for RDM generation.")
    if chunked is None:
         raise ImportError("rho_m_gen requires the _parallel module.")

         
    # Ensure m_basis respects the same structure/restrictions if applicable
    m_basis = FixedBasis(basis.d, num=m, pairs=basis.pairs) 
    
    D_N = basis.size
    D_m = m_basis.size
    shape = (D_m, D_m, D_N, D_N)

    # Handle edge cases
    if m > basis.num or basis.num == 0 or m < 0:
        return sparse.COO([], [], shape=shape)
        
    if m == 0:
        eye = sp_sparse.eye(D_N, dtype=float)
        return sparse.COO(eye).reshape((1, 1, D_N, D_N))

    # Pre-calculate indices for basis (Optimization for subset iteration)
    basis_indices = {}
    for mask_c in basis.num_ele:
        # Ensure indices are sorted (required for correct sign calculation)
        basis_indices[mask_c] = sorted([i for i in range(basis.d) if (mask_c >> i) & 1])

    # 1. Calculate connections (Sequential, optimized with Numba signs if available)
    connections, mask2idx_Nm, sorted_Nm_masks = calculate_connections_v3(
        basis, m_basis, basis.d, basis_indices)
    
    D_Nm = len(sorted_Nm_masks)
    if D_Nm == 0:
        return sparse.COO([], [], shape=shape)

    # Memory Optimization: Determine optimal integer type
    max_dim = max(D_m, D_N)
    if max_dim <= np.iinfo(np.int16).max:
        coord_dtype = np.int16 # Sufficient for d=12
    elif max_dim <= np.iinfo(np.int32).max:
        coord_dtype = np.int32
    else:
        coord_dtype = np.int64

    # 2. Parallel computation (Multiprocessing + Numba workers)
    n_workers = _ensure_workers(n_workers)
    
    r_indices = np.arange(D_Nm)
    # Using more chunks improves load balancing
    num_chunks = max(n_workers * 8, 1) 
    chunks = np.array_split(r_indices, num_chunks)
    
    # Prepare iterable. 
    iterable = [(list(chunk), connections, D_m, coord_dtype) 
                for chunk in chunks if len(chunk) > 0]

    description = f"ρ_{m} (V3.1 Numba)" if USE_NUMBA else f"ρ_{m} (V3.1 Python)"
    results = chunked(
        _process_chunk_connections_v3,
        iterable,
        n_workers=n_workers,
        description=description,
    )

    # 3. Merge results
    idx_chunks = []
    val_chunks = []
    for idx_np, val_np in results:
        if idx_np.size > 0:
            idx_chunks.append(idx_np)
            val_chunks.append(val_np)

    if not idx_chunks:
        return sparse.COO([], [], shape=shape)

    # Final concatenation
    coords_T = np.concatenate(idx_chunks, axis=0).T
    data = np.concatenate(val_chunks, axis=0)
    
    # sparse.COO handles duplicate coordinates by summing them (required by this approach).
    return sparse.COO(coords_T, data, shape=shape)

# ---------------------------------------------------------------------
# legacy generic ρ(m) tensor
# ---------------------------------------------------------------------
def _process_m_chunk(
    args: Tuple[np.ndarray, FixedBasis, FixedBasis]
) -> Tuple[List[List[int]], List[float]]:
    """
    Worker that fills one slice of the sparse ρ(m) tensor.

    Parameters
    ----------
    args
        Tuple ``(ii_chunk, m_basis, full_basis)``.

    Returns
    -------
    (indices, values)
        *indices* is a list of ``[i, j, r, c]``; *values* the matching numbers.
    """
    ii_chunk, m_basis, basis = args
    indices, values = [], []

    for ii in ii_chunk:
        for jj in range(m_basis.size):
            op = m_basis.base[jj] * of.utils.hermitian_conjugated(m_basis.base[ii])
            mat = number_preserving_matrix(op, basis.d, basis.num)
            mat = restrict_sector_matrix(mat, basis.num_ele, basis.d, basis.num)
            rows, cols = mat.nonzero()
            indices.extend([[ii, jj, r, c] for r, c in zip(rows, cols)])
            values.extend(mat.data)

    return indices, values


# .....................................................................
@subspace_aware
def rho_m_gen_legacy(
    basis: FixedBasis, m: int, *, n_workers: int | None = None
) -> sparse.COO:
    """
    Build the sparse ρ(m) tensor with indices ``[i, j, r, c]``.

    Parameters
    ----------
    basis
        Full-system **N-body** basis (often built with ``num=N``).
    m
        Order of the reduced density matrix (1 ≤ m ≤ N).
    n_workers
        Number of processes for parallel execution (defaults to *all*).

    Returns
    -------
    sparse.COO
        Shape ``(m_dim, m_dim, N_dim, N_dim)`` where
        ``m_dim = |FixedBasis(d, num=m)|`` and
        ``N_dim = basis.size``.
    """
    m_basis = FixedBasis(basis.d, num=m)
    shape = (m_basis.size, m_basis.size, basis.size, basis.size)
    n_workers = _ensure_workers(n_workers)
    
    chunks = np.array_split(np.arange(m_basis.size), n_workers or 0)

    results = chunked(
        _process_m_chunk,
        [(chunk, m_basis, basis) for chunk in chunks],
        n_workers=n_workers,
        description=f"ρ_{m}",
    )

    # merge
    idx_list, val_list = [], []
    for idx, val in results:
        idx_list.extend(idx)
        val_list.extend(val)

    coords = np.asarray(idx_list).T
    return sparse.COO(coords, val_list, shape=shape)


# ---------------------------------------------------------------------
# specialised ρ₂ blocks for pairing Hamiltonians
# ---------------------------------------------------------------------
def _block_worker(
    args: Tuple[np.ndarray, FixedBasis, np.ndarray]
) -> Tuple[List[List[int]], List[float]]:
    """
    Worker that fills the *pair-scattering* block

        c†_k  c†_{\bar l}  c_{\bar j}  c_i

    used e.g. in BCS-like Hamiltonians.
    """
    chunk, basis, full_set = args
    indices: list[list[int]] = []
    values: list[float] = []

    m_pairs = basis.d // 2  # number of (k, \bar k) pairs

    for idx1 in chunk:
        i, j = divmod(int(idx1), m_pairs)
        # op₂ acts on (j, \bar j) indices
        op2 = of.FermionOperator(((2 * j + 1, 0), (2 * i, 0)))

        for idx2 in full_set:
            k, l = divmod(int(idx2), m_pairs)
            op1 = of.FermionOperator(((2 * k, 1), (2 * l + 1, 1)))

            op = op1 * op2
            mat = number_preserving_matrix(op, basis.d, basis.num)

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([idx1, idx2, r, c])
                values.append(v)

    return indices, values

# .....................................................................
@subspace_aware
def rho_2_block_gen(basis: FixedBasis, *, n_workers: int | None = None) -> sparse.COO:
    """
    Generate the *pair-scattering* block

        ρ₂[i j, k l] = ⟨c†_k  c†_{\bar l}  c_{\bar k}  c_l⟩

    with indices shaped ``(m², m², N_dim, N_dim)``, where
    ``m = d // 2`` equals the number of *time-reversed* orbital pairs.

    This is the block that appears in standard BCS correlation matrices.
    """
    m_pairs = basis.d // 2
    it_set = np.arange(m_pairs**2)
    n_workers = _ensure_workers(n_workers)

    chunks = np.array_split(it_set, n_workers or 0)

    results = chunked(
        _block_worker,
        [(chunk, basis, it_set) for chunk in chunks],
        n_workers=n_workers,
        description="ρ₂-block",
    )

    idx_list, val_list = [], []
    for idx, val in results:
        idx_list.extend(idx)
        val_list.extend(val)

    shape = (m_pairs**2, m_pairs**2, basis.size, basis.size)
    coords = np.asarray(idx_list).T
    return sparse.COO(coords, val_list, shape=shape)

# .....................................................................
def antisymmetrise_block(rho: sparse.COO) -> sparse.COO:
    """
    Project the ordered block ρ₂[i j , k l] onto the antisymmetric subspace
    and return ρ̄₂[ i<j , k<l ].
    """
    # infer the number of pairs from the size of the first index
    m_pairs = int(round(np.sqrt(rho.shape[0])))
    if m_pairs * m_pairs != rho.shape[0]:
        raise ValueError("ρ has not the expected shape (m², m², …)")

    # ---------------------------------------------------------------------
    # build the antisymmetriser  P  (same code as before)
    pairs = [(i, j) for i in range(m_pairs) for j in range(i + 1, m_pairs)]
    n_pairs = len(pairs)

    rows, cols, data = [], [], []
    for r, (i, j) in enumerate(pairs):
        rows.extend([r, r])
        cols.extend([i * m_pairs + j, j * m_pairs + i])
        data.extend([+1 / np.sqrt(2), -1 / np.sqrt(2)])

    P = sparse.COO(np.vstack([rows, cols]), data,
                   shape=(n_pairs, m_pairs ** 2))

    # ---------------------------------------------------------------------
    # ρ̄₂ = P · ρ₂ · Pᵀ
    tmp = sparse.tensordot(P, rho, axes=(1, 0))       # P · ρ
    rho_bar = sparse.tensordot(tmp, P.T, axes=(1, 0)) # … · Pᵀ
    return rho_bar

# ---------------------------------------------------------------------
# ρ₂ *k \bar k* diagonal block
# ---------------------------------------------------------------------
def _kkbar_worker(
    args: Tuple[np.ndarray, FixedBasis, np.ndarray]
) -> Tuple[List[List[int]], List[float]]:
    """
    Worker that fills the diagonal *k \\bar k* block

        c†_k  c†_{\bar k}  c_{\bar j}  c_j
    """
    chunk, basis, full_set = args
    indices: list[list[int]] = []
    values: list[float] = []

    for ii in chunk:
        ii = int(ii)  # pair index for the *row*
        op2 = of.FermionOperator(((2 * ii + 1, 0), (2 * ii, 0)))

        for jj in full_set:
            jj = int(jj)  # pair index for the *column*
            op1 = of.FermionOperator(((2 * jj, 1), (2 * jj + 1, 1)))

            op = op1 * op2
            mat = number_preserving_matrix(op, basis.d, basis.num)

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([ii, jj, r, c])
                values.append(v)

    return indices, values


# .....................................................................
@subspace_aware
def rho_2_kkbar_gen(basis: FixedBasis, *, n_workers: int | None = None) -> sparse.COO:
    """
    Generate the diagonal *k \\bar k* block

        ρ₂[k, j] = ⟨c†_k  c†_{\bar k}  c_{\bar j}  c_j⟩

    with shape ``(m, m, N_dim, N_dim)`` where ``m = d // 2``.
    """
    m_pairs = basis.d // 2
    it_set = np.arange(m_pairs)
    n_workers = _ensure_workers(n_workers)

    chunks = np.array_split(it_set, n_workers or 0)

    results = chunked(
        _kkbar_worker,
        [(chunk, basis, it_set) for chunk in chunks],
        n_workers=n_workers,
        description="ρ₂-k k̄",
    )

    idx_list, val_list = [], []
    for idx, val in results:
        idx_list.extend(idx)
        val_list.extend(val)

    shape = (m_pairs, m_pairs, basis.size, basis.size)
    coords = np.asarray(idx_list).T
    return sparse.COO(coords, val_list, shape=shape)


# ---------------------------------------------------------------------
# contraction helper
# ---------------------------------------------------------------------
def rho_m(state: np.ndarray, rho_arrays: sparse.COO) -> sparse.COO:
    """
    Contract a *state* (ket, density matrix, or batch) with a pre-built ρ(m).

    Parameters
    ----------
    state
        • |Ψ⟩ ∈ ℂ^dim                → returns ``⟨Ψ|ρ(m)|Ψ⟩`` (*dense* 2-D block).  
        • ρ   ∈ ℂ^{dim×dim}          → returns ``Tr₂[ρ ⋅ ρ(m)]``.  
        • batch |Ψ_b⟩ ∈ ℂ^{B×dim}    → returns a *batch* of 2-D blocks.
    rho_arrays
        Tensor built by :func:`rho_m_gen` or the specialised helpers above.
    """
    ndim = state.ndim
    if ndim == 1:  # single ket
        return sparse.einsum("k,ijkl,l->ij", state, rho_arrays, state)

    if ndim == 2:  # density matrix
        return sparse.einsum("ijkl,kl->ij", rho_arrays, state)

    if ndim == 3:  # batch of kets
        return sparse.einsum("bkl,ijkl->bij", state, rho_arrays)

    raise ValueError("`state` must be a ket, density matrix, or batch thereof")

# =====================================================================
# DIRECT M-RDM CALCULATION 
# =====================================================================

@njit(cache=True)
def compute_outer_product_contribution_numba(partial_rdm, ii_arr, V_r_arr):
    """
    Numba-accelerated calculation of outer product contribution to RDM.
    Updates partial_rdm in place: RDM += V_r^\dagger @ V_r
    """
    n = len(ii_arr)
    is_complex = partial_rdm.dtype.kind == 'c'

    # Pre-calculate conjugation
    conj_V_r_arr = np.conj(V_r_arr)

    for i in range(n):
        ii = ii_arr[i]
        conj_V_i = conj_V_r_arr[i]
        
        # Leverage Hermitian property RDM[ii, jj] = conj(RDM[jj, ii])
        for j in range(i, n):
            jj = ii_arr[j]
            V_j = V_r_arr[j]
            
            # Contribution: RDM[ii, jj] += conj(V_i) * V_j
            contribution = conj_V_i * V_j
            
            partial_rdm[ii, jj] += contribution
            
            if i != j:
                # Add the symmetric part RDM[jj, ii]
                if is_complex:
                    partial_rdm[jj, ii] += np.conj(contribution)
                else:
                    # If real, the contribution is real.
                    partial_rdm[jj, ii] += contribution

def _process_chunk_direct_rdm(args):
    """
    Worker function for direct M-RDM calculation.
    Computes the contribution to the RDM from a chunk of (N-m) states (R).
    """
    r_indices_chunk, connections, D_m, psi = args
    
    # Determine dtype for the RDM, ensuring sufficient precision.
    if psi.dtype.kind == 'c':
        rdm_dtype = np.complex128 if np.dtype(psi.dtype).itemsize < 16 else psi.dtype
    else:
        rdm_dtype = np.float64 if np.dtype(psi.dtype).itemsize < 8 else psi.dtype

    # Initialize partial RDM matrix for this chunk
    partial_rdm = np.zeros((D_m, D_m), dtype=rdm_dtype)

    # Iterate over intermediate states r in the chunk
    for r_idx in r_indices_chunk:
        # Calculate the intermediate vector V_r (only non-zero elements)
        ii_list = []
        V_r_elements = []
        
        # Iterate over m-body basis I (O(D_m))
        for ii in range(D_m):
            key = (r_idx, ii)
            if key in connections:
                c_idx, sign = connections[key]
                
                # Optimization: only consider contributions if psi[c_idx] is non-zero
                if psi[c_idx] != 0:
                    # Ensure type consistency for accumulation
                    V_r_element = rdm_dtype.type(psi[c_idx] * sign)
                    ii_list.append(ii)
                    V_r_elements.append(V_r_element)

        if not V_r_elements:
            continue
            
        # Convert to numpy arrays for Numba
        ii_arr = np.array(ii_list, dtype=np.int64) # Indices
        V_r_arr = np.array(V_r_elements, dtype=rdm_dtype)
        
        # Calculate outer product contribution and add to partial_rdm
        if USE_NUMBA:
            compute_outer_product_contribution_numba(partial_rdm, ii_arr, V_r_arr)
        else:
            # Python fallback
            try:
                compute_outer_product_contribution_numba.py_func(partial_rdm, ii_arr, V_r_arr)
            except (AttributeError, TypeError):
                # If @njit acts as a passthrough (e.g. Numba disabled or not installed)
                compute_outer_product_contribution_numba(partial_rdm, ii_arr, V_r_arr)

    return partial_rdm

def rho_m_direct(basis_N: FixedBasis, m: int, psi: np.ndarray, *, n_workers: int | None = None) -> np.ndarray:
    """
    Directly calculates the m-body Reduced Density Matrix (M-RDM) for a given state psi.

    This implementation bypasses the creation of the full 4-tensor operator arrays (rho_m_gen),
    providing significant speedup and memory savings.
    It uses optimized connection mapping, multiprocessing, and Numba acceleration.
    """
    if basis_N.num is None:
         raise ValueError("basis_N.num (N) must be set for RDM calculation.")
    
    if psi.ndim != 1 or psi.shape[0] != basis_N.size:
        raise ValueError(f"psi must be a 1D state vector matching the basis size {basis_N.size}.")

    N = basis_N.num
    D = basis_N.d
    
    # Setup m-basis
    m_basis = FixedBasis(D, num=m, pairs=basis_N.pairs) 
    D_m = m_basis.size

    # Determine RDM dtype (matching logic in worker)
    if psi.dtype.kind == 'c':
        rdm_dtype = np.complex128 if np.dtype(psi.dtype).itemsize < 16 else psi.dtype
    else:
        rdm_dtype = np.float64 if np.dtype(psi.dtype).itemsize < 8 else psi.dtype

    # Handle edge cases
    if m > N or N == 0 or m < 0:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)
        
    if m == 0:
        # Norm squared (Tr[rho])
        norm_sq = np.dot(np.conj(psi), psi)
        return np.array([[norm_sq]], dtype=rdm_dtype)

    # 1. Pre-calculate indices for basis_N (Required by calculate_connections_v3)
    basis_indices = {}
    for mask_c in basis_N.num_ele:
        # Ensure mask_c is standard int and indices are sorted
        mask_c_int = int(mask_c)
        basis_indices[mask_c_int] = sorted([i for i in range(D) if (mask_c_int >> i) & 1])

    # 2. Calculate connections (Reuses the optimized logic from rho_m_gen)
    # Note: calculate_connections_v3 internally uses USE_NUMBA for sign calculations.
    connections, mask2idx_Nm, sorted_Nm_masks = calculate_connections_v3(
        basis_N, m_basis, D, basis_indices)
    
    D_Nm = len(sorted_Nm_masks)
    if D_Nm == 0:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)

    # 3. Parallel computation (Multiprocessing + Numba workers)
    n_workers = _ensure_workers(n_workers)
    
    r_indices = np.arange(D_Nm)
    # Use significantly more chunks than workers for better load balancing
    num_chunks = max(n_workers * 16, 1) 
    
    if D_Nm > 0:
        chunks = np.array_split(r_indices, min(num_chunks, D_Nm))
    else:
        chunks = []
    
    # Prepare iterable. Pass psi to the workers.
    iterable = [(list(chunk), connections, D_m, psi) 
                for chunk in chunks if len(chunk) > 0]

    description = f"RDM_{m} Direct (Numba)" if USE_NUMBA else f"RDM_{m} Direct (Python)"
    results = chunked(
        _process_chunk_direct_rdm,
        iterable,
        n_workers=n_workers,
        description=description,
    )

    # 4. Merge results (Sum partial RDMs)
    RDM_m = np.sum(results, axis=0)
    
    # Final check for real input/output (numerical noise mitigation)
    if np.isrealobj(psi) and np.isrealobj(RDM_m):
        return RDM_m.real
    else:
        return RDM_m

