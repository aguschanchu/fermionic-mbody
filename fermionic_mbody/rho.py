"""
Generators for m-body reduced density-matrix tensors ρ(m), specialised
pairing-blocks, and a convenience contraction helper rho_m.

Conventions:
We adopt the following conventions throughout the library (consistent with OpenFermion):
- Basis Ordering: We use the Jordan-Wigner transformation with Little-Endian (LE) encoding.
  Fixed-N sectors are ordered lexicographically.
- M-body basis {|i⟩}: Defined by creation strings C†ᵢ = c†_{i₁}⋯c†_{iₘ} with i₁ < ⋯ < iₘ.
- M-RDM Definition: RDM[i, j] = ⟨Ψ| C†ⱼ Cᵢ |Ψ⟩.
- Generator Tensor T: T[i, j, k, l] = ⟨k| C†ⱼ Cᵢ |l⟩, where |k⟩, |l⟩ are N-body states.

All heavy lifting is off-loaded to
fermibasis._parallel.chunked, which wraps a multiprocessing.Pool.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple, Dict, Optional, Sequence, Callable

import numpy as np
import openfermion as of
import sparse
from multiprocessing import cpu_count
from math import comb
from functools import wraps
from scipy import sparse as sp_sparse

from ._parallel import chunked
from .basis import FixedBasis
from ._ofsparse import number_preserving_matrix, restrict_sector_matrix, of_sector_bitmasks

__all__ = [
    "rho_m_gen",
    "rho_m_gen_legacy",   
    "rho_2_block_gen",
    "antisymmetrise_block",
    "rho_2_kkbar_gen",
    "rho_m",
    "rho_m_direct",
]

# ---------------------------------------------------------------------
# Numba configuration and imports
# ---------------------------------------------------------------------

try:
    import numba
    from numba import njit
    USE_NUMBA = True
except ImportError:
    # If Numba is not installed, disable optimization and provide dummy decorators.
    USE_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            if not hasattr(func, 'py_func'):
                 func.py_func = func
            return func
        
        if len(args) == 1 and callable(args[0]) and not kwargs:
             return decorator(args[0])
        return decorator

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _ensure_workers(n_workers: int | None) -> int:
    """Return a strictly positive worker count (defaults to all CPUs)."""
    return max(1, n_workers or cpu_count())

# =====================================================================
# RHO_M_GEN IMPLEMENTATION 
# =====================================================================

# ---------------------------------------------------------------------
# Numba Helper Functions
# ---------------------------------------------------------------------

def _popcount_python(n):
    """Fast population count (number of set bits) using optimized methods."""
    n = int(n) 
    if hasattr(int, 'bit_count'):
        return n.bit_count()
    else:
        count = 0
        while n > 0:
            n &= (n - 1) 
            count += 1
        return count

# Apply njit decorator 
popcount = njit(cache=True)(_popcount_python)

@njit(cache=True)
def calculate_fermionic_sign_numba(mask_c, indices_i: np.ndarray):
    """
    Calculates the fermionic sign (Jordan-Wigner, LE convention).
    indices_i must be a sorted numpy array.
    """
    sign = 1
    current_mask = int(mask_c)
    
    # Apply operators sequentially (lowest index first)
    for idx in indices_i:
        # Calculate sign based on parity of electrons before idx
        mask_before = (1 << idx) - 1
        
        # Use unified optimized popcount
        count = popcount(current_mask & mask_before)
            
        if (count & 1) == 1: # Check parity (if odd)
            sign = -sign
            
        # Update the mask (annihilate electron at idx)
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

def calculate_connections_v3(basis_N: FixedBasis, m_basis: FixedBasis, d: int, basis_N_indices: Dict[int, List[int]], statistics: str = 'fermionic'):
    """
    Calculates the connection map using optimized subset iteration.
    Uses Numba signs if available.
    """
    N = basis_N.num
    if N is None: return {}, {}, []
    m = m_basis.num
    mask2idx_m = m_basis._mask2idx_cache

    # Pass 1: Identify Nm_masks (Python loop, optimized by subset iteration)
    Nm_masks = set()
    for mask_c in basis_N.bitmasks:
        mask_c = int(mask_c)
        indices_c = basis_N_indices[mask_c]
        # Iterate over all combinations of size m (subsets)
        for indices_i in itertools.combinations(indices_c, m):
            mask_i = 0
            for idx in indices_i:
                mask_i |= (1 << idx)
            
            if mask_i in mask2idx_m:
                Nm_masks.add(mask_c ^ mask_i)

    # Create mapping for (N-m) subspace
    Nm_order = of_sector_bitmasks(d, N - m)
    Nm_set = {int(x) for x in Nm_masks}
    sorted_Nm_masks = [int(x) for x in Nm_order if int(x) in Nm_set]
    mask2idx_Nm = {mask: idx for idx, mask in enumerate(sorted_Nm_masks)}

    
    # Pass 2: Calculate connections
    connections = {}
    for c_idx, mask_c in enumerate(basis_N.bitmasks):
        mask_c = int(mask_c)
        indices_c = basis_N_indices[mask_c]
        
        for indices_i_tuple in itertools.combinations(indices_c, m):
            mask_i = 0
            for idx in indices_i_tuple:
                mask_i |= (1 << idx)
            
            if mask_i in mask2idx_m:
                ii = mask2idx_m[mask_i]
                mask_r = mask_c ^ mask_i
                r_idx = mask2idx_Nm[mask_r]
                
                if statistics == 'bosonic':
                    sign = 1
                else:
                    indices_i_np = np.array(indices_i_tuple, dtype=np.int64)
                    sign = calculate_fermionic_sign_numba(mask_c, indices_i_np)

                
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

        indices_np, values_np = compute_rdm_contributions_numba(
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
# Optimization helpers for paired systems (Isomorphism)
# ---------------------------------------------------------------------

def _paired_reduce_mask(mask_full: int, m_pairs: int) -> int:
    """
    Map a full d-mode paired mask (bits 2p, 2p+1) to a d/2-mode reduced mask:
    bit p is 1 iff both bits (2p, 2p+1) are 1.
    """
    red = 0
    for p in range(m_pairs):
        a = (mask_full >> (2*p)) & 1
        b = (mask_full >> (2*p + 1)) & 1
        if (a & b) == 1:
            red |= (1 << p)
    return red

def _rho_m_gen_paired_isomorphism(basis: FixedBasis, m: int, n_workers: int | None) -> sparse.COO:
    """Handles the optimized RDM generator calculation for pairs=True using isomorphism."""
    
    # 1. Define the reduced system parameters
    d_red = basis.d // 2
    N = basis.num
    P = None # Number of pairs (P=N/2)

    if N is not None and N % 2 == 0:
        P = N // 2
    
    # 2. Determine the dimensions of the resulting tensor.
    D_N = basis.size
    
    # D_m corresponds to the m-pair basis size C(d/2, m).
    try:
        # Construct the reduced m-basis (standard fermionic) to get the size.
        basis_m_red = FixedBasis(d_red, m, pairs=False)
        D_m = basis_m_red.size
    except Exception:
        D_m = 0

    shape = (D_m, D_m, D_N, D_N)

    # 3. Handle edge cases
    if D_N == 0 or D_m == 0 or m < 0:
        return sparse.COO([], [], shape=shape)
        
    if m == 0:
        # m=0 RDM is the identity tensor.
        eye = sp_sparse.eye(D_N, dtype=float)
        return sparse.COO(eye).reshape((1, 1, D_N, D_N))

    # 4. Construct the reduced N-body basis (P particles in d/2 modes).
    try:
        basis_P_red = FixedBasis(d_red, P, pairs=False)
    except Exception:
        return sparse.COO([], [], shape=shape)

    # 5. Critical check for isomorphism. The optimization relies on FixedBasis preserving the order.
    if basis.size != basis_P_red.size:
        raise RuntimeError(f"Isomorphism assumption failed. Paired basis size: {basis.size}, Reduced basis size: {basis_P_red.size}.")

    # 6. Recursive call: Calculate the m-RDM in the reduced basis.
    T_red = rho_m_gen(basis_P_red, m, n_workers=n_workers, _statistics='bosonic')

    D_N = basis.size
    full_masks = [int(x) for x in basis.bitmasks]
    red_masks  = [int(x) for x in basis_P_red.bitmasks]

    # Build red_mask -> full_index via pairing reduction
    m_pairs = basis.d // 2
    red_to_full = {
        _paired_reduce_mask(k_full, m_pairs): idx_full
        for idx_full, k_full in enumerate(full_masks)
    }
    perm = np.array([red_to_full[k_red] for k_red in red_masks], dtype=np.int64)

    # Fast path: identical order
    if np.array_equal(perm, np.arange(D_N, dtype=np.int64)):
        return T_red

    # Remap last two axes (k,l) from reduced order to parent basis order
    coords = T_red.coords
    new_coords = np.vstack([coords[0],
                            coords[1],
                            perm[coords[2]],
                            perm[coords[3]]])
    return sparse.COO(new_coords, T_red.data, shape=T_red.shape)

# ---------------------------------------------------------------------
# generic ρ(m) tensor (Main function)
# ---------------------------------------------------------------------

def rho_m_gen(
    basis: FixedBasis, m: int, *, n_workers: int | None = None, _statistics: str = 'fermionic'
) -> sparse.COO:
    """
    Build the sparse ρ(m) tensor
    """

    if basis.pairs and _statistics == 'fermionic':
        return _rho_m_gen_paired_isomorphism(basis, m, n_workers=n_workers)

    if basis.num is None:
         raise ValueError("basis.num must be set for RDM generation.")
    if chunked is None:
         raise ImportError("rho_m_gen requires the _parallel module.")

         
    # Ensure m_basis respects the same structure/restrictions if applicable
    m_basis = FixedBasis(basis.d, num=m, pairs=False) 
    
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
    for mask_c in basis.bitmasks:
        mask_c = int(mask_c)
        # Ensure indices are sorted (required for correct sign calculation)
        basis_indices[mask_c] = sorted([i for i in range(basis.d) if (mask_c >> i) & 1])

    # 1. Calculate connections (Sequential, optimized with Numba signs if available)
    connections, mask2idx_Nm, sorted_Nm_masks = calculate_connections_v3(
        basis, m_basis, basis.d, basis_indices, _statistics)
    
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

    description = f"ρ_{m}" if USE_NUMBA else f"ρ_{m}"
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
        Tuple (ii_chunk, m_basis, full_basis).

    Returns
    -------
    (indices, values)
        indices is a list of [i, j, r, c]; values the matching numbers.
    """
    ii_chunk, m_basis, basis = args
    indices, values = [], []

    for ii in ii_chunk:
        for jj in range(m_basis.size):
            op = m_basis.base[jj] * of.utils.hermitian_conjugated(m_basis.base[ii])
            mat = basis.get_operator_matrix(op).tocoo()
            rows, cols = mat.nonzero()
            indices.extend([[ii, jj, r, c] for r, c in zip(rows, cols)])
            values.extend(mat.data)

    return indices, values


# .....................................................................
def rho_m_gen_legacy(
    basis: FixedBasis, m: int, *, n_workers: int | None = None
) -> sparse.COO:
    """
    Build the sparse ρ(m) tensor with indices [i, j, r, c].

    Parameters
    ----------
    basis
        Full-system N-body basis (built with num=N).
    m
        Order of the reduced density matrix (1 ≤ m ≤ N).
    n_workers
        Number of processes for parallel execution (defaults to all).

    Returns
    -------
    sparse.COO
        Shape (m_dim, m_dim, N_dim, N_dim) where
        m_dim = |FixedBasis(d, num=m)| and
        N_dim = basis.size.
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
    Worker that fills the pair-scattering block

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
            mat = basis.get_operator_matrix(op).tocoo()

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([idx1, idx2, r, c])
                values.append(v)

    return indices, values

# .....................................................................
def rho_2_block_gen(basis: FixedBasis, *, n_workers: int | None = None) -> sparse.COO:
    """
    Generate the pair-scattering block

        ρ₂[i j, k l] = ⟨c†_k  c†_{\bar l}  c_{\bar k}  c_l⟩

    with indices shaped (m², m², N_dim, N_dim), where
    m = d // 2 equals the number of time-reversed orbital pairs.

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
    Worker that fills the diagonal k \bar k block

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
            mat = basis.get_operator_matrix(op).tocoo()

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([ii, jj, r, c])
                values.append(v)

    return indices, values


# .....................................................................
def rho_2_kkbar_gen(basis: FixedBasis, *, n_workers: int | None = None) -> sparse.COO:
    """
    Generate the diagonal k \bar k block

        ρ₂[k, j] = ⟨c†_k  c†_{\bar k}  c_{\bar j}  c_j⟩

    with shape (m, m, N_dim, N_dim where m = d // 2.
    """

    if basis.pairs:
        return rho_m_gen(basis, 1, n_workers=n_workers)

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
    Contract a state (ket, density matrix, or batch) with a pre-built ρ(m).

    Calculates the RDM according to the library convention: 
    RDM[i,j] = <Ψ| O_ij |Ψ> where O_ij = C_j† C_i.
    Uses the generator T[i, j, k, l] = <k| O_ij |l>.

    Parameters
    ----------
    state : ndarray
        The state to contract.
        • Ket |Ψ⟩ (1D): RDM[i,j] = Σ_kl Ψ*_k T_ijkl Ψ_l.
        • Density matrix ρ (2D): RDM[i,j] = Tr[T ρ] = Σ_kl T_ijkl ρ_lk.
        • Batch of kets (2D, B x N_dim).
        • Batch of density matrices (3D, B x N_dim x N_dim).
        
    rho_arrays : sparse.COO
        Tensor built by rho_m_gen.
    """
    ndim = state.ndim
    N_dim = rho_arrays.shape[2]

    # Case 1: Single ket (1D)
    if ndim == 1:
        if state.shape[0] != N_dim:
             raise ValueError(f"State dimension mismatch. Expected ({N_dim},), got {state.shape}.")
        return sparse.einsum("k,ijkl,l->ij", np.conj(state), rho_arrays, state)

    # Case 2: Density Matrix (2D, square) or Batch of Kets (2D, B x dim)
    if ndim == 2:
        if state.shape == (N_dim, N_dim): # Density matrix ρ
            return sparse.einsum("ijkl,lk->ij", rho_arrays, state)
        
        elif state.shape[1] == N_dim: # Batch of kets (B, dim)
            return sparse.einsum("bk,ijkl,bl->bij", np.conj(state), rho_arrays, state)
        
        else:
            raise ValueError(f"Invalid shape for 2D state: {state.shape}. Expected ({N_dim}, {N_dim}) or (B, {N_dim}).")

    # Case 3: Batch of Density Matrices (3D, B x dim x dim)
    if ndim == 3:
        if state.shape[1:] == (N_dim, N_dim):
            # RDM[b, i, j] = Σ_{k,l} T[i, j, k, l] ρ_b[l, k]
            return sparse.einsum("ijkl,blk->bij", rho_arrays, state)
        else:
             raise ValueError(f"Invalid shape for 3D state: {state.shape}. Expected (B, {N_dim}, {N_dim}).")

    raise ValueError("state must be a ket, density matrix, or batch thereof (ndim 1-3 supported).")

# =====================================================================
# DIRECT M-RDM CALCULATION 
# =====================================================================

def calculate_connections_csr(basis_N: FixedBasis, m_basis: FixedBasis, d: int, basis_N_indices: Dict[int, List[int]], statistics: str = 'fermionic'):
    """
    Calculates the connection map and returns a CSR-like representation (dict of numpy arrays)
    """
    N = basis_N.num
    if N is None: return None, {}, []
    m = m_basis.num
    mask2idx_m = m_basis._mask2idx_cache
    D_N = basis_N.size
    D_m = m_basis.size

    max_dim = max(D_N, D_m)
    # Use int32 if sufficient, otherwise int64.
    if max_dim <= np.iinfo(np.int32).max:
        idx_dtype = np.int32
    else:
        idx_dtype = np.int64

    # Identify Nm_masks (N-m subspace)
    Nm_masks = set()
    for mask_c in basis_N.bitmasks:
        mask_c_int = int(mask_c)
        indices_c = basis_N_indices[mask_c_int]
        # Iterate over m-sized subsets
        for indices_i in itertools.combinations(indices_c, m):
            mask_i = sum(1 << i for i in indices_i)
            
            if mask_i in mask2idx_m:
                # mask_r = mask_c XOR mask_i
                Nm_masks.add(mask_c_int ^ mask_i)

    # Create mapping for (N-m) subspace
    Nm_order = of_sector_bitmasks(d, N - m)
    Nm_set = {int(x) for x in Nm_masks}
    sorted_Nm_masks = [int(x) for x in Nm_order if int(x) in Nm_set]
    mask2idx_Nm = {mask: idx for idx, mask in enumerate(sorted_Nm_masks)}
    D_Nm = len(sorted_Nm_masks)

    # alculate connections and organize by r_idx (Temporary structure: list of lists)
    connections_per_r = [[] for _ in range(D_Nm)]

    # Iterate over N-basis (c_idx)
    for c_idx, mask_c in enumerate(basis_N.bitmasks):
        mask_c_int = int(mask_c)
        indices_c = basis_N_indices[mask_c_int]
        
        # Iterate over m-subsets (ii)
        for indices_i_tuple in itertools.combinations(indices_c, m):
            mask_i = sum(1 << i for i in indices_i_tuple)
            
            if mask_i in mask2idx_m:
                ii = mask2idx_m[mask_i]
                mask_r = mask_c_int ^ mask_i
                r_idx = mask2idx_Nm[mask_r]
                if statistics == 'bosonic':
                    sign = 1
                else:
                    indices_i_np = np.array(indices_i_tuple, dtype=np.int64)
                    sign = calculate_fermionic_sign_numba(mask_c_int, indices_i_np)

                
                connections_per_r[r_idx].append((ii, c_idx, sign))

    # Construct the CSR structure
    indices_ii = []
    data_c_idx = []
    data_sign = []
    # Use int64 for pointers (indptr) to ensure sufficient range
    indptr = np.zeros(D_Nm + 1, dtype=np.int64)
    
    current_ptr = 0
    for r_idx in range(D_Nm):
        conns = connections_per_r[r_idx]
        # Sorting by ii improves memory access patterns in the kernel
        conns.sort(key=lambda x: x[0])
        
        for ii, c_idx, sign in conns:
            indices_ii.append(ii)
            data_c_idx.append(c_idx)
            data_sign.append(sign)
        
        current_ptr += len(conns)
        indptr[r_idx+1] = current_ptr

    # Convert to numpy arrays with optimal types
    connections_csr = {
        'indptr': indptr,
        'indices_ii': np.array(indices_ii, dtype=idx_dtype),
        'data_c_idx': np.array(data_c_idx, dtype=idx_dtype),
        'data_sign': np.array(data_sign, dtype=np.int8) # Signs are +1/-1
    }

    return connections_csr, mask2idx_Nm, sorted_Nm_masks

@njit(cache=True, nogil=True)
def compute_rdm_chunk_from_csr(r_indices_chunk, C_idx_data, C_idx_indices, C_idx_indptr, C_sign_data, psi, D_m, rdm_dtype_np):
    """
    Numba-accelerated calculation of the RDM contribution using the CSR connection map.
    This kernel is optimized for performance and handles sparse psi efficiently.
    Releases the GIL to enable true parallel computation.
    """
    partial_rdm = np.zeros((D_m, D_m), dtype=rdm_dtype_np)
    
    # Iterate over r_idx (N-m states) in the chunk
    for r_idx in r_indices_chunk:
        
        # Get the range of connections for this r_idx from CSR indptr
        start = C_idx_indptr[r_idx]
        end = C_idx_indptr[r_idx+1]
        
        # Iterate over pairs of connections (i, j) originating from the same r_idx.
        # This nested loop structure computes RDM += Sum_r (V_r @ V_r^H) efficiently.
        
        for idx_i in range(start, end):
            ii = C_idx_indices[idx_i]
            c_idx_i = C_idx_data[idx_i]
            sign_i = C_sign_data[idx_i]
            
            # Optimization: Skip if the state amplitude is zero (handles sparse psi)
            if psi[c_idx_i] == 0:
                continue
            
            # Calculate V_i(r) = psi[c_i] * sign_i
            # Ensure type consistency
            V_i = rdm_dtype_np(psi[c_idx_i] * sign_i)
            
            # Iterate over the second connection (j >= i) to leverage Hermitian symmetry
            for idx_j in range(idx_i, end):
                jj = C_idx_indices[idx_j]
                c_idx_j = C_idx_data[idx_j]
                sign_j = C_sign_data[idx_j]
                
                if psi[c_idx_j] == 0:
                    continue

                # Calculate V_j(r) = psi[c_j] * sign_j
                V_j = rdm_dtype_np(psi[c_idx_j] * sign_j)
                
                # Contribution to RDM[ii, jj]
                # RDM[i, j] = Sum_r V_i(r) * conj(V_j(r))
                contribution = np.conj(V_j) * V_i
                
                partial_rdm[ii, jj] += contribution
                
                if idx_i != idx_j:
                    # Add symmetric part RDM[jj, ii] = conj(contribution)
                    partial_rdm[jj, ii] += np.conj(contribution)

    return partial_rdm

def _process_chunk_direct_rdm_opt(args):
    """
    Worker function wrapper for the optimized Numba kernel.
    """
    # r_indices_chunk_np is passed as a numpy array
    r_indices_chunk_np, connections_csr, D_m, psi, rdm_dtype = args
    
    # Unpack CSR structure (These are compact NumPy arrays)
    indptr = connections_csr['indptr']
    indices_ii = connections_csr['indices_ii']
    data_c_idx = connections_csr['data_c_idx']
    data_sign = connections_csr['data_sign']
    
    # Get the numpy type object for Numba compatibility
    rdm_dtype_np = np.dtype(rdm_dtype).type

    # Call the Numba optimized function (which releases the GIL)
    partial_rdm = compute_rdm_chunk_from_csr(
        r_indices_chunk_np,
        data_c_idx, indices_ii, indptr, data_sign,
        psi, D_m, rdm_dtype_np
    )

    return partial_rdm

def _rho_m_direct_paired_isomorphism(basis_N: FixedBasis, m: int, psi: np.ndarray, n_workers: int | None) -> np.ndarray:
    """Handles optimized direct RDM calculation for pairs=True using isomorphism."""

    # 1. Define the reduced system parameters
    d_red = basis_N.d // 2
    N = basis_N.num
    P = None

    if N is not None and N % 2 == 0:
        P = N // 2
    
    # 2. Determine RDM dimensions (D_m for m-pair RDM)
    try:
        basis_m_red = FixedBasis(d_red, m, pairs=False)
        D_m = basis_m_red.size
    except Exception:
        D_m = 0

    # 3. Determine RDM dtype
    if psi.dtype.kind == 'c':
        rdm_dtype = np.complex128 if np.dtype(psi.dtype).itemsize < 16 else psi.dtype
    else:
        rdm_dtype = np.float64 if np.dtype(psi.dtype).itemsize < 8 else psi.dtype

    # 4. Handle edge cases
    if basis_N.size == 0 or D_m == 0 or m < 0:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)

    if m == 0:
        norm_sq = np.dot(np.conj(psi), psi)
        return np.array([[norm_sq]], dtype=rdm_dtype)

    # 5. Construct the reduced N-body basis (P particles).
    try:
        basis_P_red = FixedBasis(d_red, P, pairs=False)
    except Exception:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)

    # 6. Critical check for isomorphism.
    if basis_N.size != basis_P_red.size:
        raise RuntimeError(f"Isomorphism assumption failed in rho_m_direct. Paired basis size: {basis_N.size}, Reduced basis size: {basis_P_red.size}.")

    # 7. Recursive call: psi remains the same as the basis ordering is preserved by the isomorphism.
    full_masks = [int(x) for x in basis_N.bitmasks]
    red_masks  = [int(x) for x in basis_P_red.bitmasks]

    m_pairs = basis_N.d // 2
    red_to_full = {
        _paired_reduce_mask(k_full, m_pairs): idx_full
        for idx_full, k_full in enumerate(full_masks)
    }
    perm = np.array([red_to_full[k_red] for k_red in red_masks], dtype=np.int64)

    # psi_red[r] = psi_full[perm[r]]
    psi_red = psi[perm] if not np.array_equal(perm, np.arange(basis_N.size)) else psi

    return rho_m_direct(basis_P_red, m, psi_red, n_workers=n_workers, _statistics='bosonic')

def rho_m_direct(basis_N: FixedBasis, m: int, psi: np.ndarray, *, n_workers: int | None = None, _statistics: str = 'fermionic') -> np.ndarray:
    """
    Directly calculates the m-body Reduced Density Matrix (M-RDM) for a given state psi.
    """

    if psi.ndim != 1 or psi.shape[0] != basis_N.size:
        raise ValueError(f"psi must be a 1D state vector matching the basis size {basis_N.size}.")

    # Optimization for pairs=True
    if basis_N.pairs and _statistics == 'fermionic':
        return _rho_m_direct_paired_isomorphism(basis_N, m, psi, n_workers=n_workers)

    if basis_N.num is None:
         raise ValueError("basis_N.num (N) must be set for RDM calculation.")
    
    if psi.ndim != 1 or psi.shape[0] != basis_N.size:
        raise ValueError(f"psi must be a 1D state vector matching the basis size {basis_N.size}.")

    N = basis_N.num
    D = basis_N.d
    
    # Setup m-basis
    m_basis = FixedBasis(D, num=m, pairs=False) 
    D_m = m_basis.size

    # Determine RDM dtype
    if psi.dtype.kind == 'c':
        rdm_dtype = np.complex128 if np.dtype(psi.dtype).itemsize < 16 else psi.dtype
    else:
        rdm_dtype = np.float64 if np.dtype(psi.dtype).itemsize < 8 else psi.dtype

    # Handle edge cases
    if m > N or N == 0 or m < 0:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)
        
    if m == 0:
        norm_sq = np.dot(np.conj(psi), psi)
        return np.array([[norm_sq]], dtype=rdm_dtype)

    # Pre-calculate indices for basis_N (Required for subset iteration)
    basis_indices = {}
    for mask_c in basis_N.bitmasks:
        mask_c_int = int(mask_c)
        # Ensure indices are sorted (crucial for correct sign calculation)
        basis_indices[mask_c_int] = sorted([i for i in range(D) if (mask_c_int >> i) & 1])

    # Calculate connections using the optimized CSR function
    # This avoids the creation of the large intermediate dictionary.
    connections_csr, mask2idx_Nm, sorted_Nm_masks = calculate_connections_csr(
        basis_N, m_basis, D, basis_indices, _statistics)
    
    D_Nm = len(sorted_Nm_masks)
    if D_Nm == 0 or connections_csr is None:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)

    # Parallel computation (Multiprocessing + Numba workers with GIL release)
    n_workers = _ensure_workers(n_workers)
    
    r_indices = np.arange(D_Nm)
    # Use significantly more chunks than workers for better load balancing
    num_chunks = max(n_workers * 32, 1)
    
    if D_Nm > 0:
        # Split into numpy arrays directly
        chunks_np = np.array_split(r_indices, min(num_chunks, D_Nm))
    else:
        chunks_np = []
    
    # Prepare iterable. Pass the compact CSR structure.
    iterable = [(chunk_np, connections_csr, D_m, psi, rdm_dtype) 
                for chunk_np in chunks_np if len(chunk_np) > 0]

    description = f"RDM_{m} Direct (Optimized)"
    
    # Use the existing 'chunked' utility for parallel processing
    results = chunked(
        _process_chunk_direct_rdm_opt, # Use the optimized worker
        iterable,
        n_workers=n_workers,
        description=description,
    )

    # Merge results (Sum partial RDMs)
    if not results:
        return np.zeros((D_m, D_m), dtype=rdm_dtype)

    RDM_m = np.sum(results, axis=0)
    
    # Final check for real input/output (numerical noise mitigation)
    if np.isrealobj(psi):
        if RDM_m.dtype.kind == 'c':
            # If input was real, RDM must be real. Discard numerical noise in imaginary part.
            return RDM_m.real
        else:
            return RDM_m
    
    return RDM_m
