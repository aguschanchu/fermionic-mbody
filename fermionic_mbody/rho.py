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

from ._parallel import chunked
from .basis import FixedBasis

__all__ = [
    "rho_m_gen",
    "rho_2_block_gen",
    "rho_2_kkbar_gen",
    "rho_m",
]


# ---------------------------------------------------------------------
# generic ρ(m) tensor
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
            mat = np.real(
                of.get_sparse_operator(op, n_qubits=basis.d)
            )[np.ix_(basis.num_ele, basis.num_ele)]

            rows, cols = mat.nonzero()
            indices.extend([[ii, jj, r, c] for r, c in zip(rows, cols)])
            values.extend(mat.data)

    return indices, values


# .....................................................................
def rho_m_gen(
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
    m_basis = FixedBasis(basis.d, num=m, pairs=basis.pairs)
    shape = (m_basis.size, m_basis.size, basis.size, basis.size)

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

        c†_k  c†_{\bar l}  c_{\bar k}  c_l

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

            mat = np.real(
                of.get_sparse_operator(op1 * op2, n_qubits=basis.d)
            )[np.ix_(basis.num_ele, basis.num_ele)]

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([idx1, idx2, r, c])
                values.append(v)

    return indices, values


# .....................................................................
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

            mat = np.real(
                of.get_sparse_operator(op1 * op2, n_qubits=basis.d)
            )[np.ix_(basis.num_ele, basis.num_ele)]

            rows, cols = mat.nonzero()
            for r, c, v in zip(rows, cols, mat.data):
                indices.append([ii, jj, r, c])
                values.append(v)

    return indices, values


# .....................................................................
def rho_2_kkbar_gen(basis: FixedBasis, *, n_workers: int | None = None) -> sparse.COO:
    """
    Generate the diagonal *k \\bar k* block

        ρ₂[k, j] = ⟨c†_k  c†_{\bar k}  c_{\bar j}  c_j⟩

    with shape ``(m, m, N_dim, N_dim)`` where ``m = d // 2``.
    """
    m_pairs = basis.d // 2
    it_set = np.arange(m_pairs)
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
