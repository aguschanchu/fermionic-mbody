from __future__ import annotations
import itertools
from typing import Iterator, List, Sequence, Tuple

import numpy as np
from scipy import sparse
import openfermion as of
from functools import lru_cache
from itertools import combinations

# ---------------------------------------------------------------------
#  private routines copied from OF
# ---------------------------------------------------------------------
def _iterate_basis_order_(reference_determinant: Sequence[bool],
                          order: int) -> Iterator[np.ndarray]:
    occupied_indices = np.where(reference_determinant)[0]
    unoccupied_indices = np.where(~np.asarray(reference_determinant))[0]

    for occ_ind, unocc_ind in itertools.product(
            itertools.combinations(occupied_indices, order),
            itertools.combinations(unoccupied_indices, order)):
        basis_state = np.asarray(reference_determinant, dtype=bool).copy()
        basis_state[list(occ_ind)] = False
        basis_state[list(unocc_ind)] = True
        yield basis_state


def _iterate_basis_spin_order_(reference_determinant: Sequence[bool],
                               alpha_order: int,
                               beta_order: int) -> Iterator[np.ndarray]:
    reference_determinant = np.asarray(reference_determinant, dtype=bool)
    occupied_alpha_indices = np.where(reference_determinant[::2])[0] * 2
    unoccupied_alpha_indices = np.where(~reference_determinant[::2])[0] * 2
    occupied_beta_indices = np.where(reference_determinant[1::2])[0] * 2 + 1
    unoccupied_beta_indices = np.where(~reference_determinant[1::2])[0] * 2 + 1

    for (alpha_occ, alpha_unocc,
         beta_occ, beta_unocc) in itertools.product(
            itertools.combinations(occupied_alpha_indices, alpha_order),
            itertools.combinations(unoccupied_alpha_indices, alpha_order),
            itertools.combinations(occupied_beta_indices, beta_order),
            itertools.combinations(unoccupied_beta_indices, beta_order)):
        basis_state = reference_determinant.copy()
        basis_state[list(alpha_occ)] = False
        basis_state[list(alpha_unocc)] = True
        basis_state[list(beta_occ)] = False
        basis_state[list(beta_unocc)] = True
        yield basis_state


def _iterate_basis_(reference_determinant: Sequence[bool],
                    excitation_level: int,
                    spin_preserving: bool) -> Iterator[np.ndarray]:
    if not spin_preserving:
        for order in range(excitation_level + 1):
            yield from _iterate_basis_order_(reference_determinant, order)
    else:
        alpha_exc_lvl = min(np.sum(reference_determinant[::2]), excitation_level)
        beta_exc_lvl  = min(np.sum(reference_determinant[1::2]), excitation_level)

        for order in range(excitation_level + 1):
            for alpha_order in range(alpha_exc_lvl + 1):
                beta_order = order - alpha_order
                if beta_order < 0 or beta_order > beta_exc_lvl:
                    continue
                yield from _iterate_basis_spin_order_(reference_determinant,
                                                      alpha_order, beta_order)


def iterate_basis(reference: Sequence[bool],
                  exc_level: int,
                  spin_preserving: bool = False):
    return _iterate_basis_(reference, exc_level, spin_preserving)

# ---------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def mask_to_index_map(d: int, n: int) -> dict[int, int]:
    """
    Return a dict {bitmask:int  ->  position 0…C(d,n)-1} for the
    ascending-bitmask ordering used by FixedBasis/number_preserving_matrix.
    """
    masks = [sum(1 << i for i in comb)
             for comb in combinations(range(d), n)]
    return {m: i for i, m in enumerate(sorted(masks))}

def restrict_sector_matrix(mat: sparse.spmatrix,
                           subset_masks: np.ndarray,
                           d: int,
                           n: int):
    mapping = mask_to_index_map(d, n)
    idx = [mapping[m] for m in subset_masks]
    return mat[np.ix_(idx, idx)]

# ---------------------------------------------------------------------
#  the high-level helper used by rho.py
# ---------------------------------------------------------------------
def number_preserving_matrix(op: of.FermionOperator,
                             d: int,
                             n: int,
                             *,
                             spin_preserving: bool = False
                             ) -> sparse.csc_matrix:
    """
    Return the sparse matrix of op in the |N=n〉 sector
    using the ascending-bitmask ordering
    """
    mat = of.linalg.get_number_preserving_sparse_operator(
        op,
        num_qubits=d,
        num_electrons=n,
        spin_preserving=spin_preserving,
        reference_determinant=None,
        excitation_level=None,
    )

    ref_det = np.array([i < n for i in range(d)], dtype=bool)
    states  = np.asarray(
        list(_iterate_basis_(ref_det, n, spin_preserving)))
    bitmasks = states.dot(1 << np.arange(d)[::-1])
    order = np.argsort(bitmasks)

    return mat[order][:, order]
