from __future__ import annotations

from __future__ import annotations
import numpy as np
from scipy import sparse
import openfermion as of
from functools import lru_cache
from itertools import combinations

# ---------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def of_sector_bitmasks(d: int, n: int) -> np.ndarray:
    """
    True OpenFermion fixed-N basis order via introspection.
    Returns the array of bitmasks in the exact column/row order used by
    openfermion.linalg.get_number_preserving_sparse_operator.
    """
    if not (0 <= n <= d):
        return np.array([], dtype=np.int64)

    # Weighted number operator: W = sum_i (2**i) * c_i^\dagger c_i
    W = of.FermionOperator()
    for i in range(d):
        W += (1 << i) * of.FermionOperator(((i, 1), (i, 0)))

    mat_W = of.linalg.get_number_preserving_sparse_operator(
        W, num_qubits=d, num_electrons=n, spin_preserving=False
    ).tocsc()

    # The diagonal entries are the bitmasks in OpenFermion's basis order.
    diag = np.asarray(mat_W.diagonal()).real.astype(np.int64)
    return diag

@lru_cache(maxsize=None)
def mask_to_index_map(d: int, n: int) -> dict[int, int]:
    """Map bitmask -> position in OpenFermion's N-particle basis order."""
    masks = of_sector_bitmasks(d, n)
    return {int(m): int(i) for i, m in enumerate(masks)}

def restrict_sector_matrix(mat: sparse.spmatrix,
                           subset_masks: np.ndarray,
                           d: int,
                           n: int):
    """
    Reorder/restrict an OpenFermion N-particle matrix 'mat'
    to rows/cols corresponding to 'subset_masks' (bitmasks),
    preserving OpenFermion's index order.
    """
    mapping = mask_to_index_map(d, n)
    idx = [mapping[int(m)] for m in subset_masks]
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
    Return the sparse matrix of op in the |N=n〉 sector.
    """
    mat_comb = of.linalg.get_number_preserving_sparse_operator(
        op,
        num_qubits=d,
        num_electrons=n,
        spin_preserving=spin_preserving,
    )
    return mat_comb.tocsc()