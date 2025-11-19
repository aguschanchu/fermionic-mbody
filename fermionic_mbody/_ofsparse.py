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
    Return an array of bitmasks for the N-particle sector basis, 
    ordered lexicographically.
    
    This order matches the one used by OpenFermion's 
    get_number_preserving_sparse_operator (when spin_preserving=False).
    """
    if not (0 <= n <= d):
        return np.array([], dtype=np.int64)

    # In the LE convention, this naturally produces bitmasks in increasing numerical order.
    masks = []
    for indices in combinations(range(d), n):
        mask = 0
        for i in indices:
            mask |= (1 << i)
        masks.append(mask)
        
    return np.array(masks, dtype=np.int64)


@lru_cache(maxsize=None)
def mask_to_index_map(d: int, n: int) -> dict[int, int]:
    """
    Map bitmask -> position in OpenFermion's N-particle basis order
    """
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
    Return the sparse matrix of op in the |N=n〉 sector
    using the OpenFermion ordering (B_Asc)
    """
    # Get matrix in B_Comb order from OpenFermion
    mat_comb = of.linalg.get_number_preserving_sparse_operator(
        op,
        num_qubits=d,
        num_electrons=n,
        spin_preserving=spin_preserving,
        reference_determinant=None,
        excitation_level=None,
    )

    return mat_comb.tocsc()