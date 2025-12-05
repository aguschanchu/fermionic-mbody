"""
Fixed single-particle operator bases and ⟨ψ|Ô|φ⟩ helpers.

This module contains the FixedBasis class, which is the backbone
for every ρ(m) tensor we later build: it keeps a canonical vector basis
|eₖ⟩, an aligned list of openfermion.FermionOperator objects, and
utility methods to map between them quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Dict
from scipy import sparse as sp_sparse
import itertools
from ._ofsparse import number_preserving_matrix, restrict_sector_matrix, of_sector_bitmasks, mask_to_index_map

import numpy as np
import openfermion as of

__all__ = ["FixedBasis"]


# ---------------------------------------------------------------------
# FixedBasis
# ---------------------------------------------------------------------
@dataclass(slots=True)
class FixedBasis:
    """
    Single-particle ladder-operator basis for fermionic many-body work.

    The class builds one fixed set of second-quantised operators
    (c†_i or products thereof) together with a canonical vector basis
    |e_k⟩ so that you can

    - map between integer indices and FermionOperator objects,
    - compute ⟨ψ|Ô|φ⟩ matrix elements quickly,
    - generate reduced density-matrix blocks.

    Parameters
    ----------
    d
        Number of single-particle modes.
    num
        Restrict the basis to m-body operators (num = m).
        If None, return the full 0…d operator basis.
    pairs
        When True, only keep operators that act on paired levels
        (e.g. time-reversed orbitals k / \\bar k).
        
    """

    d: int
    num: Optional[int] = None
    pairs: bool = False

    # --- internals filled in __post_init__ --------------------------------
    base: List[of.FermionOperator] = field(init=False, repr=False)
    bitmasks: np.ndarray = field(init=False, repr=False)
    size: int = field(init=False)
    _mask2idx_cache: Dict[int, int] = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def idx_to_repr(self, idx: int, dtype=np.float64) -> np.ndarray:
        """Return the canonical |e_idx⟩ vector in C^size."""
        if not (0 <= idx < self.size):
            raise IndexError(f"Index {idx} out of bounds for basis size {self.size}.")
        vec = np.zeros(self.size, dtype=dtype)
        vec[idx] = 1.0
        return vec

    @property
    def num_ele(self) -> np.ndarray:
        return self.bitmasks

    # .....................................................................
    def opr_to_idx(self, opr: of.FermionOperator) -> Optional[int]:
            """
            Return idx such that base[idx] == opr (ignoring global phase).
            """
            norm_op = of.transforms.normal_ordered(opr)
            
            if len(norm_op.terms) != 1:
                return None
                
            term, coeff = next(iter(norm_op.terms.items()))

            if any(cd != 1 for _, cd in term):
                return None

            if not np.isclose(abs(coeff), 1.0):
                return None

            bitmask = sum(1 << i for i, _ in term)
            return self._mask2idx_cache.get(bitmask)

    def get_operator_matrix(self, op: of.FermionOperator) -> sp_sparse.spmatrix:
        """
        Return the sparse matrix representation O[i, j] = <e_i| op |e_j>
        in this fixed basis.
        """
        if self.num is not None:
            mat_N = number_preserving_matrix(op, self.d, self.num)
            # Fast path: if our bitmask order matches OF exactly, return as-is.
            of_masks = of_sector_bitmasks(self.d, self.num)
            if np.array_equal(self.bitmasks, of_masks):
                return mat_N
            # Otherwise, restrict/reorder to our subset order.
            return restrict_sector_matrix(mat_N, self.bitmasks, self.d, self.num)

        # Full Fock space
        mat_fock = of.get_sparse_operator(op, self.d)
        if self.size == (1 << self.d):
            return mat_fock
        indices = self.bitmasks
        return mat_fock[np.ix_(indices, indices)]

    # Update return types to complex and use get_operator_matrix
    def idx_mean_val(self, idx: int, op: of.FermionOperator) -> complex:
        """Compute ⟨e_idx| op |e_idx⟩ in the canonical basis."""
        mat = self.get_operator_matrix(op)
        return complex(mat[idx, idx])

    def idx_contraction(
        self, bra_idx: int, ket_idx: int, op: of.FermionOperator) -> complex:
        """Compute ⟨e_bra| op |e_ket⟩ given canonical indices."""
        mat = self.get_operator_matrix(op)
        return complex(mat[bra_idx, ket_idx])

    # .....................................................................
    def opr_to_vect(self,
                    opr: of.FermionOperator,
                    *,
                    dtype=np.complex128,
                    tol: float = 0.0,
                    allow_missing: bool = False) -> np.ndarray:
            """
            Expand a (number-conserving) FermionOperator in this basis.

            Parameters
            ----------
            opr : FermionOperator
                The operator to be expanded.  It must be a sum of pure
                creation strings whose particle number equals
                self.num.
            dtype : numpy dtype, optional
                Type of the returned vector (default complex128).
            tol : float, optional
                Terms with |coeff| < tol are ignored.
            allow_missing : bool, optional
                If True the routine silently skips strings that are not
                present in the basis; otherwise it raises a ValueError.

            Returns
            -------
            vec : ndarray, shape (|basis|,)
                Coefficient vector such that
                    Σ_k vec[k] · basis.base[k]  ==  opr
                after normal ordering.
            """
            vec = np.zeros(self.size, dtype=dtype)

            norm_op = of.transforms.normal_ordered(opr)

            for term, coeff in norm_op.terms.items():

                if abs(coeff) <= tol:
                    continue                                    

                # ensure creation only  (cd = 1)  otherwise not in this basis
                if any(cd != 1 for _, cd in term):
                    raise ValueError(
                        f"term {term} contains annihilation operator; "
                        "basis is creation strings only")

                # build bit mask
                mask = 0
                for i, _ in term:
                    mask |= 1 << i

                idx = self._mask2idx_cache.get(mask)
                if idx is None:
                    if allow_missing:
                        continue
                    raise ValueError(f"term {term} not contained in the basis")

                vec[idx] += coeff

            return vec

    def vec_to_op(self, vec: np.ndarray, tol: float = 0.0) -> of.FermionOperator:
        op = of.FermionOperator()
        for idx, coeff in enumerate(vec):
            if abs(coeff) > tol:
                op += complex(coeff) * self.base[idx]
        return op

    # ---------------------------------------------------------------------
    # static shortcuts
    # ---------------------------------------------------------------------
    @staticmethod
    def int_to_bin(k: int, d: int) -> str:
        """Little-endian, zero-padded binary string of length d."""
        return np.base_repr(k, base=2).zfill(d)[::-1]

    # .....................................................................
    @staticmethod
    def bin_to_op(bits: str) -> of.FermionOperator:
        """Convert a bit-string → product of creation operators."""
        idxs = [(i, 1) for i, b in enumerate(bits) if b == "1"]
        return of.FermionOperator(idxs)

    @property
    def m(self) -> Optional[int]:
        """Alias for num (kept for backward-compatibility)."""
        return self.num
    
    @staticmethod
    def _det2op(det: int, n_qubits: int) -> of.FermionOperator:
        """
        Convert an integer det into the corresponding creation-operator string 
        |vac⟩ → |det⟩ using the Jordan-Wigner transformation (Little-Endian).
        
        The resulting operator creates the state with indices ordered increasingly:
        c†_{i_1} c†_{i_2} ... |vac⟩ (i_1 < i_2 < ...).
        """
        terms = []
        for q in range(n_qubits):
            if (det >> q) & 1:
                terms.append((q, 1))
        
        return of.FermionOperator(tuple(terms))
    
    # ---------------------------------------------------------------------
    # internal machinery
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:  
        if self.pairs and self.d % 2 != 0:
            raise ValueError("Pairing restriction (pairs=True) requires an even number of modes (d).")
            
        self.base, self.bitmasks = self._create_basis()
        self.size = len(self.base)
        self._mask2idx_cache = {int(mask): idx for idx, mask in enumerate(self.bitmasks)}

    def _create_basis(self) -> Tuple[List[of.FermionOperator], np.ndarray]:
        """
        Generate basis list + bit-mask array.
        """
        masks_list: List[int] = []

        # 1. Generate Raw Masks
        if self.num is not None:
            # Validate
            if self.num < 0 or self.num > self.d:
                return [], np.array([], dtype=np.int64)
            if self.pairs and (self.num % 2 != 0):
                return [], np.array([], dtype=np.int64)

            # Use optimized helper
            raw_masks = of_sector_bitmasks(self.d, self.num)
            
            if self.pairs:
                m_pairs = self.d // 2
                for m in raw_masks:
                    m = int(m)
                    # Check pairing: XOR bits 2p and 2p+1 must be 0
                    is_paired = True
                    for p in range(m_pairs):
                        if ((m >> (2*p)) & 1) != ((m >> (2*p+1)) & 1):
                            is_paired = False
                            break
                    if is_paired:
                        masks_list.append(m)
            else:
                masks_list = [int(m) for m in raw_masks]
        
        else:
            # Full Fock Space
            if self.pairs:
                m_pairs = self.d // 2
                for k_pair in range(1 << m_pairs):
                    k = 0
                    for i in range(m_pairs):
                        if (k_pair >> i) & 1:
                            k |= (3 << (2 * i))
                    masks_list.append(k)
            else:
                masks_list = list(range(1 << self.d))

        masks_arr = np.array(masks_list, dtype=np.int64)
        
        basis = [self._det2op(k, self.d) for k in masks_list]
        return basis, masks_arr

    # -----------------------------------------------------------------
    # alternate constructor: build a Basis from a subset of another one
    # -----------------------------------------------------------------
    @classmethod
    def from_subset(
        cls,
        parent: "FixedBasis",
        indices: Sequence[int], 
    ) -> "FixedBasis":
        """
        Build a new FixedBasis that keeps only indices from parent.

        Parameters
        ----------
        parent
            The full FixedBasis you want to down-select.
        indices
            1-D sequence (list, tuple, ndarray) of integer indices to keep.

        Notes
        -----
        - All cache arrays (size, canonicals, signs) are rebuilt
          consistently.
        - The new instance inherits d, num and pairs from
          parent; you can change them afterwards if you really need to.
        """
        new = cls.__new__(cls)
        
        new.d, new.num, new.pairs = parent.d, parent.num, parent.pairs
        
        indices_arr = np.asarray(indices)
        
        new.base     = [parent.base[i] for i in indices]
        new.bitmasks = parent.bitmasks[indices_arr]
        new.size     = len(new.base)
        
        # Rebuild caches
        new._mask2idx_cache = {int(mask): idx for idx, mask in enumerate(new.bitmasks)}
        
        return new

    # ---------------------------------------------------------------------
    # convenience serialization hooks
    # ---------------------------------------------------------------------
    def __getstate__(self) -> dict:
        """
        Return a small dict that is safe to send to Ray workers.
        """
        # Save using the new name 'bitmasks'
        return {"d": self.d, "num": self.num, "pairs": self.pairs, 'bitmasks': self.bitmasks}

    def __setstate__(self, state: dict) -> None:
        """Rebuild heavy caches locally after unpickling."""
        self.d     = state["d"]
        self.num   = state["num"]
        self.pairs = state["pairs"]

        # Handle backward compatibility: Load from 'bitmasks' or 'num_ele'
        if 'bitmasks' in state:
             self.bitmasks = np.asarray(state["bitmasks"], dtype=np.int64)
        elif 'num_ele' in state:
             # Load from old serialization format
             self.bitmasks = np.asarray(state["num_ele"], dtype=np.int64)
        else:
            raise ValueError("Missing 'bitmasks' or 'num_ele' in serialized state.")

        # rebuild every cache
        self.base   = [self._det2op(det, self.d) for det in self.bitmasks]
        self.size      = len(self.base)
        self._mask2idx_cache = {int(mask): idx for idx, mask in enumerate(self.bitmasks)}

    # ---------------------------------------------------------------------
    # convenience FermionOperator factories
    # ---------------------------------------------------------------------
    @staticmethod
    def cdc(i: int, j: int) -> of.FermionOperator:
        """c†_i c_j with normal-ordering preserved."""
        return of.FermionOperator(((i, 1), (j, 0)))

    @staticmethod
    def cc(i: int, j: int) -> of.FermionOperator:
        """c_i c_j (two annihilations)."""
        return of.FermionOperator(((i, 0), (j, 0)))
