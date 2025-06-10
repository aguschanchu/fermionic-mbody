"""
Fixed single–particle operator bases and fast ⟨ψ|Ô|φ⟩ helpers.

This module contains the :class:`FixedBasis` class, which is the backbone
for every ρ(m) tensor we later build: it keeps a *canonical* vector basis
|eₖ⟩, an aligned list of :class:`openfermion.FermionOperator` objects, and
utility methods to map between them quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

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

    The class builds *one* fixed set of second-quantised operators
    (c†_i or products thereof) together with a canonical vector basis
    |e_k⟩ so that you can

    * map between integer indices and FermionOperator objects,
    * compute ⟨ψ|Ô|φ⟩ matrix elements quickly,
    * generate reduced density-matrix blocks.

    Parameters
    ----------
    d
        Number of single-particle modes.
    num
        Restrict the basis to *m-body* operators (`num = m`).
        If *None*, return the full 0…d operator basis.
    pairs
        When *True*, only keep operators that act on *paired* levels
        (e.g. time-reversed orbitals k / \\bar k).

    Notes
    -----
    *If* you plan to use the specialised ``ρ₂`` helpers below
    (those assuming two-level “k / \\bar k” pairs) you most likely
    want ``d`` to be *even* and ``pairs=True``.
    """

    d: int
    num: Optional[int] = None
    pairs: bool = False

    # --- internals filled in __post_init__ --------------------------------
    base: List[of.FermionOperator] = field(init=False, repr=False)
    num_ele: np.ndarray = field(init=False, repr=False)
    size: int = field(init=False)
    canonicals: np.ndarray = field(init=False, repr=False)
    signs: np.ndarray = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def idx_to_repr(self, idx: int) -> np.ndarray:
        """Return the canonical |e_idx⟩ vector in ℂ^size."""
        return self.canonicals[idx]

    # .....................................................................
    def opr_to_idx(self, opr: of.FermionOperator) -> Optional[int]:
        """
        Return *idx* such that ``base[idx]`` equals `opr` once normal-ordered.

        Returns
        -------
        Optional[int]
            `None` if the operator is outside the current basis.
        """
        norm_op = of.transforms.normal_ordered(opr)
        if not norm_op.terms:  # empty operator
            return None

        occ_indices = tuple(i for i, _ in next(iter(norm_op.terms.keys())))
        bitmask = sum(1 << i for i in occ_indices)

        matches = np.where(self.num_ele == bitmask)[0]
        return int(matches[0]) if matches.size else None

    # .....................................................................
    def idx_mean_val(self, idx: int, op: of.FermionOperator) -> float:
        """Compute ⟨e_idx| op |e_idx⟩ in the canonical basis."""
        ket = self.idx_to_repr(idx)
        return float(np.real(ket.T @ of.get_sparse_operator(op, self.d) @ ket))

    # .....................................................................
    def idx_contraction(
        self, bra_idx: int, ket_idx: int, op: of.FermionOperator
    ) -> float:
        """Compute ⟨e_bra| op |e_ket⟩ given canonical indices."""
        bra, ket = map(self.idx_to_repr, (bra_idx, ket_idx))
        return float(np.real(bra.T @ of.get_sparse_operator(op, self.d) @ ket))

    # ---------------------------------------------------------------------
    # static shortcuts
    # ---------------------------------------------------------------------
    @staticmethod
    def int_to_bin(k: int, d: int) -> str:
        """Little-endian, zero-padded binary string of length `d`."""
        return np.base_repr(k, base=2).zfill(d)[::-1]

    # .....................................................................
    @staticmethod
    def bin_to_op(bits: str) -> of.FermionOperator:
        """Convert a bit-string → product of creation operators."""
        idxs = [(i, 1) for i, b in enumerate(bits) if b == "1"]
        return of.FermionOperator(idxs)

    # ---------------------------------------------------------------------
    # internal machinery
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401  (simple verb ok)
        """Populate internal tables after dataclass creation."""
        self.base, self.num_ele = self._create_basis()
        self.size = len(self.base)
        self.canonicals = np.eye(self.size)
        self.signs = self._signs_gen()

    # ................................................................. internal helpers
    def _create_basis(self) -> Tuple[List[of.FermionOperator], np.ndarray]:
        """Generate basis list + bit-mask array."""
        basis, masks = [], []
        for k in range(1 << self.d):  # noqa: PLR1703 (d is small anyway)
            bits = self.int_to_bin(k, self.d)

            # particle-number restriction?
            if self.num is not None and bits.count("1") != self.num:
                continue

            # k / \bar k pairing restriction?
            if self.pairs and not np.all(bits[::2] == bits[1::2]):
                continue

            basis.append(self.bin_to_op(bits))
            masks.append(k)

        return basis, np.fromiter(masks, dtype=int)

    # .....................................................................
    def _signs_gen(self) -> np.ndarray:
        """Return sign( leading coefficient ) for each operator in `base`."""
        signs = []
        for op in self.base:
            op_n = of.transforms.normal_ordered(op)
            coeff = next(iter(op_n.terms.values())) if op_n.terms else 0
            signs.append(np.sign(coeff))
        return np.asarray(signs)

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
