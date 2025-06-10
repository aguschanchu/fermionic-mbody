from __future__ import annotations
from math import cos, sin
from typing import Tuple

import numpy as np
import openfermion as of
from openfermion.chem import MolecularData
import openfermionpyscf                       # registers PySCF plugin
from scipy.sparse import csr_matrix


def generate(r: float = 3.5) -> Tuple[csr_matrix, "pyscf.gto.Mole", MolecularData]:
    """
    Return (sparse Hamiltonian, pyscf.mol, MolecularData) for H₂O at bond length *r* (Bohr).
    STO-3G basis, restricted Hartree–Fock.
    """
    theta = np.deg2rad(104.5)
    x, z = r * sin(theta / 2), r * cos(theta / 2)

    geom = [("O", (0.0, 0.0, 0.0)),
            ("H", ( x, 0.0,  z)),
            ("H", (-x, 0.0,  z))]

    mol_data = MolecularData(geom, basis="sto-3g", multiplicity=1, charge=0)

    mol_data = openfermionpyscf.run_pyscf(mol_data, run_scf=True)
    h2_mol   = mol_data.get_molecular_hamiltonian()

    jw_ham   = of.jordan_wigner(h2_mol)
    sparse_h = of.get_sparse_operator(jw_ham).tocsr()
    
    try:
        pyscf_mol = mol_data.molecule                 # old OpenFermion
    except AttributeError:
        pyscf_mol = mol_data._pyscf_data["mol"]       # OpenFermion-PySCF 0.5+

    return sparse_h, pyscf_mol, mol_data
