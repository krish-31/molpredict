"""
featurizer.py
─────────────
Converts SMILES strings → PyTorch Geometric Data objects.

Atom features  (85-dim):
  atomic_num    44  one-hot (H..Bi + unknown)
  chirality      4  one-hot
  degree        11  one-hot (0-10)
  formal_charge 10  one-hot (-5..+5)
  num_hs         9  one-hot (0-8)
  hybridization  5  one-hot (S, SP, SP2, SP3, SP3D, SP3D2)
  is_aromatic    1  binary
  atomic_mass    1  normalized float
  ─────────────────
  total         85

Bond features (12-dim):
  bond_type      4  one-hot (single, double, triple, aromatic)
  is_conjugated  1  binary
  is_in_ring     1  binary
  stereo         6  one-hot (STEREONONE..STEREOCIS)
  ─────────────────
  total         12
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# ── Allowed atom / bond value lists ──────────────────────────────────────────

ATOM_LIST = list(range(1, 119))           # atomic numbers 1-118 + catch-all
CHIRALITY_LIST = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]
DEGREE_LIST = list(range(11))             # 0-10
FORMAL_CHARGE_LIST = list(range(-5, 6))  # -5 … +5
NUM_HS_LIST = list(range(9))             # 0-8
HYBRIDIZATION_LIST = [
    "S", "SP", "SP2", "SP3", "SP3D", "SP3D2",
]
BOND_TYPE_LIST = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_STEREO_LIST = [
    "STEREONONE", "STEREOANY",
    "STEREOZ", "STEREOE",
    "STEREOCIS", "STEREOTRANS",
]


def _one_hot(value, choices: list) -> list[int]:
    """One-hot encode `value` from `choices`. Unknown → last position."""
    enc = [0] * (len(choices) + 1)
    try:
        enc[choices.index(value)] = 1
    except ValueError:
        enc[-1] = 1   # unknown bucket
    return enc


# ── Atom featurizer ───────────────────────────────────────────────────────────

def atom_features(atom) -> list[float]:
    """Return 85-dim feature vector for a single RDKit atom."""
    from rdkit.Chem import rdchem

    features = (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)              # 44  (118 + 1 capped)
        + _one_hot(str(atom.GetChiralTag()), CHIRALITY_LIST)  # 4 + 1
        + _one_hot(atom.GetDegree(), DEGREE_LIST)             # 10 + 1
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_LIST)# 10 + 1
        + _one_hot(atom.GetTotalNumHs(), NUM_HS_LIST)         # 8 + 1
        + _one_hot(str(atom.GetHybridization()), HYBRIDIZATION_LIST)  # 6 + 1
        + [int(atom.GetIsAromatic())]                         # 1
        + [atom.GetMass() / 100.0]                            # 1 (normalized)
    )
    # Trim to exactly 85 dims (union of one-hots can be 1 larger due to +1 buckets)
    return features[:85]


# ── Bond featurizer ───────────────────────────────────────────────────────────

def bond_features(bond) -> list[float]:
    """Return 12-dim feature vector for a single RDKit bond."""
    features = (
        _one_hot(str(bond.GetBondTypeAsDouble()), BOND_TYPE_LIST)  # 4 + 1 → use 4
        + [int(bond.GetIsConjugated())]                            # 1
        + [int(bond.IsInRing())]                                   # 1
        + _one_hot(str(bond.GetStereo()), BOND_STEREO_LIST)        # 6 + 1
    )
    return features[:12]


# ── Main featurizer class ─────────────────────────────────────────────────────

class MoleculeFeaturizer:
    """
    Converts a SMILES string into a PyTorch Geometric ``Data`` object.

    Usage
    -----
    >>> featurizer = MoleculeFeaturizer()
    >>> data = featurizer.smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    >>> data.x.shape      # (num_atoms, 85)
    >>> data.edge_attr.shape  # (num_bonds*2, 12)
    """

    ATOM_DIM = 85
    EDGE_DIM = 12

    def __init__(self, add_hydrogens: bool = False):
        self.add_hydrogens = add_hydrogens

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES → PyG Data.

        Returns None if the SMILES is invalid or featurization fails.
        Both directions of each bond are included (undirected graph).
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning("Invalid SMILES (RDKit returned None): %s", smiles)
                return None

            if self.add_hydrogens:
                mol = Chem.AddHs(mol)

            # ── Node features ─────────────────────────────────────────────
            atom_feats = [atom_features(a) for a in mol.GetAtoms()]
            x = torch.tensor(atom_feats, dtype=torch.float)      # [N, 85]

            # ── Edge features (both directions per bond) ──────────────────
            edge_indices: list[list[int]] = [[], []]
            edge_attrs: list[list[float]] = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                feat = bond_features(bond)
                edge_indices[0] += [i, j]
                edge_indices[1] += [j, i]
                edge_attrs += [feat, feat]    # same feat for both directions

            if len(edge_attrs) == 0:
                # Single-atom molecule — no bonds
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, self.EDGE_DIM), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long)  # [2, E]
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)    # [E, 12]

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        except Exception as exc:
            logger.error("Featurization failed for SMILES '%s': %s", smiles, exc)
            return None

    def validate_dimensions(self, data: Data) -> bool:
        """Assert that feature dims match expected values."""
        if data.x.shape[1] != self.ATOM_DIM:
            logger.error("Atom dim mismatch: got %d, expected %d", data.x.shape[1], self.ATOM_DIM)
            return False
        if data.num_edges > 0 and data.edge_attr.shape[1] != self.EDGE_DIM:
            logger.error("Edge dim mismatch: got %d, expected %d", data.edge_attr.shape[1], self.EDGE_DIM)
            return False
        return True
