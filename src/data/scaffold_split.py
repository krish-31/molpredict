"""
scaffold_split.py
─────────────────
Bemis-Murcko scaffold-based train / val / test split.

Strategy
--------
1. Compute the Murcko scaffold SMILES for every molecule.
2. Group molecules by scaffold.
3. Sort scaffold groups by size (largest → test; smallest → train).
   This maximises structural diversity in the test set.
4. Greedily assign scaffold groups to test → val → train buckets.

Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks"
           (OGB benchmark splits use this exact strategy).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _get_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES for a molecule. Falls back to full SMILES on error."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles   # treat invalid SMILES as its own scaffold
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold if scaffold else smiles
    except Exception:
        return smiles


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split a list of SMILES into train / val / test index sets using scaffold splits.

    Parameters
    ----------
    smiles_list : list of SMILES strings (length N)
    frac_train, frac_val, frac_test : fractions (must sum to 1.0)
    seed : random seed for tie-breaking

    Returns
    -------
    train_idx, val_idx, test_idx : lists of integer indices into smiles_list
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    n = len(smiles_list)

    # ── Step 1: build scaffold → molecule index mapping ───────────────────
    logger.info("Computing Murcko scaffolds for %d molecules…", n)
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx, smi in enumerate(smiles_list):
        scaffold = _get_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # ── Step 2: sort scaffold groups (largest group last → goes to test) ──
    scaffold_groups = sorted(
        scaffold_to_indices.values(),
        key=lambda grp: (len(grp), rng.random()),   # stable tie-break
    )

    # ── Step 3: fill buckets ──────────────────────────────────────────────
    train_cutoff = int(np.floor(frac_train * n))
    val_cutoff   = train_cutoff + int(np.floor(frac_val * n))

    train_idx: list[int] = []
    val_idx:   list[int] = []
    test_idx:  list[int] = []

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    logger.info(
        "Scaffold split complete → train: %d | val: %d | test: %d "
        "| unique scaffolds: %d",
        len(train_idx), len(val_idx), len(test_idx),
        len(scaffold_to_indices),
    )

    return train_idx, val_idx, test_idx
