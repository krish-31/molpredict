"""
dataset.py
──────────
PyTorch Geometric InMemoryDataset for Tox21.

Label convention
----------------
  Original CSV: 0 = inactive, 1 = active, '' / NaN = not measured
  Stored as   : 0 = inactive, 1 = active, -1 = missing (mask in loss)

Tox21 has 12 binary tasks (all NR-* and SR-* columns).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Callable

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.featurizer import MoleculeFeaturizer
from src.data.scaffold_split import scaffold_split

logger = logging.getLogger(__name__)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

TOX21_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
)


class Tox21Dataset(InMemoryDataset):
    """
    Tox21 molecular toxicity dataset as a PyG InMemoryDataset.

    Parameters
    ----------
    root : str
        Root directory where raw/ and processed/ subdirs will be created.
    split : str | None
        One of 'train', 'val', 'test', or None (returns all).
    transform, pre_transform : optional PyG transforms
    frac_train, frac_val, frac_test : scaffold split fractions
    seed : random seed for scaffold split tie-breaking

    Usage
    -----
    >>> dataset = Tox21Dataset(root='data', split='train')
    >>> len(dataset)
    >>> dataset[0].x.shape      # [num_atoms, 85]
    >>> dataset[0].y.shape      # [12]  — labels (0/1/-1)
    """

    def __init__(
        self,
        root: str = "data",
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42,
    ):
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.frac_test = frac_test
        self.seed = seed
        self._split = split

        super().__init__(root, transform, pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Apply split filter
        if split is not None:
            split_mask = self._load_split_mask(split)
            self.data, self.slices = self.collate(
                [self.get(i) for i in split_mask]
            )

    @property
    def raw_file_names(self) -> List[str]:
        return ["tox21.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["tox21_processed.pt", "tox21_splits.pt"]

    def download(self) -> None:
        """Download Tox21 CSV from S3."""
        import urllib.request, gzip, shutil

        gz_path = os.path.join(self.raw_dir, "tox21.csv.gz")
        csv_path = os.path.join(self.raw_dir, "tox21.csv")

        if os.path.exists(csv_path):
            logger.info("Raw CSV already exists, skipping download.")
            return

        logger.info("Downloading Tox21 dataset from %s …", TOX21_URL)
        urllib.request.urlretrieve(TOX21_URL, gz_path)

        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        logger.info("Tox21 CSV saved to %s", csv_path)

    def process(self) -> None:
        """Featurize molecules and compute scaffold split indices."""
        csv_path = os.path.join(self.raw_dir, "tox21.csv")
        df = pd.read_csv(csv_path)

        featurizer = MoleculeFeaturizer()
        data_list: List[Data] = []
        valid_smiles: List[str] = []
        skipped = 0

        logger.info("Featurizing %d molecules…", len(df))

        for _, row in df.iterrows():
            smiles = row["smiles"]
            graph = featurizer.smiles_to_graph(smiles)

            if graph is None:
                skipped += 1
                continue

            # Labels: -1 for missing, else 0/1
            labels = []
            for task in TOX21_TASKS:
                val = row.get(task, float("nan"))
                if pd.isna(val):
                    labels.append(-1)
                else:
                    labels.append(int(val))

            graph.y = torch.tensor(labels, dtype=torch.long)   # [12]
            graph.smiles = smiles

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)
            valid_smiles.append(smiles)

        logger.info(
            "Featurization done. Valid: %d | Skipped (invalid SMILES): %d",
            len(data_list), skipped,
        )

        # Save processed dataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Compute and save scaffold split indices
        train_idx, val_idx, test_idx = scaffold_split(
            valid_smiles,
            frac_train=self.frac_train,
            frac_val=self.frac_val,
            frac_test=self.frac_test,
            seed=self.seed,
        )
        torch.save(
            {"train": train_idx, "val": val_idx, "test": test_idx},
            self.processed_paths[1],
        )
        logger.info("Scaffold split indices saved.")

    def _load_split_mask(self, split: str) -> List[int]:
        splits = torch.load(self.processed_paths[1])
        if split not in splits:
            raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test")
        return splits[split]

    @property
    def num_tasks(self) -> int:
        return len(TOX21_TASKS)

    @property
    def task_names(self) -> List[str]:
        return TOX21_TASKS
