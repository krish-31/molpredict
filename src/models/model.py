"""
model.py
────────
Full multi-task GIN model: backbone + task heads assembled.

                SMILES
                  │
           MoleculeFeaturizer
                  │
            PyG Data object
                  │
           GINBackbone (5 layers)
                  │
         graph_emb [B, 300]
                  │
      ┌───────────┴───────────┐
  Head_0  Head_1  …  Head_11   (12 TaskAttentionHeads)
      │
  12 logits [B, 1] each
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.models.gin import GINBackbone
from src.models.task_heads import TaskHeadCollection
from src.data.featurizer import MoleculeFeaturizer

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class MTGRLModel(nn.Module):
    """
    Multi-Task Graph Representation Learning Model.

    Parameters
    ----------
    atom_dim, edge_dim : feature dimensions (85, 12 for Tox21)
    hidden_dim         : GIN + head embedding width (300)
    num_layers         : GIN message-passing rounds (5)
    dropout            : dropout probability (0.5)
    pooling            : 'sum' | 'mean' | 'max'
    num_tasks          : number of prediction heads (12)
    task_names         : list of task name strings (for inference output dicts)
    """

    def __init__(
        self,
        atom_dim: int = 85,
        edge_dim: int = 12,
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.5,
        pooling: str = "sum",
        num_tasks: int = 12,
        task_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.task_names = task_names or TOX21_TASKS
        self.num_tasks  = num_tasks

        self.backbone = GINBackbone(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )

        self.heads = TaskHeadCollection(
            num_tasks=num_tasks,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Lazy featurizer — created on first call to predict()
        self._featurizer: Optional[MoleculeFeaturizer] = None

    # ── Core forward pass ─────────────────────────────────────────────────────
    def forward(self, batch) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        batch : PyG Batch (from DataLoader)

        Returns
        -------
        logits : list of num_tasks tensors, each [B, 1]
        """
        graph_emb = self.backbone(batch)     # [B, H]
        return self.heads(graph_emb)         # list of [B, 1]

    # ── Convenience inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self, smiles: str, device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, float]:
        """
        Single-molecule inference.

        Parameters
        ----------
        smiles  : SMILES string
        device  : 'cpu' | 'cuda'

        Returns
        -------
        dict mapping task_name → probability (float, 0–1)
        """
        if self._featurizer is None:
            self._featurizer = MoleculeFeaturizer()

        data = self._featurizer.smiles_to_graph(smiles)
        if data is None:
            return {t: float("nan") for t in self.task_names}

        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data]).to(device)
        self.eval()
        self.to(device)

        logits = self.forward(batch)           # list of [1, 1]
        probs  = [torch.sigmoid(l).item() for l in logits]

        return {name: round(p, 6) for name, p in zip(self.task_names, probs)}

    @torch.no_grad()
    def predict_batch(
        self, smiles_list: List[str], device: Union[str, torch.device] = "cpu"
    ) -> List[Dict[str, float]]:
        """
        Batch inference over a list of SMILES.

        Returns a list of dicts (one per molecule).
        Invalid SMILES → all NaN.
        """
        if self._featurizer is None:
            self._featurizer = MoleculeFeaturizer()

        from torch_geometric.data import Batch

        graphs, valid_mask = [], []
        for smi in smiles_list:
            g = self._featurizer.smiles_to_graph(smi)
            graphs.append(g)
            valid_mask.append(g is not None)

        # Build batch from valid graphs only
        valid_graphs = [g for g in graphs if g is not None]
        if not valid_graphs:
            nan_dict = {t: float("nan") for t in self.task_names}
            return [nan_dict] * len(smiles_list)

        batch = Batch.from_data_list(valid_graphs).to(device)
        self.eval()
        self.to(device)

        logits = self.forward(batch)                        # list of [V, 1]
        probs  = torch.cat([torch.sigmoid(l) for l in logits], dim=1)  # [V, 12]

        results = []
        valid_iter = iter(probs.cpu().tolist())
        for is_valid in valid_mask:
            if is_valid:
                row = next(valid_iter)
                results.append({n: round(p, 6) for n, p in zip(self.task_names, row)})
            else:
                results.append({t: float("nan") for t in self.task_names})

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = self.backbone.count_parameters()
        heads    = self.heads.count_parameters()
        print(f"Total params : {total:,}")
        print(f"  Backbone   : {backbone:,}")
        print(f"  Task heads : {heads:,}")
        return total

    @classmethod
    def from_config(cls, cfg: dict) -> "MTGRLModel":
        """Construct model from a YAML-loaded config dict."""
        return cls(
            atom_dim   = cfg.get("atom_dim", 85),
            edge_dim   = cfg.get("edge_dim", 12),
            hidden_dim = cfg.get("hidden_dim", 300),
            num_layers = cfg.get("gin_layers", 5),
            dropout    = cfg.get("dropout", 0.5),
            pooling    = cfg.get("pooling", "sum"),
            num_tasks  = cfg.get("num_tasks", 12),
            task_names = cfg.get("task_names", TOX21_TASKS),
        )

    @classmethod
    def load_checkpoint(cls, path: str, cfg: dict, device: str = "cpu") -> "MTGRLModel":
        """Load a saved model.pt checkpoint."""
        model = cls.from_config(cfg)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
