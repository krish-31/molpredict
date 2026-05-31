"""
task_heads.py
─────────────
One attention-based prediction head per task.

Architecture (per head)
──────────────────────
  graph_emb  [B, H]
       │
  Attention gate:
    Linear(H→H) → Tanh → Linear(H→1) → Sigmoid  ←  scalar α ∈ (0,1)
       │
  Weighted embedding:  h' = α · graph_emb
       │
  Predictor MLP:
    Linear(H → H/2) → ReLU → Dropout → Linear(H/2 → 1)
       │
  logit  [B, 1]   (sigmoid applied externally by loss / inference)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAttentionHead(nn.Module):
    """
    Single task-specific head with a soft-attention gating mechanism.

    The attention gate learns *which dimensions* of the shared embedding
    are relevant for this particular task, allowing the model to ignore
    irrelevant features rather than back-propagating gradients that
    would corrupt the shared backbone.
    """

    def __init__(self, hidden_dim: int = 300, dropout: float = 0.5):
        super().__init__()

        # Attention gate
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Predictor MLP
        mid = hidden_dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        graph_emb : [B, hidden_dim]

        Returns
        -------
        logit : [B, 1]   (raw — no sigmoid here)
        """
        alpha = self.attn_gate(graph_emb)          # [B, 1]
        weighted = alpha * graph_emb               # [B, H]
        return self.predictor(weighted)            # [B, 1]


class TaskHeadCollection(nn.Module):
    """
    Container for num_tasks TaskAttentionHead modules.

    Usage
    -----
    >>> heads = TaskHeadCollection(num_tasks=12, hidden_dim=300)
    >>> logits = heads(graph_emb)   # list of 12 tensors, each [B, 1]
    """

    def __init__(self, num_tasks: int = 12, hidden_dim: int = 300, dropout: float = 0.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.heads = nn.ModuleList([
            TaskAttentionHead(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_tasks)
        ])

    def forward(self, graph_emb: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of num_tasks logit tensors, each [B, 1]."""
        return [head(graph_emb) for head in self.heads]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
