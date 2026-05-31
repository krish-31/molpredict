"""
losses.py
─────────
Two loss components:

1. MultiTaskBCELoss
   ─────────────────
   Per-task binary cross-entropy that:
   • handles class imbalance via pos_weight (auto-computed from train labels)
   • masks out missing labels (y == -1) before computing loss

2. LearnedUncertaintyLoss  (Kendall et al., 2018)
   ──────────────────────────────────────────────
   L_total = Σ_i  [ exp(-log_σ_i) · L_i  +  log_σ_i ]
                      ^^^^^^^^^^^             ^^^^^^^^
                      1/σ² weight        regularisation

   With σ_i = exp(log_σ_i / 2)  this is equivalent to:
       L_total = Σ_i  [ (1/σ_i²) · L_i  +  log(σ_i) ]

   log_σ_i is a learnable parameter initialised at 0 (σ_i = 1, equal weights).
   Tasks the model is uncertain about get σ_i > 1  →  lower effective weight.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskBCELoss(nn.Module):
    """
    Masked binary cross-entropy for multi-task learning.

    Parameters
    ----------
    pos_weights : list of per-task positive class weights [num_tasks]
                  Pass None to use equal weights (1.0 each).
    num_tasks   : number of tasks (12 for Tox21)

    Forward
    -------
    logits  : list of [B, 1] tensors
    labels  : [B, num_tasks] long tensor (0 / 1 / -1=missing)

    Returns
    -------
    task_losses : [num_tasks] float tensor — mean BCE per task
                  (tasks with no valid labels → 0.0)
    """

    def __init__(
        self,
        num_tasks: int = 12,
        pos_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        if pos_weights is not None:
            self.register_buffer(
                "pos_weights", torch.tensor(pos_weights, dtype=torch.float)
            )
        else:
            self.register_buffer(
                "pos_weights", torch.ones(num_tasks)
            )

    def forward(
        self,
        logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : list of num_tasks tensors, each [B, 1]
        labels : [B, num_tasks]  long  (0/1/-1)

        Returns
        -------
        task_losses : [num_tasks] float tensor
        """
        task_losses = []

        for i, logit in enumerate(logits):
            y = labels[:, i].float()           # [B]
            mask = y != -1                     # valid labels mask

            if mask.sum() == 0:
                task_losses.append(logit.new_zeros(1).squeeze())
                continue

            logit_i = logit.squeeze(-1)[mask]  # [M]
            y_i     = y[mask]                  # [M]
            pw      = self.pos_weights[i].to(logit_i.device)

            loss_i = F.binary_cross_entropy_with_logits(
                logit_i, y_i,
                pos_weight=pw,
                reduction="mean",
            )
            task_losses.append(loss_i)

        return torch.stack(task_losses)   # [num_tasks]


class LearnedUncertaintyLoss(nn.Module):
    """
    Kendall et al. (2018) multi-task loss with learnable per-task uncertainty.

    L_total = Σ_i  [ exp(-log_σ_i) · L_i  +  log_σ_i ]

    Parameters
    ----------
    num_tasks : number of tasks

    Forward
    -------
    task_losses : [num_tasks] tensor of per-task BCE losses

    Returns
    -------
    total_loss : scalar
    """

    def __init__(self, num_tasks: int = 12):
        super().__init__()
        # log_sigma initialised at 0 → σ = 1 → equal weights at start
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        task_losses : [num_tasks] per-task BCE losses

        Returns
        -------
        total_loss : scalar
        """
        # weight = exp(-log_σ) = 1/σ
        precision = torch.exp(-self.log_sigma)
        total = (precision * task_losses + self.log_sigma).sum()
        return total

    def get_weights(self) -> Dict[str, float]:
        """Return current per-task uncertainty weights as a dict."""
        weights = torch.exp(-self.log_sigma).detach().cpu()
        return {f"task_{i}": round(w.item(), 6) for i, w in enumerate(weights)}

    def get_sigma(self) -> Dict[str, float]:
        """Return current σ_i values (uncertainty) per task."""
        sigmas = torch.exp(self.log_sigma / 2).detach().cpu()
        return {f"sigma_{i}": round(s.item(), 6) for i, s in enumerate(sigmas)}
