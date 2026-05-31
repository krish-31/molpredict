"""
pcgrad.py
─────────
PCGrad — Projecting Conflicting Gradients (Yu et al., 2020).

Algorithm (per step)
─────────────────────
For each task i:
  1. Compute gradient g_i  w.r.t. shared parameters
  2. For every other task j:
       if dot(g_i, g_j) < 0:                     # conflict detected
           g_i ← g_i − ( dot(g_i,g_j) / ||g_j||² ) · g_j   # project
  3. Sum projected gradients:  g_final = Σ_i  g_i_projected

This removes the component of g_i that conflicts with g_j,
while keeping the aligned component intact.

Reference: Yu, Tianhe, et al.
           "Gradient surgery for multi-task learning."
           NeurIPS 2020.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn


class PCGrad:
    """
    Wraps a standard PyTorch optimizer and replaces the backward pass
    with PCGrad's conflict-aware gradient projection.

    Usage
    -----
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> pc_optimizer = PCGrad(optimizer)
    >>>
    >>> # In training loop:
    >>> pc_optimizer.zero_grad()
    >>> task_losses = [loss_0, loss_1, ..., loss_11]
    >>> pc_optimizer.pc_backward(task_losses)
    >>> pc_optimizer.step()

    Attributes
    ----------
    conflict_rate : float
        Fraction of task pairs (i, j) with i≠j that had conflicting
        gradients in the most recent pc_backward() call. Useful for monitoring.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optim = optimizer
        self.conflict_rate: float = 0.0
        self._last_conflict_count: int = 0
        self._last_total_pairs: int = 0

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def pc_backward(self, losses: List[torch.Tensor]) -> None:
        """
        Compute PCGrad-projected gradients and write them into .grad attributes.

        Parameters
        ----------
        losses : list of per-task scalar losses (length = num_tasks)
                 Each must be a leaf in the computation graph.
        """
        # Get parameters that require gradients
        shared_params = [
            p for group in self._optim.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

        num_tasks = len(losses)

        # ── Step 1: compute per-task gradient vectors ──────────────────────
        task_grads: List[List[Optional[torch.Tensor]]] = []

        for i, loss in enumerate(losses):
            # Retain graph for all but the last task
            retain = (i < num_tasks - 1)
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=retain,
                create_graph=False,
                allow_unused=True,
            )
            # Replace None with zero tensor of matching shape
            task_grads.append([
                g.clone() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, shared_params)
            ])

        # ── Step 2: project conflicting gradients ─────────────────────────
        projected = self._project(task_grads)

        # ── Step 3: sum projected gradients and write to .grad ────────────
        for pi, param in enumerate(shared_params):
            merged = sum(grads[pi] for grads in projected)
            if param.grad is None:
                param.grad = merged
            else:
                param.grad.copy_(merged)

    def _project(
        self,
        task_grads: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Return a new list of projected gradient vectors."""
        num_tasks = len(task_grads)
        projected = [list(g) for g in task_grads]   # deep copy per param

        conflict_count = 0
        total_pairs    = 0

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                total_pairs += 1

                # Flatten both grad vectors for dot-product computation
                gi_flat = torch.cat([g.view(-1) for g in projected[i]])
                gj_flat = torch.cat([g.view(-1) for g in task_grads[j]])

                dot = torch.dot(gi_flat, gj_flat)

                if dot < 0:
                    conflict_count += 1
                    norm_sq = torch.dot(gj_flat, gj_flat).clamp(min=1e-12)

                    # Projection coefficient
                    coeff = dot / norm_sq

                    # Subtract conflicting component param-by-param
                    offset = 0
                    for pi in range(len(projected[i])):
                        sz = task_grads[j][pi].numel()
                        gj_slice = gj_flat[offset: offset + sz].view_as(projected[i][pi])
                        projected[i][pi] = projected[i][pi] - coeff * gj_slice
                        offset += sz

        # Track conflict statistics
        self._last_conflict_count = conflict_count
        self._last_total_pairs    = total_pairs
        self.conflict_rate = conflict_count / max(total_pairs, 1)

        return projected
