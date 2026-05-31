"""
metrics.py
──────────
Evaluation metrics for multi-task molecular property prediction.

Functions
─────────
compute_task_auc(logits, labels, task_idx)
    ROC-AUC for a single task, handling missing labels (y == -1).

compute_all_aucs(all_logits, all_labels, task_names)
    ROC-AUC for all tasks + average.

compute_pos_weights(dataset, num_tasks)
    Per-task positive class weights for imbalanced BCE.

compute_conflict_matrix(task_grads)
    Pairwise cosine similarity matrix of task gradient vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def compute_task_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_idx: int,
) -> Optional[float]:
    """
    ROC-AUC for a single task column.

    Parameters
    ----------
    logits  : [N, num_tasks]  raw logit tensor
    labels  : [N, num_tasks]  long tensor (0/1/-1)
    task_idx: column index

    Returns
    -------
    auc : float or None if all labels are the same class / all missing
    """
    y_true = labels[:, task_idx].cpu().numpy()
    y_score = torch.sigmoid(logits[:, task_idx]).cpu().detach().numpy()

    mask = y_true != -1
    if mask.sum() == 0:
        return None

    y_true  = y_true[mask]
    y_score = y_score[mask]

    # sklearn raises if only one class is present
    if len(np.unique(y_true)) < 2:
        return None

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        logger.warning("AUC computation failed for task %d: %s", task_idx, e)
        return None


def compute_all_aucs(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    task_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute ROC-AUC for every task and the macro-average.

    Parameters
    ----------
    all_logits : [N, num_tasks]
    all_labels : [N, num_tasks]
    task_names : list of str (optional)

    Returns
    -------
    dict with keys: task names + "avg_auc"
    """
    num_tasks = all_logits.shape[1]
    names = task_names or [f"task_{i}" for i in range(num_tasks)]

    results: Dict[str, float] = {}
    valid_aucs: List[float] = []

    for i, name in enumerate(names):
        auc = compute_task_auc(all_logits, all_labels, i)
        if auc is not None:
            results[name] = round(auc, 6)
            valid_aucs.append(auc)
        else:
            results[name] = float("nan")

    results["avg_auc"] = round(float(np.mean(valid_aucs)), 6) if valid_aucs else float("nan")
    return results


def compute_pos_weights(
    dataset,
    num_tasks: int = 12,
) -> List[float]:
    """
    Compute per-task positive class weights from a dataset.

    pos_weight_i = N_neg_i / N_pos_i   (clipped to [0.1, 100])

    Parameters
    ----------
    dataset : iterable of PyG Data objects with .y attribute [num_tasks]

    Returns
    -------
    list of float, length num_tasks
    """
    pos_counts = np.zeros(num_tasks)
    neg_counts = np.zeros(num_tasks)

    for data in dataset:
        y = data.y.numpy()  # [num_tasks]
        for i in range(num_tasks):
            if y[i] == 1:
                pos_counts[i] += 1
            elif y[i] == 0:
                neg_counts[i] += 1

    weights = []
    for p, n in zip(pos_counts, neg_counts):
        if p == 0:
            weights.append(1.0)
        else:
            w = n / p
            weights.append(float(np.clip(w, 0.1, 100.0)))

    logger.info("Positive class weights: %s", [round(w, 2) for w in weights])
    return weights


def compute_conflict_matrix(
    task_grads: List[List[torch.Tensor]],
) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix of task gradient vectors.

    Parameters
    ----------
    task_grads : list of num_tasks lists of gradient tensors (per param)

    Returns
    -------
    matrix : [num_tasks, num_tasks] numpy array
             diagonal = 1.0 (self-similarity)
             negative values → conflicting tasks
    """
    num_tasks = len(task_grads)

    # Flatten each task gradient into a single vector
    flat_grads = []
    for grads in task_grads:
        flat = torch.cat([g.view(-1) for g in grads])
        flat_grads.append(flat)

    matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            gi = flat_grads[i]
            gj = flat_grads[j]
            norm_i = gi.norm().clamp(min=1e-12)
            norm_j = gj.norm().clamp(min=1e-12)
            cos_sim = (torch.dot(gi, gj) / (norm_i * norm_j)).item()
            matrix[i, j] = cos_sim

    return matrix
