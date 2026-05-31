"""
trainer.py
──────────
Full training loop for the MTGRLModel with:
  • PCGrad gradient projection
  • Learned uncertainty loss weighting (Kendall et al.)
  • Masked BCE loss per task
  • Early stopping
  • Checkpoint saving (best val avg AUC)
  • W&B logging
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.models.model import MTGRLModel
from src.training.losses import MultiTaskBCELoss, LearnedUncertaintyLoss
from src.training.pcgrad import PCGrad
from src.training.metrics import compute_all_aucs, compute_pos_weights

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates the full training + evaluation lifecycle.

    Parameters
    ----------
    model        : MTGRLModel instance
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    test_loader  : DataLoader for test split
    cfg          : dict loaded from YAML config
    device       : 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model: MTGRLModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: dict,
        device: str = "cuda",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.device       = device

        self.num_tasks  = cfg.get("num_tasks", 12)
        self.task_names = cfg.get("task_names", [f"task_{i}" for i in range(self.num_tasks)])
        self.max_epochs = cfg.get("max_epochs", 200)
        self.patience   = cfg.get("early_stopping_patience", 20)
        self.use_pcgrad = cfg.get("use_pcgrad", True)
        self.use_uncertainty = cfg.get("use_uncertainty_weighting", True)
        self.grad_clip  = cfg.get("grad_clip_norm", 1.0)
        self.ckpt_dir   = cfg.get("checkpoint_dir", "checkpoints/v1.0.0")
        self.use_wandb  = cfg.get("use_wandb", False)
        self.log_interval = cfg.get("log_interval", 5)

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ── Losses ────────────────────────────────────────────────────────
        pos_weights = compute_pos_weights(train_loader.dataset, self.num_tasks)
        self.bce_loss = MultiTaskBCELoss(
            num_tasks=self.num_tasks,
            pos_weights=pos_weights,
        ).to(device)

        self.uncertainty_loss = LearnedUncertaintyLoss(num_tasks=self.num_tasks).to(device)

        # ── Optimizer ─────────────────────────────────────────────────────
        all_params = (
            list(model.parameters())
            + list(self.uncertainty_loss.parameters())
        )
        base_optimizer = torch.optim.Adam(
            all_params,
            lr=cfg.get("learning_rate", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-5),
        )

        if self.use_pcgrad:
            self.optimizer = PCGrad(base_optimizer)
        else:
            self.optimizer = base_optimizer

        self.scheduler = CosineAnnealingLR(
            base_optimizer if self.use_pcgrad else self.optimizer,
            T_max=self.max_epochs,
            eta_min=1e-5,
        )

        # ── State ─────────────────────────────────────────────────────────
        self.best_val_auc  = 0.0
        self.best_epoch    = 0
        self.epochs_no_imp = 0
        self.history: List[Dict] = []

    # ── Training epoch ────────────────────────────────────────────────────────
    def _train_epoch(self) -> Dict:
        self.model.train()
        self.bce_loss.train()

        total_loss    = 0.0
        task_loss_sum = torch.zeros(self.num_tasks)
        conflict_rate = 0.0
        n_batches     = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch)           # list of [B, 1]
            task_losses = self.bce_loss(logits, batch.y)  # [num_tasks]

            if self.use_uncertainty:
                loss = self.uncertainty_loss(task_losses)
            else:
                loss = task_losses.sum()

            if self.use_pcgrad:
                # pc_backward needs per-task losses separated
                self.optimizer.pc_backward(list(task_losses))
                conflict_rate += self.optimizer.conflict_rate
            else:
                loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss    += loss.item()
            task_loss_sum += task_losses.detach().cpu()
            n_batches     += 1

        conflict_rate /= max(n_batches, 1)
        return {
            "train_loss":    total_loss / n_batches,
            "task_losses":   (task_loss_sum / n_batches).tolist(),
            "conflict_rate": conflict_rate,
        }

    # ── Evaluation epoch ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()

        all_logits_list = [[] for _ in range(self.num_tasks)]
        all_labels_list = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)                    # list of [B, 1]
            labels = batch.y.cpu()                        # [B, num_tasks]

            for i, logit in enumerate(logits):
                all_logits_list[i].append(logit.cpu())

            all_labels_list.append(labels)

        # Stack
        all_logits = torch.cat(
            [torch.cat(lst, dim=0) for lst in all_logits_list], dim=1
        )   # [N, num_tasks]
        all_labels = torch.cat(all_labels_list, dim=0)   # [N, num_tasks]

        auc_dict = compute_all_aucs(all_logits, all_labels, self.task_names)
        return auc_dict

    # ── Main training loop ────────────────────────────────────────────────────
    def run(self) -> Dict:
        """
        Run the complete training loop.

        Returns
        -------
        Dict with final test metrics.
        """
        logger.info("Starting training — max_epochs=%d, device=%s", self.max_epochs, self.device)
        logger.info("PCGrad=%s | Uncertainty=%s", self.use_pcgrad, self.use_uncertainty)

        if self.use_wandb:
            import wandb
            wandb.config.update(self.cfg)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch()
            val_metrics   = self._eval_epoch(self.val_loader)
            val_auc       = val_metrics.get("avg_auc", 0.0)

            self.scheduler.step()

            # ── Checkpoint ─────────────────────────────────────────────
            if val_auc > self.best_val_auc:
                self.best_val_auc  = val_auc
                self.best_epoch    = epoch
                self.epochs_no_imp = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckpt_dir, "model.pt"),
                )
            else:
                self.epochs_no_imp += 1

            # ── Logging ────────────────────────────────────────────────
            elapsed = time.time() - t0
            row = {
                "epoch":         epoch,
                "train_loss":    round(train_metrics["train_loss"], 6),
                "conflict_rate": round(train_metrics["conflict_rate"], 4),
                "val_avg_auc":   val_auc,
                "best_val_auc":  self.best_val_auc,
                "best_epoch":    self.best_epoch,
                "elapsed_s":     round(elapsed, 2),
            }
            row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "avg_auc"})

            # Uncertainty weights
            if self.use_uncertainty:
                row.update(self.uncertainty_loss.get_sigma())

            self.history.append(row)

            if epoch % self.log_interval == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_auc=%.4f | best=%.4f (ep %d) | "
                    "conflict=%.3f | %.1fs",
                    epoch, self.max_epochs,
                    row["train_loss"], val_auc, self.best_val_auc, self.best_epoch,
                    row["conflict_rate"], elapsed,
                )

            if self.use_wandb:
                import wandb
                wandb.log(row, step=epoch)

            # ── Early stopping ─────────────────────────────────────────
            if self.epochs_no_imp >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, self.patience,
                )
                break

        # ── Final test evaluation ───────────────────────────────────────
        logger.info("Loading best checkpoint (epoch %d, val_auc=%.4f)…",
                    self.best_epoch, self.best_val_auc)
        best_path = os.path.join(self.ckpt_dir, "model.pt")
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        test_metrics = self._eval_epoch(self.test_loader)
        logger.info("Test avg AUC: %.4f", test_metrics["avg_auc"])

        for task, auc in test_metrics.items():
            if task != "avg_auc":
                logger.info("  %-20s AUC = %.4f", task, auc)

        if self.use_wandb:
            import wandb
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
            wandb.finish()

        return test_metrics

    def evaluate(self, loader: DataLoader) -> Dict:
        """Public evaluation method for any DataLoader."""
        return self._eval_epoch(loader)
