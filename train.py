"""
train.py
────────
Main entry point for training MTGRLModel on Tox21.

Usage
-----
  python train.py                                    # uses configs/tox21_gin5.yaml
  python train.py --config configs/tox21_gin5.yaml
  python train.py --config configs/tox21_gin5.yaml --no-pcgrad --no-uncertainty
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.dataset import Tox21Dataset
from src.models.model import MTGRLModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MTGRLModel on Tox21")
    parser.add_argument("--config", default="configs/tox21_gin5.yaml")
    parser.add_argument("--no-pcgrad",      action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--device",  type=str, default=None)
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.no_pcgrad:
        cfg["use_pcgrad"] = False
    if args.no_uncertainty:
        cfg["use_uncertainty_weighting"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logger.warning("CUDA not available — training on CPU (slow!)")

    set_seed(cfg.get("seed", 42))

    logger.info("Config: %s", cfg)
    logger.info("Device: %s", device)

    # ── W&B initialisation ────────────────────────────────────────────────
    if cfg.get("use_wandb", False):
        import wandb, os as _os
        wandb.init(
            project=cfg.get("wandb_project", "graphmol-tox21"),
            entity=cfg.get("wandb_entity", None),
            config=cfg,
            name=f"gin{cfg['gin_layers']}_h{cfg['hidden_dim']}"
                 f"_pcgrad{cfg['use_pcgrad']}_unc{cfg['use_uncertainty_weighting']}",
        )

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir = cfg.get("data_dir", "data")
    logger.info("Loading Tox21 dataset from '%s'…", data_dir)

    train_dataset = Tox21Dataset(root=data_dir, split="train")
    val_dataset   = Tox21Dataset(root=data_dir, split="val")
    test_dataset  = Tox21Dataset(root=data_dir, split="test")

    logger.info("Dataset sizes → train: %d | val: %d | test: %d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    batch_size = cfg.get("batch_size", 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MTGRLModel.from_config(cfg)
    model.count_parameters()

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
    )

    test_metrics = trainer.run()

    # ── Print final results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (scaffold split)")
    print("=" * 60)
    for task, auc in sorted(test_metrics.items()):
        marker = "◀ BEST" if task == "avg_auc" else ""
        print(f"  {task:<25} {auc:.4f}  {marker}")
    print("=" * 60)


if __name__ == "__main__":
    main()
