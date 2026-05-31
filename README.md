# Multi-Task Graph Representation Learning for Molecular Property Prediction

> Predict **12 Tox21 toxicity endpoints simultaneously** from SMILES strings using a
> Graph Isomorphism Network (GIN) backbone, PCGrad gradient projection, and
> learned uncertainty weighting — evaluated on scaffold splits.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Tox21 dataset
python scripts/download_tox21.py

# 3. Run training (GPU recommended — use Kaggle/Colab)
python train.py --config configs/tox21_gin5.yaml

# 4. Start API server (CPU is fine for inference)
uvicorn src.api.main:app --reload
```

## Architecture

```
SMILES  ──▶  Featurizer  ──▶  GIN Backbone (5 layers, 300-dim)
                                        │
                           ┌────────────┴────────────┐
                    ×12 Attention Heads (one per task)
                           │
                  12 Toxicity Predictions (probabilities)
```

## Key Techniques

| Technique | Purpose |
|-----------|---------|
| **GINEConv** | Maximally expressive message passing with edge features |
| **PCGrad** | Projects conflicting task gradients to prevent negative transfer |
| **Uncertainty Weighting** | Kendall et al. — auto-scales loss per task via learnable σᵢ |
| **Scaffold Split** | Murcko scaffold-based train/val/test — harder, more realistic |

## Results (Scaffold Split, Tox21)

| Model | Avg ROC-AUC |
|-------|------------|
| Single-Task GIN | 0.822 |
| MTL + Equal Weight | 0.815 |
| MTL + Uncertainty | 0.831 |
| MTL + PCGrad | 0.835 |
| **Ours (Full)** | **0.843** |

## Project Structure

```
├── src/
│   ├── data/          # Featurizer, Dataset, Scaffold Split
│   ├── models/        # GIN backbone, Task heads, Full model
│   ├── training/      # PCGrad, Losses, Metrics, Trainer
│   └── utils/         # Config, logging
├── configs/           # YAML hyperparameter configs
├── scripts/           # Data download helpers
├── notebooks/         # Kaggle training notebooks
├── tests/             # Unit tests
├── frontend/          # React dashboard (npm run dev)
└── train.py           # Main entry point
```

## Running on Kaggle (Recommended for training)

1. Upload `notebooks/02_model_training.ipynb` to Kaggle
2. Set **Accelerator = GPU P100**, **Internet = ON**
3. Add `WANDB_API_KEY` to Kaggle Secrets
4. Click **Run All** (~1.5 hours)
5. Download `model.pt` from Output tab

## License

MIT
