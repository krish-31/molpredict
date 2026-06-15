# Implementation Plan
## Multi-Task Graph Representation Learning for Molecular Property Prediction

**Version:** 1.0  
**Date:** May 2026  
**Timeline:** 6 Weeks  
**Status:** ✅ Approved — Executing

---

## Overview

Build an end-to-end system that:
1. Converts SMILES → molecular graphs using RDKit
2. Trains a GIN backbone with 12 task-specific attention heads
3. Applies PCGrad + learned uncertainty weighting to prevent negative transfer
4. Uses scaffold splits for realistic evaluation
5. Exposes predictions via a FastAPI REST API
6. Provides a React web dashboard for interactive use

---

## Locked Decisions

> [!NOTE]
> **Q1 — Scope:** Tox21 only (12 binary classification tasks). Regression tasks (ESOL, Lipophilicity) deferred to Phase 2.

> [!NOTE]
> **Q2 — Deployment:** Docker on local machine (CPU inference). No cloud deployment needed for v1.

> [!NOTE]
> **Q3 — Logging:** Weights & Biases (wandb). API key stored as Kaggle secret.

> [!IMPORTANT]
> **ML Training Platform:** Kaggle Notebooks (P100 GPU, 30h/week free). Training runs ~1.5h per experiment. Checkpoint downloaded and placed at `checkpoints/v1.0.0/model.pt` for local Docker inference.

---

## Project Structure

```
project/
├── data/
│   ├── raw/                  # downloaded Tox21 CSV
│   ├── processed/            # featurized PyG datasets
│   └── splits/               # scaffold split indices
├── src/
│   ├── data/
│   │   ├── featurizer.py     # SMILES → atom/bond features
│   │   ├── dataset.py        # PyG Dataset class
│   │   └── scaffold_split.py # Murcko scaffold splitting
│   ├── models/
│   │   ├── gin.py            # GIN backbone (GINEConv layers)
│   │   ├── task_heads.py     # 12 attention-based task heads
│   │   └── model.py          # Full multi-task model
│   ├── training/
│   │   ├── pcgrad.py         # PCGrad optimizer wrapper
│   │   ├── losses.py         # Learned uncertainty loss (Kendall)
│   │   ├── trainer.py        # Training loop + eval
│   │   └── metrics.py        # ROC-AUC, RMSE computation
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── batch.py
│   │   │   ├── train.py
│   │   │   └── metrics.py
│   │   ├── schemas.py        # Pydantic models
│   │   └── inference.py      # model loading + inference
│   └── utils/
│       ├── config.py         # Config dataclass / YAML loading
│       └── logging.py
├── frontend/                 # React + Vite app
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── api/              # API client
│   └── package.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_ablation_study.ipynb
├── tests/
│   ├── test_featurizer.py
│   ├── test_pcgrad.py
│   ├── test_model.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── alembic/                  # DB migrations
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Phase 1 — Data Pipeline (Week 1)

### Milestone: Working molecular graph dataset with scaffold splits

#### [NEW] `src/data/featurizer.py`
- Implement `MoleculeFeaturizer` class
  - `smiles_to_graph(smiles) → torch_geometric.data.Data`
  - Atom features: atomic number (44), chirality (4), degree (11), formal charge (10), H count (9), hybridization (5), aromaticity (1), mass (1) → **85-dim**
  - Bond features: bond type (4), conjugated (1), in_ring (1), stereo (6) → **12-dim**
  - Handle invalid SMILES with `None` return + logging
  - Unit test: verify feature dims on Aspirin

#### [NEW] `src/data/dataset.py`
- `Tox21Dataset(torch_geometric.data.InMemoryDataset)`
  - Download raw CSV from MoleculeNet
  - Run `MoleculeFeaturizer` on all SMILES
  - Store as `.pt` processed file
  - Handle 12-task labels with NaN masking (→ `-1` sentinel)

#### [NEW] `src/data/scaffold_split.py`
- `scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1)`
  - Use `rdkit.Chem.Scaffolds.MurckoScaffold`
  - Group by scaffold → sort by group size descending
  - Assign large scaffold groups to test first (harder split)
  - Return `train_indices, val_indices, test_indices`

#### [NEW] `data/` setup
- Download script: `scripts/download_tox21.py` (from MoleculeNet URL)
- Run featurization pipeline and validate statistics

**Verification:**
```bash
python -m pytest tests/test_featurizer.py -v
python scripts/download_tox21.py
python -c "from src.data.dataset import Tox21Dataset; d = Tox21Dataset('data/'); print(len(d))"
```
Expected: ~8014 molecules, 85-dim nodes, 12-dim edges

---

## Phase 2 — GIN Backbone + Task Heads (Week 2)

### Milestone: Forward pass from molecular graph → 12 property predictions

#### [NEW] `src/models/gin.py`
- `GINBackbone(nn.Module)`
  - `GINEConv` layers (handles edge features)
  - Each layer: Linear → BatchNorm → ReLU → Dropout
  - Residual connections between layers
  - Global sum pooling → graph-level embedding `[B, hidden_dim]`
  - Config: `num_layers=5, hidden_dim=300, dropout=0.5`

#### [NEW] `src/models/task_heads.py`
- `TaskAttentionHead(nn.Module)` — one per task
  - Attention: `Linear(H) → Tanh → Linear(1) → Softmax`
  - Predictor: `Linear(H→H/2) → ReLU → Dropout → Linear(H/2→1)`
- `TaskHeadCollection(nn.ModuleList)` — wraps all 12 heads

#### [NEW] `src/models/model.py`
- `MTGRLModel(nn.Module)` — full model
  - `forward(batch) → List[Tensor]` — 12 logit tensors
  - `predict(smiles) → Dict[str, float]` — single-molecule inference
  - `predict_batch(smiles_list) → List[Dict]`
  - `count_parameters()` helper

**Verification:**
```bash
python -m pytest tests/test_model.py -v
python -c "
from src.models.model import MTGRLModel
m = MTGRLModel()
print('Params:', m.count_parameters())
"
```
Expected: ~1.2M parameters; forward pass without error on dummy batch

---

## Phase 3 — Training: PCGrad + Uncertainty Loss (Week 3)

### Milestone: Full training loop with both loss strategies

#### [NEW] `src/training/losses.py`
- `MultiTaskBCELoss(nn.Module)`
  - Per-task BCE with `pos_weight` for class imbalance
  - NaN label masking: skip samples where label == -1
- `LearnedUncertaintyLoss(nn.Module)`
  - Learnable `log_sigma` parameter `[num_tasks]`
  - `forward(task_losses) → scalar` via Kendall formula
  - `get_weights() → Dict[str, float]` for monitoring

#### [NEW] `src/training/pcgrad.py`
- `PCGrad` optimizer wrapper
  - `pc_backward(losses: List[Tensor])` — per-task loss list
  - `_compute_task_gradients()` — retain_graph per task
  - `_project_conflicting_gradients()` — pairwise dot-product check
  - `_count_conflicts() → float` — conflict rate monitoring
  - `step()` — delegates to wrapped optimizer

#### [NEW] `src/training/metrics.py`
- `compute_roc_auc(logits, labels, task_idx)` — handles NaN
- `compute_avg_auc(all_logits, all_labels)` — mean over tasks
- `compute_conflict_matrix(task_grads)` — pairwise dot products

#### [NEW] `src/training/trainer.py`
- `Trainer` class
  - `train_epoch()` → total loss, per-task losses, conflict rate
  - `eval_epoch()` → per-task AUC, avg AUC
  - Early stopping with patience
  - Checkpoint saving: save when `avg_val_auc` improves
  - Logging: per-epoch dict to file + optionally wandb
  - `run()` → full training loop

**Verification:**
```bash
python -m pytest tests/test_pcgrad.py -v   # verify gradient projection math
python -c "
from src.training.trainer import Trainer
t = Trainer(config='configs/debug.yaml')   # tiny dataset, 2 epochs
t.run()
print('Training complete')
"
```
Expected: Loss decreases over 2 debug epochs; no NaN losses

---

## Phase 4 — Experiments & Ablation (Week 4)

### Milestone: Reproduce benchmark results + ablation study

#### Training Runs (in order):
1. **Baseline — Single-Task GINs** (12 separate models)
   ```bash
   python train.py --mode single_task --dataset tox21 --split scaffold
   ```

2. **MTL Naïve (equal weights, no PCGrad)**
   ```bash
   python train.py --mode mtl --no-pcgrad --no-uncertainty --split scaffold
   ```

3. **MTL + Uncertainty only**
   ```bash
   python train.py --mode mtl --no-pcgrad --uncertainty --split scaffold
   ```

4. **MTL + PCGrad only**
   ```bash
   python train.py --mode mtl --pcgrad --no-uncertainty --split scaffold
   ```

5. **Full model (PCGrad + Uncertainty)**
   ```bash
   python train.py --mode mtl --pcgrad --uncertainty --split scaffold
   ```

6. **Random split comparison (model 5 config)**
   ```bash
   python train.py --mode mtl --pcgrad --uncertainty --split random
   ```

#### [NEW] `notebooks/03_ablation_study.ipynb`
- Load metrics from all 6 runs
- Plot AUC comparison table
- Plot conflict rate over epochs (PCGrad vs naïve)
- Plot per-task uncertainty weights σ_i evolution
- Generalization gap: scaffold vs. random split

**Verification:**  
Target: Full model avg AUC ≥ 0.82 on scaffold test split

---

## Phase 5 — API Layer (Week 4–5)

### Milestone: FastAPI server running with all endpoints

#### [NEW] `src/api/main.py`
- FastAPI app setup
- CORS, exception handlers, lifespan (model load on startup)
- Mount routes

#### [NEW] `src/api/inference.py`
- `ModelInference` class (singleton)
  - Load model from checkpoint on startup
  - `predict(smiles) → PredictResponse`
  - `predict_batch(smiles_list) → List`
  - GPU/CPU device management
  - Warm-up on startup (dummy forward pass)

#### [NEW] `src/api/routes/predict.py`
- `POST /predict` — single SMILES
- `POST /predict/batch` — batch (async job)
- `GET /predict/batch/{job_id}` — job status
- `GET /predict/export/{request_id}` — download CSV

#### [NEW] `src/api/routes/train.py`
- `POST /train` — start training run (async)
- `GET /train/{run_id}` — run status
- `GET /train/{run_id}/metrics` — epoch metrics stream (SSE)

#### [NEW] `src/api/routes/model.py`
- `GET /model/versions` — list model versions
- `GET /model/metrics` — current production model test metrics
- `POST /model/{version}/activate` — set production model

#### [NEW] Database setup
- `alembic upgrade head` — create all tables
- `scripts/seed_tox21.py` — populate molecules + labels from dataset

**Verification:**
```bash
pytest tests/test_api.py -v
uvicorn src.api.main:app --reload
# Test with:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

---

## Phase 6 — Frontend (Week 5)

### Milestone: Full React dashboard connected to API

#### Setup
```bash
cd frontend
npm create vite@latest . -- --template react
npm install recharts axios react-router-dom
```

#### [NEW] Pages and Components
| File | Description |
|------|-------------|
| `pages/Landing.jsx` | Hero section, quick-pick chips, feature cards |
| `pages/Predict.jsx` | SMILES input, 2D viewer, 12-property results |
| `pages/BatchUpload.jsx` | Drag-and-drop CSV, progress bar, preview table |
| `pages/TrainConfig.jsx` | Hyperparameter form, MTL strategy toggles |
| `pages/TrainMonitor.jsx` | Live charts (loss, AUC, conflict rate, σ_i) |
| `pages/Results.jsx` | Final eval table, ROC curve viewer |
| `components/SMILESInput.jsx` | Validated input with green/red badge |
| `components/MoleculeViewer.jsx` | RDKit.js 2D SVG render |
| `components/PropertyBar.jsx` | Animated probability bar |
| `components/AUCHeatmap.jsx` | Epoch × Task heatmap (Recharts) |
| `components/ConflictChart.jsx` | Bar chart for conflict rate |
| `api/client.js` | Axios API client with base URL config |

#### Design System (`index.css`)
- Dark mode palette (as defined in Webflow doc)
- Inter font from Google Fonts
- CSS custom properties for all tokens
- Component-level CSS classes (no Tailwind)
- Smooth transitions on all interactive elements

**Verification:**
```bash
npm run dev
# Manual: Enter Aspirin SMILES → verify 12 properties displayed
# Manual: Upload batch CSV → verify progress and download
```

---

## Phase 7 — Containerization & Deployment (Week 6)

#### [NEW] `docker/Dockerfile.api`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### [NEW] `docker/Dockerfile.frontend`
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY frontend/ .
RUN npm ci && npm run build
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```

#### [NEW] `docker/docker-compose.yml`
```yaml
services:
  db:       postgres:15
  api:      ./docker/Dockerfile.api  (port 8000)
  frontend: ./docker/Dockerfile.frontend (port 3000)
  redis:    redis:7-alpine
```

**Verification:**
```bash
docker-compose up --build
curl http://localhost:8000/health
# Open http://localhost:3000 in browser
```

---

## Testing Strategy

| Test Type | Coverage | Tool |
|-----------|----------|------|
| Unit — Featurizer | Atom/bond feature dims, invalid SMILES | pytest |
| Unit — PCGrad | Gradient projection math, conflict detection | pytest |
| Unit — Loss | Uncertainty weighting formula, NaN masking | pytest |
| Unit — Model | Forward pass shapes, parameter count | pytest |
| Integration — API | All endpoints, error codes | pytest + httpx |
| E2E | SMILES → prediction → CSV download | pytest + httpx |
| Performance | Latency benchmarks (CPU + GPU) | pytest |

---

## Week-by-Week Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Data pipeline | Featurizer, Dataset, Scaffold split |
| 2 | Model | GIN backbone + 12 task heads |
| 3 | Training | PCGrad + Uncertainty loss + Trainer |
| 4 | Experiments | 6 ablation runs + notebook analysis |
| 4–5 | API | FastAPI + DB + all endpoints |
| 5 | Frontend | React dashboard, all pages |
| 6 | Packaging | Docker, README, final benchmarks |

---

## Success Criteria

- [ ] Average ROC-AUC ≥ 0.82 on Tox21 scaffold test split
- [ ] PCGrad reduces gradient conflict rate by ≥ 20% vs. naïve MTL
- [ ] Single-molecule prediction latency ≤ 500 ms (CPU)
- [ ] All 6 API endpoints returning correct responses
- [ ] Frontend displays all 12 property predictions with animated bars
- [ ] Docker Compose brings up full stack cleanly
- [ ] Test coverage ≥ 80% on core ML modules

---

*End of Implementation Plan v1.0*
