# Technical Requirements Document (TRD)
## Multi-Task Graph Representation Learning for Molecular Property Prediction

**Version:** 1.0  
**Date:** May 2026  
**Status:** Draft

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│   React Web App   │   REST API Consumers   │   Jupyter Notebook │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────────┐
│                        API LAYER (FastAPI)                      │
│  /predict  │  /predict/batch  │  /model/metrics  │  /train     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     ML INFERENCE ENGINE                         │
│  SMILES Parser → Graph Builder → GIN Backbone → Task Heads      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       DATA LAYER                                │
│   MoleculeDB (SQLite/Postgres)  │  Model Registry  │  Logs     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline

### 2.1 Input Format
- **Primary:** SMILES strings (Simplified Molecular-Input Line-Entry System)
- **Secondary:** SDF files (batch ingestion)
- **Labels:** Binary labels for 12 Tox21 tasks (0 = non-toxic, 1 = toxic, NaN = missing)

### 2.2 SMILES → Molecular Graph Conversion

Using **RDKit**:

```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
```

**Node Features (per atom):**
| Feature | Encoding | Dim |
|---------|----------|-----|
| Atomic number | Integer (one-hot, 44 elements) | 44 |
| Chirality | One-hot (4 classes) | 4 |
| Degree | One-hot (0–10) | 11 |
| Formal charge | One-hot (−5 to +5) | 10 |
| Num Hydrogens | One-hot (0–8) | 9 |
| Hybridization | One-hot (SP, SP2, SP3, SP3D, SP3D2) | 5 |
| Aromaticity | Binary | 1 |
| Atomic mass | Normalized float | 1 |
| **Total** | | **85** |

**Edge Features (per bond):**
| Feature | Encoding | Dim |
|---------|----------|-----|
| Bond type | One-hot (single, double, triple, aromatic) | 4 |
| Conjugated | Binary | 1 |
| In ring | Binary | 1 |
| Stereo | One-hot (6 classes) | 6 |
| **Total** | | **12** |

### 2.3 Scaffold Splitting

Using **RDKit Murcko Scaffold**:

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold
```

**Split Rationale:**
- Group molecules by scaffold
- Assign scaffold groups to train/val/test (80/10/10) by sorted scaffold size (largest scaffolds go to test)
- Ensures test molecules are **structurally novel** relative to training data
- Harder and more realistic than random splits

### 2.4 Dataset Statistics (Tox21)

| Split | Molecules | Scaffold Groups |
|-------|-----------|-----------------|
| Train | ~6,200 | ~1,400 |
| Val | ~775 | ~175 |
| Test | ~775 | ~175 |

**Class Imbalance Handling:**
- Compute positive class weight per task: `w_pos = (N_neg / N_pos)`
- Use `BCEWithLogitsLoss(pos_weight=w_pos)` per task
- Mask NaN labels: `loss = loss[~torch.isnan(labels)]`

---

## 3. Model Architecture

### 3.1 GIN Backbone (Graph Isomorphism Network)

**Theoretical Foundation:**  
GIN is maximally expressive among message-passing GNNs (Weisfeiler-Lehman test equivalent).

**Update Rule:**
```
h_v^(k) = MLP^(k) [ (1 + ε^(k)) · h_v^(k−1) + Σ_{u ∈ N(v)} h_u^(k−1) ]
```

Where:
- `h_v^(k)` = node embedding at layer k
- `ε^(k)` = learnable (or fixed) scalar
- `N(v)` = neighbors of node v
- MLP = 2-layer MLP with BatchNorm + ReLU

**Architecture Config:**
```python
GINConfig:
  num_layers:    5
  hidden_dim:    300
  dropout:       0.5
  residual:      True        # skip connections between layers
  graph_pooling: "sum"       # global sum pooling → graph-level embedding
  input_dim:     85          # atom feature dim
  edge_dim:      12          # bond feature dim (used in GINEConv)
```

**Output:** A single graph-level embedding vector of shape `[batch_size, 300]`

### 3.2 Task-Specific Attention Heads

One head per task (12 total):

```python
class TaskHead(nn.Module):
    def __init__(self, hidden_dim=300, num_tasks=1):
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)   # binary classification logit
        )
    
    def forward(self, graph_emb):
        attn = self.attention(graph_emb)          # [B, 1]
        weighted = attn * graph_emb               # [B, H]
        return self.predictor(weighted)           # [B, 1]
```

### 3.3 Full Model

```
Input: SMILES
  ↓ RDKit Featurizer
Molecular Graph [N_atoms × 85 node features, N_bonds × 12 edge features]
  ↓ GINEConv × 5 layers
Node Embeddings [N_atoms × 300]
  ↓ Global Sum Pooling
Graph Embedding [B × 300]
  ↓ ×12 Task Attention Heads (parallel)
[B × 1] × 12 logits
  ↓ Sigmoid
12 Predicted Property Probabilities
```

---

## 4. Loss Function — Learned Uncertainty Weighting

Based on **Kendall et al. (2018)** — "Multi-Task Learning Using Uncertainty to Weigh Losses":

**Per-task loss:**
```
L_i(θ) = BCE(ŷ_i, y_i)   [for binary classification tasks]
```

**Combined multi-task loss:**
```
L_total = Σ_i [ (1 / (2σ_i²)) · L_i(θ) + log(σ_i) ]
```

Where:
- `σ_i` = learnable log-uncertainty parameter per task (initialized to 0)
- Tasks with high uncertainty automatically receive lower weight
- `log(σ_i)` acts as a regularizer preventing σ → ∞

**Implementation:**
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=12):
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        # losses: tensor of shape [12]
        weights = torch.exp(-2 * self.log_sigma)         # 1/σ²
        weighted = weights * losses + self.log_sigma      # L_i/σ² + log(σ)
        return weighted.sum()
```

---

## 5. PCGrad — Projecting Conflicting Gradients

### 5.1 Motivation
When gradients from two tasks `g_i` and `g_j` are conflicting (their dot product < 0), updating in both directions simultaneously leads to interference — this is **negative transfer**.

### 5.2 Algorithm

For each task pair (i, j):
```
if g_i · g_j < 0:
    g_i ← g_i - (g_i · g_j / ||g_j||²) · g_j
```

This **projects** `g_i` onto the plane orthogonal to `g_j`, removing the conflicting component.

### 5.3 Implementation

```python
class PCGrad:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._task_losses = []

    def pc_backward(self, losses):
        # losses: list of 12 per-task scalar losses
        task_grads = self._compute_task_grads(losses)
        projected_grads = self._project_grads(task_grads)
        self._set_grads(projected_grads)
        
    def _project_grads(self, grads):
        projected = copy.deepcopy(grads)
        for i, g_i in enumerate(projected):
            for j, g_j in enumerate(grads):
                if i == j: continue
                dot = sum((a * b).sum() for a, b in zip(g_i, g_j))
                if dot < 0:
                    norm_sq = sum((b * b).sum() for b in g_j)
                    for k in range(len(g_i)):
                        projected[i][k] -= (dot / norm_sq) * g_j[k]
        return projected

    def step(self):
        self.optimizer.step()
```

### 5.4 Monitoring
- **Conflict Rate** = fraction of task pairs (i,j) where `g_i · g_j < 0` per epoch
- Logged per epoch; expected to decrease as model converges

---

## 6. Training Loop

```
For each epoch:
  For each mini-batch:
    1. Forward pass → 12 logit tensors
    2. Compute per-task BCE losses (mask NaN labels)
    3. Compute total loss via Learned Uncertainty Weighting
    4. PCGrad.pc_backward(per_task_losses)
    5. Clip gradients (max_norm=1.0)
    6. Optimizer.step() [Adam, lr=1e-3, weight_decay=1e-5]
    7. LR scheduler step [CosineAnnealingLR]
  
  Validation:
    - Compute ROC-AUC per task (handle NaN labels)
    - Log average AUC + per-task AUC
    - Save checkpoint if avg_AUC improved

Early stopping: patience=20 epochs
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Epochs (max) | 200 |
| GIN layers | 5 |
| Hidden dim | 300 |
| Dropout | 0.5 |
| Early stopping patience | 20 |

---

## 7. Evaluation Protocol

### 7.1 Metrics
- **Classification tasks:** ROC-AUC (area under ROC curve)
- **Regression tasks (Phase 2):** RMSE, R²
- **Multi-task summary:** Average ROC-AUC across all 12 tasks

### 7.2 Baselines

| Model | Strategy |
|-------|----------|
| Single-Task GIN | One GIN per task, independent training |
| MTL-GIN (equal weights) | Shared GIN, naïve equal weighting |
| MTL-GIN + Uncertainty | Shared GIN + Kendall weighting |
| MTL-GIN + PCGrad | Shared GIN + PCGrad (no uncertainty) |
| **Ours** | Shared GIN + Uncertainty + PCGrad |

### 7.3 Ablation Studies
1. Remove PCGrad → measure negative transfer rate
2. Remove uncertainty weighting → measure task dominance
3. Reduce GIN layers (3 vs 5) → measure capacity effect
4. Random split vs. scaffold split → measure generalization gap

---

## 8. API Design

### 8.1 `POST /predict`

**Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "return_attention": false
}
```

**Response:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": {
    "NR-AR":       { "probability": 0.12, "label": 0 },
    "NR-AR-LBD":   { "probability": 0.08, "label": 0 },
    "NR-AhR":      { "probability": 0.45, "label": 0 },
    "NR-Aromatase":{ "probability": 0.23, "label": 0 },
    "NR-ER":       { "probability": 0.31, "label": 0 },
    "NR-ER-LBD":   { "probability": 0.17, "label": 0 },
    "NR-PPAR-gamma":{ "probability": 0.09, "label": 0 },
    "SR-ARE":      { "probability": 0.52, "label": 1 },
    "SR-ATAD5":    { "probability": 0.14, "label": 0 },
    "SR-HSE":      { "probability": 0.28, "label": 0 },
    "SR-MMP":      { "probability": 0.11, "label": 0 },
    "SR-p53":      { "probability": 0.19, "label": 0 }
  },
  "inference_time_ms": 47
}
```

### 8.2 Error Codes
| Code | Meaning |
|------|---------|
| 422 | Invalid SMILES string |
| 503 | Model not loaded |
| 500 | Internal featurization error |

---

## 9. Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch 2.1 + PyTorch Geometric 2.4 |
| Cheminformatics | RDKit 2023.09 |
| API Server | FastAPI 0.110 + Uvicorn |
| Database | PostgreSQL 15 (prod) / SQLite (dev) |
| Frontend | React 18 + Vite + Recharts |
| Molecular Viz | RDKit.js / Kekule.js |
| Containerization | Docker + Docker Compose |
| Experiment Tracking | Weights & Biases (wandb) |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-asyncio |

---

## 10. Performance Requirements

| Scenario | Target |
|----------|--------|
| Single SMILES prediction (GPU) | < 50 ms |
| Single SMILES prediction (CPU) | < 500 ms |
| Batch of 1,000 molecules (GPU) | < 10 s |
| Model file size | < 50 MB |
| Training time (Tox21, A100 GPU) | < 2 hours |
| Training time (Tox21, CPU) | < 24 hours |

---

*End of TRD v1.0*
