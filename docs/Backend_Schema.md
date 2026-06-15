# Backend Schema Design
## Multi-Task Graph Representation Learning for Molecular Property Prediction

**Version:** 1.0  
**Date:** May 2026

---

## 1. Database Overview

**Primary DB:** PostgreSQL 15 (production) / SQLite 3.42 (development)  
**ORM:** SQLAlchemy 2.0 with Alembic migrations

```
┌──────────────┐       ┌────────────────┐       ┌───────────────────┐
│  molecules   │──────▷│  predictions   │◁──────│  prediction_runs  │
└──────────────┘       └────────────────┘       └───────────────────┘
                                                         │
                                               ┌─────────▼──────────┐
                                               │   training_runs     │
                                               └─────────┬──────────┘
                                                         │
                                               ┌─────────▼──────────┐
                                               │   epoch_metrics     │
                                               └────────────────────┘
```

---

## 2. Entity Definitions

---

### 2.1 `molecules` Table

Stores canonical molecular records.

```sql
CREATE TABLE molecules (
    id              SERIAL PRIMARY KEY,
    smiles          TEXT        NOT NULL,
    canonical_smiles TEXT       NOT NULL UNIQUE,   -- RDKit canonicalized
    inchi           TEXT,
    inchikey        TEXT        UNIQUE,
    molecular_formula TEXT,
    molecular_weight FLOAT,
    num_atoms       INTEGER,
    num_bonds       INTEGER,
    num_rings       INTEGER,
    hbd             INTEGER,    -- H-bond donors
    hba             INTEGER,    -- H-bond acceptors
    logp            FLOAT,
    tpsa            FLOAT,      -- topological polar surface area
    murcko_scaffold TEXT,       -- for scaffold split grouping
    source_dataset  VARCHAR(64),            -- e.g., 'tox21', 'bace', 'esol'
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_molecules_inchikey ON molecules(inchikey);
CREATE INDEX idx_molecules_scaffold ON molecules(murcko_scaffold);
CREATE INDEX idx_molecules_dataset  ON molecules(source_dataset);
```

---

### 2.2 `molecular_labels` Table

Stores ground-truth labels for each task per molecule. NULLs represent missing measurements.

```sql
CREATE TABLE molecular_labels (
    id              SERIAL PRIMARY KEY,
    molecule_id     INTEGER     NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
    dataset         VARCHAR(64) NOT NULL,   -- 'tox21'
    split           VARCHAR(16) NOT NULL,   -- 'train' | 'val' | 'test'

    -- Tox21 Tasks (NULL = not measured)
    nr_ar           SMALLINT,   -- 0 or 1
    nr_ar_lbd       SMALLINT,
    nr_ahr          SMALLINT,
    nr_aromatase    SMALLINT,
    nr_er           SMALLINT,
    nr_er_lbd       SMALLINT,
    nr_ppar_gamma   SMALLINT,
    sr_are          SMALLINT,
    sr_atad5        SMALLINT,
    sr_hse          SMALLINT,
    sr_mmp          SMALLINT,
    sr_p53          SMALLINT,

    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (molecule_id, dataset)
);

CREATE INDEX idx_labels_molecule ON molecular_labels(molecule_id);
CREATE INDEX idx_labels_split    ON molecular_labels(split);
```

---

### 2.3 `model_versions` Table

Registry of trained model artifacts.

```sql
CREATE TABLE model_versions (
    id              SERIAL PRIMARY KEY,
    version_tag     VARCHAR(64)  NOT NULL UNIQUE,   -- e.g., 'v1.0.0', 'gin5_pcgrad_uncertainty'
    description     TEXT,
    architecture    JSONB        NOT NULL,
    -- example: {"gin_layers": 5, "hidden_dim": 300, "dropout": 0.5, "pooling": "sum"}

    training_config JSONB        NOT NULL,
    -- example: {"dataset": "tox21", "split": "scaffold", "batch_size": 128, "lr": 0.001}

    mtl_config      JSONB        NOT NULL,
    -- example: {"pcgrad": true, "uncertainty_weighting": true}

    checkpoint_path TEXT         NOT NULL,   -- path to .pt file
    onnx_path       TEXT,                    -- optional ONNX export
    
    -- Performance on test set
    avg_roc_auc     FLOAT,
    per_task_auc    JSONB,
    -- example: {"NR-AR": 0.821, "NR-AhR": 0.891, ...}

    total_params    INTEGER,
    model_size_mb   FLOAT,

    is_production   BOOLEAN      DEFAULT FALSE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at     TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_model_production ON model_versions(is_production);
```

---

### 2.4 `training_runs` Table

Tracks individual training experiments.

```sql
CREATE TABLE training_runs (
    id              SERIAL PRIMARY KEY,
    run_name        VARCHAR(128) NOT NULL,
    run_uuid        UUID         NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    model_version_id INTEGER     REFERENCES model_versions(id),

    status          VARCHAR(32)  NOT NULL DEFAULT 'pending',
    -- 'pending' | 'running' | 'completed' | 'failed' | 'stopped'

    config          JSONB        NOT NULL,
    -- full merged config: architecture + training + mtl
    
    dataset         VARCHAR(64)  NOT NULL DEFAULT 'tox21',
    split_strategy  VARCHAR(32)  NOT NULL DEFAULT 'scaffold',

    total_epochs    INTEGER,
    best_epoch      INTEGER,
    best_avg_auc    FLOAT,

    wandb_run_id    VARCHAR(128),
    log_file_path   TEXT,

    started_at      TIMESTAMP WITH TIME ZONE,
    completed_at    TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_runs_status ON training_runs(status);
CREATE INDEX idx_runs_uuid   ON training_runs(run_uuid);
```

---

### 2.5 `epoch_metrics` Table

Per-epoch training and validation metrics (used for live monitoring charts).

```sql
CREATE TABLE epoch_metrics (
    id                  SERIAL PRIMARY KEY,
    training_run_id     INTEGER     NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch               INTEGER     NOT NULL,

    -- Losses
    train_total_loss    FLOAT,
    val_total_loss      FLOAT,

    -- Per-task train losses (JSONB)
    train_task_losses   JSONB,
    -- {"NR-AR": 0.42, "NR-AhR": 0.38, ...}

    -- Per-task val AUC
    val_task_auc        JSONB,
    -- {"NR-AR": 0.821, "NR-AhR": 0.891, ...}
    avg_val_auc         FLOAT,

    -- Gradient conflict monitoring
    conflict_rate       FLOAT,       -- fraction of (i,j) pairs with dot < 0
    conflict_pairs      JSONB,
    -- [{"tasks": ["NR-ER", "SR-MMP"], "dot": -0.14}, ...]

    -- Learned uncertainty weights σ_i per task
    uncertainty_weights JSONB,
    -- {"NR-AR": 0.82, "NR-AhR": 0.61, ...}

    -- Learning rate at this epoch
    learning_rate       FLOAT,

    recorded_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (training_run_id, epoch)
);

CREATE INDEX idx_epoch_run ON epoch_metrics(training_run_id);
```

---

### 2.6 `prediction_requests` Table

Logs all inference requests (for auditing and caching).

```sql
CREATE TABLE prediction_requests (
    id              BIGSERIAL PRIMARY KEY,
    request_uuid    UUID         NOT NULL DEFAULT gen_random_uuid(),
    molecule_id     INTEGER      REFERENCES molecules(id),

    model_version_id INTEGER     NOT NULL REFERENCES model_versions(id),

    input_smiles    TEXT         NOT NULL,
    is_valid_smiles BOOLEAN      NOT NULL,

    -- Predictions
    predictions     JSONB,
    -- {"NR-AR": {"probability": 0.12, "label": 0}, ...}

    attention_maps  JSONB,       -- optional, only if requested
    inference_time_ms INTEGER,

    source          VARCHAR(32)  DEFAULT 'api',
    -- 'api' | 'batch' | 'web'

    batch_job_id    INTEGER      REFERENCES batch_jobs(id),

    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_pred_uuid       ON prediction_requests(request_uuid);
CREATE INDEX idx_pred_molecule   ON prediction_requests(molecule_id);
CREATE INDEX idx_pred_model      ON prediction_requests(model_version_id);
CREATE INDEX idx_pred_created    ON prediction_requests(created_at DESC);
```

---

### 2.7 `batch_jobs` Table

Tracks batch prediction uploads.

```sql
CREATE TABLE batch_jobs (
    id              SERIAL PRIMARY KEY,
    job_uuid        UUID         NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    model_version_id INTEGER     NOT NULL REFERENCES model_versions(id),

    status          VARCHAR(32)  NOT NULL DEFAULT 'pending',
    -- 'pending' | 'processing' | 'completed' | 'failed'

    total_molecules INTEGER,
    processed       INTEGER      DEFAULT 0,
    failed          INTEGER      DEFAULT 0,

    input_file_path TEXT,        -- uploaded CSV path
    output_file_path TEXT,       -- results CSV path

    error_log       TEXT,

    started_at      TIMESTAMP WITH TIME ZONE,
    completed_at    TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_batch_uuid   ON batch_jobs(job_uuid);
CREATE INDEX idx_batch_status ON batch_jobs(status);
```

---

## 3. API Request / Response Schemas (Pydantic)

### 3.1 Prediction Request
```python
class PredictRequest(BaseModel):
    smiles: str                          # required
    threshold: float = 0.5              # classification cutoff
    return_attention: bool = False      # include attention map
    model_version: Optional[str] = None # default: production model

class PredictResponse(BaseModel):
    request_id: str                      # UUID
    smiles: str
    canonical_smiles: str
    is_valid: bool
    molecular_properties: MolecularProperties
    predictions: Dict[str, TaskPrediction]
    model_version: str
    inference_time_ms: int
    attention_maps: Optional[Dict[str, List[float]]] = None

class TaskPrediction(BaseModel):
    probability: float
    label: int                           # 0 or 1
    task_description: str

class MolecularProperties(BaseModel):
    molecular_formula: str
    molecular_weight: float
    num_atoms: int
    hbd: int
    hba: int
    logp: float
    tpsa: float
```

### 3.2 Batch Job
```python
class BatchJobCreate(BaseModel):
    model_version: Optional[str] = None
    threshold: float = 0.5

class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    total_molecules: int
    processed: int
    progress_pct: float
    eta_seconds: Optional[int]
    download_url: Optional[str]         # available when completed
```

### 3.3 Training
```python
class TrainingConfig(BaseModel):
    run_name: str
    # Architecture
    gin_layers: int = 5
    hidden_dim: int = 300
    dropout: float = 0.5
    pooling: str = "sum"                # sum | mean | max
    # Training
    dataset: str = "tox21"
    split_strategy: str = "scaffold"    # scaffold | random
    batch_size: int = 128
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20
    # MTL
    use_pcgrad: bool = True
    use_uncertainty_weighting: bool = True

class TrainingRunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_epoch: int
    total_epochs: int
    best_avg_auc: float
    current_conflict_rate: float
    latest_metrics: EpochMetrics

class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    avg_val_auc: float
    per_task_auc: Dict[str, float]
    conflict_rate: float
    uncertainty_weights: Dict[str, float]
```

---

## 4. File Storage Schema

```
storage/
├── checkpoints/
│   ├── v1.0.0/
│   │   ├── model.pt           # PyTorch state dict
│   │   ├── config.json        # training config
│   │   └── metrics.json       # final test metrics
│   └── ...
├── uploads/
│   ├── batch_jobs/
│   │   ├── {job_uuid}_input.csv
│   │   └── {job_uuid}_output.csv
│   └── ...
├── logs/
│   └── training/
│       └── {run_uuid}/
│           ├── train.log
│           └── events.jsonl    # epoch-level events
└── exports/
    └── predictions/
        └── {request_uuid}.csv
```

---

## 5. Caching Strategy

| Cache Layer | Tool | TTL | Key |
|-------------|------|-----|-----|
| Molecule predictions | Redis | 24h | `pred:{inchikey}:{model_version}` |
| Model inference (ONNX) | In-process LRU | — | in-memory |
| Scaffold computation | PostgreSQL materialized view | Manual refresh | — |
| Batch job status | Redis | 1h | `batch:{job_uuid}` |

---

## 6. Indexing Strategy

| Table | Index | Rationale |
|-------|-------|-----------|
| `molecules` | `inchikey` | De-duplication checks |
| `molecules` | `murcko_scaffold` | Scaffold split queries |
| `prediction_requests` | `(molecule_id, model_version_id)` | Cache lookup |
| `epoch_metrics` | `(training_run_id, epoch)` | Chart streaming |
| `batch_jobs` | `status` | Queue polling |

---

## 7. Migrations (Alembic)

```
alembic/versions/
├── 001_initial_schema.py         # molecules, molecular_labels
├── 002_model_registry.py         # model_versions
├── 003_training_runs.py          # training_runs, epoch_metrics
├── 004_predictions.py            # prediction_requests, batch_jobs
└── 005_add_attention_cache.py    # attention_maps column
```

---

*End of Backend Schema v1.0*
