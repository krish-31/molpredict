from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

# ── Molecule Properties & Prediction ─────────────────────────────────

class MolecularProperties(BaseModel):
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    num_rings: Optional[int] = None

class TaskPrediction(BaseModel):
    probability: float
    label: int
    task_description: str

class PredictRequest(BaseModel):
    smiles: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_attention: bool = False
    model_version: Optional[str] = None

class PredictResponse(BaseModel):
    request_id: str
    smiles: str
    canonical_smiles: Optional[str] = None
    compound_name: Optional[str] = None
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    pubchem_cid: Optional[int] = None
    name_confidence: Optional[str] = None
    synonyms: Optional[List[str]] = None
    is_valid: bool
    molecular_properties: Optional[MolecularProperties] = None
    predictions: Dict[str, TaskPrediction]
    model_version: str
    inference_time_ms: int
    svg_structure: Optional[str] = None
    attention_maps: Optional[Dict[str, List[float]]] = None

# ── Batch Job ────────────────────────────────────────────────────────

class BatchJobCreate(BaseModel):
    model_version: Optional[str] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    total_molecules: int
    processed: int
    failed: int
    progress_pct: float
    eta_seconds: Optional[int] = None
    download_url: Optional[str] = None
    created_at: datetime

class BatchResultRow(BaseModel):
    smiles: str
    canonical_smiles: Optional[str] = None
    compound_name: Optional[str] = None
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    pubchem_cid: Optional[int] = None
    name_confidence: Optional[str] = None
    synonyms: Optional[List[str]] = None
    is_valid: bool
    predictions: Dict[str, float]  # Task -> Probability
    molecular_properties: Optional[MolecularProperties] = None

# ── Training ──────────────────────────────────────────────────────────

class TrainingConfigSchema(BaseModel):
    run_name: str
    gin_layers: int = 5
    hidden_dim: int = 300
    dropout: float = 0.5
    pooling: str = "sum"  # sum | mean | max
    dataset: str = "tox21"
    split_strategy: str = "scaffold"  # scaffold | random
    batch_size: int = 128
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20
    use_pcgrad: bool = True
    use_uncertainty_weighting: bool = True

class EpochMetricsSchema(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    avg_val_auc: float
    per_task_auc: Dict[str, float]
    conflict_rate: float
    uncertainty_weights: Dict[str, float]

class TrainingRunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_epoch: int
    total_epochs: int
    best_avg_auc: float
    current_conflict_rate: float
    latest_metrics: Optional[EpochMetricsSchema] = None
    created_at: datetime
