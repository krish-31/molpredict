import uuid
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, Table
)
from sqlalchemy.orm import relationship
from src.api.database import Base

class Molecule(Base):
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)
    smiles = Column(Text, nullable=False)
    canonical_smiles = Column(Text, nullable=False, unique=True, index=True)
    inchi = Column(Text)
    inchikey = Column(String(27), unique=True, index=True)
    molecular_formula = Column(Text)
    molecular_weight = Column(Float)
    num_atoms = Column(Integer)
    num_bonds = Column(Integer)
    num_rings = Column(Integer)
    hbd = Column(Integer)
    hba = Column(Integer)
    logp = Column(Float)
    tpsa = Column(Float)
    murcko_scaffold = Column(Text, index=True)
    source_dataset = Column(String(64), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    labels = relationship("MolecularLabel", back_populates="molecule", cascade="all, delete-orphan")
    prediction_requests = relationship("PredictionRequest", back_populates="molecule")


class MolecularLabel(Base):
    __tablename__ = "molecular_labels"

    id = Column(Integer, primary_key=True, index=True)
    molecule_id = Column(Integer, ForeignKey("molecules.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset = Column(String(64), nullable=False)
    split = Column(String(16), nullable=False, index=True)

    # Tox21 Tasks (0 or 1, or None if missing)
    nr_ar = Column(Integer)
    nr_ar_lbd = Column(Integer)
    nr_ahr = Column(Integer)
    nr_aromatase = Column(Integer)
    nr_er = Column(Integer)
    nr_er_lbd = Column(Integer)
    nr_ppar_gamma = Column(Integer)
    sr_are = Column(Integer)
    sr_atad5 = Column(Integer)
    sr_hse = Column(Integer)
    sr_mmp = Column(Integer)
    sr_p53 = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    molecule = relationship("Molecule", back_populates="labels")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String(64), nullable=False, unique=True, index=True)
    description = Column(Text)
    architecture = Column(JSON, nullable=False)
    training_config = Column(JSON, nullable=False)
    mtl_config = Column(JSON, nullable=False)
    checkpoint_path = Column(Text, nullable=False)
    onnx_path = Column(Text)
    avg_roc_auc = Column(Float)
    per_task_auc = Column(JSON)
    total_params = Column(Integer)
    model_size_mb = Column(Float)
    is_production = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)

    prediction_requests = relationship("PredictionRequest", back_populates="model_version")
    training_runs = relationship("TrainingRun", back_populates="model_version")
    batch_jobs = relationship("BatchJob", back_populates="model_version")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(128), nullable=False)
    run_uuid = Column(String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    status = Column(String(32), nullable=False, default="pending", index=True)  # 'pending', 'running', 'completed', 'failed', 'stopped'
    config = Column(JSON, nullable=False)
    dataset = Column(String(64), nullable=False, default="tox21")
    split_strategy = Column(String(32), nullable=False, default="scaffold")
    total_epochs = Column(Integer)
    best_epoch = Column(Integer)
    best_avg_auc = Column(Float)
    wandb_run_id = Column(String(128))
    log_file_path = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="training_runs")
    epoch_metrics = relationship("EpochMetrics", back_populates="training_run", cascade="all, delete-orphan")


class EpochMetrics(Base):
    __tablename__ = "epoch_metrics"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    train_total_loss = Column(Float)
    val_total_loss = Column(Float)
    train_task_losses = Column(JSON)
    val_task_auc = Column(JSON)
    avg_val_auc = Column(Float)
    conflict_rate = Column(Float)
    conflict_pairs = Column(JSON)
    uncertainty_weights = Column(JSON)
    learning_rate = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    training_run = relationship("TrainingRun", back_populates="epoch_metrics")


class PredictionRequest(Base):
    __tablename__ = "prediction_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_uuid = Column(String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    molecule_id = Column(Integer, ForeignKey("molecules.id"), nullable=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    input_smiles = Column(Text, nullable=False)
    is_valid_smiles = Column(Boolean, nullable=False)
    predictions = Column(JSON)  # {"NR-AR": {"probability": 0.12, "label": 0}, ...}
    attention_maps = Column(JSON)
    inference_time_ms = Column(Integer)
    source = Column(String(32), default="api")  # 'api', 'batch', 'web'
    batch_job_id = Column(Integer, ForeignKey("batch_jobs.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    molecule = relationship("Molecule", back_populates="prediction_requests")
    model_version = relationship("ModelVersion", back_populates="prediction_requests")
    batch_job = relationship("BatchJob", back_populates="prediction_requests")


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_uuid = Column(String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    status = Column(String(32), nullable=False, default="pending", index=True)  # 'pending', 'processing', 'completed', 'failed'
    total_molecules = Column(Integer)
    processed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    input_file_path = Column(Text)
    output_file_path = Column(Text)
    error_log = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="batch_jobs")
    prediction_requests = relationship("PredictionRequest", back_populates="batch_job", cascade="all, delete-orphan")
