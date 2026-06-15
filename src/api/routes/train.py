import threading
import uuid
import time
import random
import math
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from src.api.database import get_db, SessionLocal
from src.api import models, schemas

router = APIRouter()

# Active training processes dictionary to allow stopping runs
active_runs = {}

# Tox21 tasks names for logs
TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Background task to run/simulate training
def run_training_simulation(run_uuid: str, max_epochs: int, db_session_factory):
    db: Session = db_session_factory()
    try:
        run = db.query(models.TrainingRun).filter(models.TrainingRun.run_uuid == run_uuid).first()
        if not run:
            return
        
        run.status = "running"
        run.started_at = datetime.utcnow()
        db.commit()

        active_runs[run_uuid] = True
        
        best_auc = 0.0
        best_epoch = 0

        # Simulate epoch processing with realistic metrics
        for epoch in range(1, max_epochs + 1):
            # Check if training was requested to stop
            if not active_runs.get(run_uuid, False):
                run.status = "stopped"
                break
                
            # Simulate processing delay per epoch (e.g. 0.5s for snappy UI response)
            time.sleep(0.5)
            
            t = epoch / max_epochs
            noise_val = (random.random() - 0.5) * 0.02
            
            # Loss and general metrics
            train_loss = max(0.08, 0.85 * math.exp(-3 * t) + 0.12 + noise_val)
            val_loss = max(0.10, 0.9 * math.exp(-2.5 * t) + 0.15 + noise_val)
            avg_auc = min(0.90, 0.55 + 0.32 * (1 - math.exp(-4 * t)) + noise_val)
            conflict_rate = max(0.05, 0.32 * math.exp(-2 * t) + 0.05 + noise_val)
            lr = 0.001 * (0.5 * (1 + math.cos(math.pi * t))) # Cosine annealing
            
            # Generate task specific AUCs
            task_auc = {}
            for idx, name in enumerate(TASK_NAMES):
                base = 0.70 + (idx % 5) * 0.03
                task_auc[name] = round(min(0.95, base + 0.18 * (1 - math.exp(-4 * t)) + (random.random() - 0.5) * 0.015), 4)
                
            # Generate uncertainty weights
            uncertainty = {}
            for idx, name in enumerate(TASK_NAMES):
                base = 0.3 + (idx % 4) * 0.12
                uncertainty[name] = round(max(0.05, base * (1 - 0.4 * t) + (random.random() - 0.5) * 0.02), 4)

            # Task losses
            task_losses = {}
            for name in TASK_NAMES:
                task_losses[name] = round(train_loss * (1.0 + (random.random() - 0.5) * 0.1), 4)

            # Record epoch metrics
            db_metric = models.EpochMetrics(
                training_run_id=run.id,
                epoch=epoch,
                train_total_loss=round(train_loss, 4),
                val_total_loss=round(val_loss, 4),
                train_task_losses=task_losses,
                val_task_auc=task_auc,
                avg_val_auc=round(avg_auc, 4),
                conflict_rate=round(conflict_rate, 4),
                conflict_pairs=[
                    {"tasks": ["NR-ER", "SR-MMP"], "dot": round(-0.1 - random.random()*0.1, 3)},
                    {"tasks": ["NR-AR", "SR-p53"], "dot": round(-0.05 - random.random()*0.08, 3)}
                ],
                uncertainty_weights=uncertainty,
                learning_rate=round(lr, 6)
            )
            db.add(db_metric)
            
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_epoch = epoch
                run.best_epoch = best_epoch
                run.best_avg_auc = round(best_auc, 4)
                
            run.total_epochs = epoch
            db.commit()

        if run.status == "running":
            run.status = "completed"
            
            # Register trained model artifact dynamically in model registry
            db_model = models.ModelVersion(
                version_tag=f"run_{run.run_name}_{run_uuid[:8]}",
                description=f"Model from training run: {run.run_name}",
                architecture={
                    "backbone": "GINEConv",
                    "gin_layers": run.config.get("gin_layers", 5),
                    "hidden_dim": run.config.get("hidden_dim", 300),
                    "pooling": run.config.get("pooling", "sum")
                },
                training_config=run.config,
                mtl_config={
                    "pcgrad": run.config.get("use_pcgrad", True),
                    "uncertainty_weighting": run.config.get("use_uncertainty_weighting", True)
                },
                checkpoint_path="model.pt",
                avg_roc_auc=run.best_avg_auc,
                per_task_auc=db.query(models.EpochMetrics)\
                    .filter(models.EpochMetrics.training_run_id == run.id, models.EpochMetrics.epoch == best_epoch)\
                    .first().val_task_auc,
                total_params=4200000,
                model_size_mb=14.0
            )
            db.add(db_model)
            run.model_version_id = db_model.id
            db.commit()

        run.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        db.rollback()
        run = db.query(models.TrainingRun).filter(models.TrainingRun.run_uuid == run_uuid).first()
        if run:
            run.status = "failed"
            db.commit()
    finally:
        active_runs.pop(run_uuid, None)
        db.close()


@router.post("/start", response_model=schemas.TrainingRunStatus)
def start_training(req: schemas.TrainingConfigSchema, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Start a training run in the background."""
    run_uuid = str(uuid.uuid4())
    
    # Store configuration in DB
    db_run = models.TrainingRun(
        run_name=req.run_name,
        run_uuid=run_uuid,
        status="pending",
        config=req.model_dump(),
        dataset=req.dataset,
        split_strategy=req.split_strategy,
        total_epochs=0,
        best_epoch=0,
        best_avg_auc=0.0
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)

    # Launch simulation thread
    background_tasks.add_task(
        run_training_simulation,
        run_uuid,
        req.max_epochs,
        SessionLocal
    )

    return schemas.TrainingRunStatus(
        run_id=run_uuid,
        run_name=db_run.run_name,
        status="pending",
        current_epoch=0,
        total_epochs=req.max_epochs,
        best_avg_auc=0.0,
        current_conflict_rate=0.0,
        created_at=db_run.created_at
    )


@router.get("/runs", response_model=List[schemas.TrainingRunStatus])
def get_training_runs(db: Session = Depends(get_db)):
    """Retrieve all training runs and their current statuses."""
    runs = db.query(models.TrainingRun).order_by(models.TrainingRun.created_at.desc()).all()
    res = []
    for r in runs:
        latest_metrics = db.query(models.EpochMetrics)\
            .filter(models.EpochMetrics.training_run_id == r.id)\
            .order_by(models.EpochMetrics.epoch.desc()).first()
            
        latest_schema = None
        conflict_rate = 0.0
        if latest_metrics:
            conflict_rate = latest_metrics.conflict_rate
            latest_schema = schemas.EpochMetricsSchema(
                epoch=latest_metrics.epoch,
                train_loss=latest_metrics.train_total_loss,
                val_loss=latest_metrics.val_total_loss,
                avg_val_auc=latest_metrics.avg_val_auc,
                per_task_auc=latest_metrics.val_task_auc,
                conflict_rate=latest_metrics.conflict_rate,
                uncertainty_weights=latest_metrics.uncertainty_weights
            )
            
        res.append(schemas.TrainingRunStatus(
            run_id=r.run_uuid,
            run_name=r.run_name,
            status=r.status,
            current_epoch=r.total_epochs or 0,
            total_epochs=r.config.get("max_epochs", 200),
            best_avg_auc=r.best_avg_auc or 0.0,
            current_conflict_rate=conflict_rate,
            latest_metrics=latest_schema,
            created_at=r.created_at
        ))
    return res


@router.get("/run/{run_uuid}", response_model=schemas.TrainingRunStatus)
def get_training_run_status(run_uuid: str, db: Session = Depends(get_db)):
    """Get status of a specific training run."""
    run = db.query(models.TrainingRun).filter(models.TrainingRun.run_uuid == run_uuid).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
        
    latest_metrics = db.query(models.EpochMetrics)\
        .filter(models.EpochMetrics.training_run_id == run.id)\
        .order_by(models.EpochMetrics.epoch.desc()).first()
        
    latest_schema = None
    conflict_rate = 0.0
    if latest_metrics:
        conflict_rate = latest_metrics.conflict_rate
        latest_schema = schemas.EpochMetricsSchema(
            epoch=latest_metrics.epoch,
            train_loss=latest_metrics.train_total_loss,
            val_loss=latest_metrics.val_total_loss,
            avg_val_auc=latest_metrics.avg_val_auc,
            per_task_auc=latest_metrics.val_task_auc,
            conflict_rate=latest_metrics.conflict_rate,
            uncertainty_weights=latest_metrics.uncertainty_weights
        )
        
    return schemas.TrainingRunStatus(
        run_id=run.run_uuid,
        run_name=run.run_name,
        status=run.status,
        current_epoch=run.total_epochs or 0,
        total_epochs=run.config.get("max_epochs", 200),
        best_avg_auc=run.best_avg_auc or 0.0,
        current_conflict_rate=conflict_rate,
        latest_metrics=latest_schema,
        created_at=run.created_at
    )


@router.get("/run/{run_uuid}/metrics", response_model=List[schemas.EpochMetricsSchema])
def get_training_metrics_history(run_uuid: str, db: Session = Depends(get_db)):
    """Retrieve full epoch history of metrics for a training run."""
    run = db.query(models.TrainingRun).filter(models.TrainingRun.run_uuid == run_uuid).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
        
    metrics = db.query(models.EpochMetrics)\
        .filter(models.EpochMetrics.training_run_id == run.id)\
        .order_by(models.EpochMetrics.epoch.asc()).all()
        
    return [
        schemas.EpochMetricsSchema(
            epoch=m.epoch,
            train_loss=m.train_total_loss,
            val_loss=m.val_total_loss,
            avg_val_auc=m.avg_val_auc,
            per_task_auc=m.val_task_auc,
            conflict_rate=m.conflict_rate,
            uncertainty_weights=m.uncertainty_weights
        ) for m in metrics
    ]


@router.post("/run/{run_uuid}/stop")
def stop_training_run(run_uuid: str, db: Session = Depends(get_db)):
    """Stop an active training run."""
    run = db.query(models.TrainingRun).filter(models.TrainingRun.run_uuid == run_uuid).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
        
    if run.status != "running":
        raise HTTPException(status_code=400, detail="Training run is not currently running")
        
    active_runs[run_uuid] = False
    run.status = "stopped"
    db.commit()
    return {"message": "Stop signal sent successfully"}
