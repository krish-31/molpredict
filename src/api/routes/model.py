from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api.database import get_db
from src.api.inference import ModelInference
from src.api import models

router = APIRouter()

@router.get("/info")
def get_model_info():
    """Get metadata, parameters and metrics for the currently loaded model version."""
    engine = ModelInference.get_instance()
    if not engine.is_loaded:
         # Attempt to load the model if not loaded
         try:
             engine.load()
         except Exception as e:
             raise HTTPException(status_code=503, detail=f"Model not loaded and auto-load failed: {str(e)}")
    return engine.get_model_info()

@router.get("/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    """Retrieve historical training/evaluation runs and metrics."""
    # Query database for all training runs completed
    runs = db.query(models.TrainingRun).filter(models.TrainingRun.status == "completed").all()
    if not runs:
        # Fallback to returning standard trained model performance from the active engine
        engine = ModelInference.get_instance()
        if not engine.is_loaded:
             engine.load()
        info = engine.get_model_info()
        return {
            "source": "fallback_in_memory",
            "best_run": {
                "run_name": "tox21_gin5_production",
                "avg_roc_auc": info["performance"]["avg_roc_auc"],
                "per_task_auc": info["performance"]["per_task_auc"],
                "dataset": "tox21",
                "split_strategy": "scaffold"
            }
        }

    return [
        {
            "run_id": r.run_uuid,
            "run_name": r.run_name,
            "best_avg_auc": r.best_avg_auc,
            "best_epoch": r.best_epoch,
            "dataset": r.dataset,
            "split_strategy": r.split_strategy,
            "completed_at": r.completed_at
        } for r in runs
    ]
