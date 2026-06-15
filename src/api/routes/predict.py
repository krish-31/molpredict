import os
import uuid
import time
import pandas as pd
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from rdkit import Chem

from src.api.database import get_db
from src.api.inference import ModelInference
from src.api import models, schemas

router = APIRouter()

# Helper function to get or create a molecule in the DB
def get_or_create_molecule(db: Session, smiles: str) -> Optional[models.Molecule]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    canonical_smiles = Chem.MolToSmiles(mol)
    
    # Check if molecule already exists
    db_mol = db.query(models.Molecule).filter(models.Molecule.canonical_smiles == canonical_smiles).first()
    if db_mol:
        return db_mol
    
    # Compute basic molecular properties
    from rdkit.Chem import Descriptors, rdMolDescriptors
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = round(Descriptors.MolWt(mol), 2)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        logp = round(Descriptors.MolLogP(mol), 2)
        tpsa = round(Descriptors.TPSA(mol), 2)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
    except Exception:
        formula, mw, num_atoms, num_bonds, hbd, hba, logp, tpsa, num_rings = (
            None, None, None, None, None, None, None, None, None
        )
        
    # Scaffold calculation
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
    except Exception:
        scaffold_smiles = None

    # InChI & InChIKey
    try:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchi, inchikey = None, None

    db_mol = models.Molecule(
        smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi,
        inchikey=inchikey,
        molecular_formula=formula,
        molecular_weight=mw,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        num_rings=num_rings,
        hbd=hbd,
        hba=hba,
        logp=logp,
        tpsa=tpsa,
        murcko_scaffold=scaffold_smiles,
        source_dataset="user_upload"
    )
    db.add(db_mol)
    try:
        db.commit()
        db.refresh(db_mol)
    except Exception:
        db.rollback()
        db_mol = db.query(models.Molecule).filter(models.Molecule.canonical_smiles == canonical_smiles).first()
    return db_mol


def get_production_model_version(db: Session) -> models.ModelVersion:
    # Check if we have model registry entry
    db_model = db.query(models.ModelVersion).filter(models.ModelVersion.is_production == True).first()
    if not db_model:
        # Create default registry entry on the fly
        db_model = models.ModelVersion(
            version_tag="v1.0.0",
            description="Trained GIN-5L multi-task model",
            architecture=ModelInference.get_instance().get_model_info()["architecture"],
            training_config={},
            mtl_config={},
            checkpoint_path="model.pt",
            avg_roc_auc=0.843,
            is_production=True
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
    return db_model


@router.post("/predict", response_model=schemas.PredictResponse)
def predict_single(req: schemas.PredictRequest, db: Session = Depends(get_db)):
    """Run GIN model inference on a single SMILES string."""
    engine = ModelInference.get_instance()
    if not engine.is_loaded:
        engine.load()

    # Predict via inference service
    pred_res = engine.predict_single(req.smiles, req.threshold)
    
    # Store request logs in database if valid
    db_mol = None
    if pred_res["is_valid"]:
        db_mol = get_or_create_molecule(db, req.smiles)
        
    db_model = get_production_model_version(db)
    
    # Save PredictionRequest log
    request_uuid = str(uuid.uuid4())
    db_request = models.PredictionRequest(
        request_uuid=request_uuid,
        molecule_id=db_mol.id if db_mol else None,
        model_version_id=db_model.id,
        input_smiles=req.smiles,
        is_valid_smiles=pred_res["is_valid"],
        predictions=pred_res["predictions"] if pred_res["is_valid"] else None,
        inference_time_ms=pred_res["inference_time_ms"],
        source="web"
    )
    db.add(db_request)
    db.commit()
    
    # Construct response
    predictions_dict = {}
    if pred_res["is_valid"]:
        for name, task_pred in pred_res["predictions"].items():
            predictions_dict[name] = schemas.TaskPrediction(
                probability=task_pred["probability"],
                label=task_pred["label"],
                task_description=task_pred["task_description"]
            )
            
    mol_props = None
    if pred_res["is_valid"] and pred_res["molecular_properties"]:
        props = pred_res["molecular_properties"]
        mol_props = schemas.MolecularProperties(
            molecular_formula=props.get("molecular_formula"),
            molecular_weight=props.get("molecular_weight"),
            num_atoms=props.get("num_atoms"),
            num_bonds=props.get("num_bonds"),
            hbd=props.get("hbd"),
            hba=props.get("hba"),
            logp=props.get("logp"),
            tpsa=props.get("tpsa"),
            num_rings=props.get("num_rings")
        )

    props_dict = pred_res.get("molecular_properties")
    formula = props_dict.get("molecular_formula") if props_dict else None
    mw = props_dict.get("molecular_weight") if props_dict else None

    return schemas.PredictResponse(
        request_id=request_uuid,
        smiles=req.smiles,
        canonical_smiles=pred_res["canonical_smiles"],
        compound_name=pred_res.get("compound_name"),
        formula=formula,
        molecular_weight=mw,
        pubchem_cid=pred_res.get("pubchem_cid"),
        name_confidence=pred_res.get("name_confidence"),
        synonyms=pred_res.get("synonyms"),
        is_valid=pred_res["is_valid"],
        molecular_properties=mol_props,
        predictions=predictions_dict,
        model_version=pred_res["model_version"],
        inference_time_ms=pred_res["inference_time_ms"],
        svg_structure=pred_res.get("svg_structure")
    )


# ── Batch Prediction Background Process ──────────────────────────────

def run_batch_job_task(job_uuid: str, input_file_path: str, output_file_path: str, threshold: float, db_session_factory):
    db: Session = db_session_factory()
    try:
        job = db.query(models.BatchJob).filter(models.BatchJob.job_uuid == job_uuid).first()
        if not job:
            return
        
        job.status = "processing"
        job.started_at = datetime.utcnow()
        db.commit()
        
        df = pd.read_csv(input_file_path)
        # Find column containing smiles (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.lower().strip() == "smiles":
                smiles_col = col
                break
        
        if smiles_col is None:
            raise ValueError("CSV must contain a 'smiles' column")
            
        smiles_list = df[smiles_col].astype(str).tolist()
        job.total_molecules = len(smiles_list)
        db.commit()
        
        engine = ModelInference.get_instance()
        if not engine.is_loaded:
            engine.load()
            
        # We can chunk predictions for massive files or run in single batch if reasonable
        chunk_size = 200
        results = []
        
        processed_count = 0
        failed_count = 0
        
        for idx in range(0, len(smiles_list), chunk_size):
            chunk = smiles_list[idx:idx+chunk_size]
            chunk_results = engine.predict_batch(chunk, threshold)
            
            for smi_idx, res in enumerate(chunk_results):
                global_idx = idx + smi_idx
                raw_smiles = chunk[smi_idx]
                
                # Update job counters
                if res["is_valid"]:
                    processed_count += 1
                else:
                    failed_count += 1
                
                # Save request log inside database (optional but recommended in schema)
                # To prevent DB bottleneck, we save molecules in batch if desired, or skip database molecule persistence for large batches
                
            results.extend(chunk_results)
            
            # Periodically update progress in DB
            job.processed = processed_count
            job.failed = failed_count
            db.commit()
            
        # Generate output CSV
        output_rows = []
        from src.api.inference import OPTIMAL_THRESHOLDS
        for res in results:
            row = {
                "smiles": res["smiles"],
                "compound_name": res.get("compound_name", "Unknown Compound"),
                "pubchem_cid": res.get("pubchem_cid"),
                "name_confidence": res.get("name_confidence", "low"),
                "synonyms": "; ".join(res.get("synonyms", [])) if res.get("synonyms") else "",
                "is_valid": res["is_valid"]
            }
            if res["is_valid"]:
                # Expose predictions
                for name, pred in res["predictions"].items():
                    row[name] = pred["probability"]
                
                # Expose molecular properties
                props = res["molecular_properties"]
                if props:
                    row["formula"] = props.get("molecular_formula")
                    row["molecular_weight"] = props.get("molecular_weight")
                    for prop, val in props.items():
                        row[prop] = val
                else:
                    row["formula"] = None
                    row["molecular_weight"] = None

                # Compute flagged count and max probability
                flagged_count = 0
                max_prob = 0.0
                for name, pred in res["predictions"].items():
                    prob = pred["probability"]
                    task_threshold = OPTIMAL_THRESHOLDS.get(name, 0.5) if threshold == 0.5 else threshold
                    if prob >= task_threshold:
                        flagged_count += 1
                    if prob > max_prob:
                        max_prob = prob
                
                # Risk level classification
                if flagged_count >= 3 or max_prob >= 0.70:
                    risk_level = "HIGH"
                elif flagged_count >= 1:
                    risk_level = "MODERATE"
                else:
                    risk_level = "LOW"
                
                row["flagged_endpoints"] = flagged_count
                row["risk_level"] = risk_level
            else:
                row["formula"] = None
                row["molecular_weight"] = None
                row["flagged_endpoints"] = 0
                row["risk_level"] = "LOW"
            output_rows.append(row)
            
        out_df = pd.DataFrame(output_rows)
        out_df.to_csv(output_file_path, index=False)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        db.rollback()
        job = db.query(models.BatchJob).filter(models.BatchJob.job_uuid == job_uuid).first()
        if job:
            job.status = "failed"
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/batch", response_model=schemas.BatchJobStatus)
async def upload_batch_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    db: Session = Depends(get_db)
):
    """Upload a CSV with a 'smiles' column to process in background."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
    job_uuid = str(uuid.uuid4())
    
    # Save input file to filesystem
    upload_dir = "storage/uploads/batch_jobs"
    os.makedirs(upload_dir, exist_ok=True)
    
    input_file_path = os.path.join(upload_dir, f"{job_uuid}_input.csv")
    output_file_path = os.path.join(upload_dir, f"{job_uuid}_output.csv")
    
    with open(input_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    db_model = get_production_model_version(db)
    
    # Create batch job entry
    db_job = models.BatchJob(
        job_uuid=job_uuid,
        model_version_id=db_model.id,
        status="pending",
        total_molecules=0,
        processed=0,
        failed=0,
        input_file_path=input_file_path,
        output_file_path=output_file_path
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Dispatch background task
    from src.api.database import SessionLocal
    background_tasks.add_task(
        run_batch_job_task,
        job_uuid,
        input_file_path,
        output_file_path,
        threshold,
        SessionLocal
    )
    
    return schemas.BatchJobStatus(
        job_id=job_uuid,
        status="pending",
        total_molecules=0,
        processed=0,
        failed=0,
        progress_pct=0.0,
        created_at=db_job.created_at
    )


@router.get("/batch/{job_uuid}", response_model=schemas.BatchJobStatus)
def get_batch_job_status(job_uuid: str, db: Session = Depends(get_db)):
    """Retrieve the progress and status of a batch job."""
    job = db.query(models.BatchJob).filter(models.BatchJob.job_uuid == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
        
    progress = 0.0
    if job.total_molecules and job.total_molecules > 0:
        progress = ((job.processed + job.failed) / job.total_molecules) * 100.0
        
    download_url = None
    if job.status == "completed":
        download_url = f"/api/predict/batch/{job_uuid}/download"
        
    # Rough estimate of remaining time (15ms per molecule)
    eta = None
    if job.status == "processing" and job.total_molecules:
        remaining = job.total_molecules - (job.processed + job.failed)
        eta = int(remaining * 0.015)  # 15ms per molecule
        
    return schemas.BatchJobStatus(
        job_id=job_uuid,
        status=job.status,
        total_molecules=job.total_molecules or 0,
        processed=job.processed or 0,
        failed=job.failed or 0,
        progress_pct=round(progress, 2),
        eta_seconds=eta,
        download_url=download_url,
        created_at=job.created_at
    )


@router.get("/batch/{job_uuid}/download")
def download_batch_results(job_uuid: str, db: Session = Depends(get_db)):
    """Download the output prediction CSV for a completed batch job."""
    job = db.query(models.BatchJob).filter(models.BatchJob.job_uuid == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
        
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Batch job is not completed yet")
        
    if not job.output_file_path or not os.path.exists(job.output_file_path):
        raise HTTPException(status_code=404, detail="Results output file not found")
        
    return FileResponse(
        path=job.output_file_path,
        media_type="text/csv",
        filename=f"batch_results_{job_uuid[:8]}.csv"
    )


@router.get("/batch/{job_uuid}/preview")
def get_batch_job_preview(job_uuid: str, db: Session = Depends(get_db)):
    """Get preview of first 10 results in batch predictions."""
    job = db.query(models.BatchJob).filter(models.BatchJob.job_uuid == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
        
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Batch job is not completed yet")
        
    if not job.output_file_path or not os.path.exists(job.output_file_path):
        raise HTTPException(status_code=404, detail="Results output file not found")
        
    try:
        df = pd.read_csv(job.output_file_path)
        records = df.to_dict(orient="records")
        clean_records = []
        for r in records:
            clean_row = {
                k: (None if pd.isna(v) else v)
                for k, v in r.items()
            }
            clean_records.append(clean_row)
        return clean_records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

