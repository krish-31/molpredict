import sys
from unittest.mock import MagicMock

# ── Mock RDKit descriptor modules ─────────────────────────────────────
mock_descriptors = MagicMock()
mock_descriptors.MolWt.return_value = 46.07
mock_descriptors.NumHDonors.return_value = 1
mock_descriptors.NumHAcceptors.return_value = 1
mock_descriptors.MolLogP.return_value = -0.17
mock_descriptors.TPSA.return_value = 20.23

mock_rdMolDescriptors = MagicMock()
mock_rdMolDescriptors.CalcMolFormula.return_value = "C2H6O"
mock_rdMolDescriptors.CalcNumRings.return_value = 0

mock_scaffold_module = MagicMock()
mock_scaffold_module.GetScaffoldForMol.return_value = MagicMock()

# Mock Mol object methods
mock_mol = MagicMock()
mock_mol.GetNumAtoms.return_value = 9
mock_mol.GetNumBonds.return_value = 8

# Mock Chem module
mock_chem = MagicMock()
mock_chem.MolFromSmiles.return_value = mock_mol
mock_chem.MolToSmiles.return_value = "CCO"
mock_chem.MolToInchi.return_value = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
mock_chem.MolToInchiKey.return_value = "LFQSCRIOHQLZFS-UHFFFAOYSA-N"
mock_chem.Descriptors = mock_descriptors
mock_chem.rdMolDescriptors = mock_rdMolDescriptors

# Mock rdkit parent module
mock_rdkit = MagicMock()
mock_rdkit.Chem = mock_chem

# Register all mocks in sys.modules
sys.modules['rdkit'] = mock_rdkit
sys.modules['rdkit.Chem'] = mock_chem
sys.modules['rdkit.Chem.Descriptors'] = mock_descriptors
sys.modules['rdkit.Chem.rdMolDescriptors'] = mock_rdMolDescriptors
sys.modules['rdkit.Chem.Scaffolds'] = mock_scaffold_module
sys.modules['rdkit.Chem.Scaffolds.MurckoScaffold'] = mock_scaffold_module

# Mock ModelInference instance to bypass GIN model execution
mock_inference_instance = MagicMock()
mock_inference_instance.is_loaded = True
mock_inference_instance.predict_single.return_value = {
    "smiles": "CCO",
    "canonical_smiles": "CCO",
    "is_valid": True,
    "predictions": {
        "NR-AR": {"probability": 0.15, "label": 0, "task_description": "Androgen Receptor"},
        "NR-AhR": {"probability": 0.65, "label": 1, "task_description": "Aryl Hydrocarbon Receptor"}
    },
    "molecular_properties": {
        "molecular_formula": "C2H6O",
        "molecular_weight": 46.07,
        "num_atoms": 9,
        "num_bonds": 8,
        "hbd": 1,
        "hba": 1,
        "logp": -0.17,
        "tpsa": 20.23,
        "num_rings": 0
    },
    "inference_time_ms": 12,
    "model_version": "v1.0.0"
}
mock_inference_instance.get_model_info.return_value = {
    "version": "v1.0.0",
    "architecture": {"backbone": "GINEConv", "gin_layers": 5, "hidden_dim": 300},
    "training": {"dataset": "Tox21", "split_strategy": "scaffold"},
    "performance": {"avg_roc_auc": 0.843, "per_task_auc": {}},
    "total_parameters": 4200000,
    "device": "cpu",
    "is_loaded": True,
    "total_predictions_served": 5
}

# Create and register mock module for src.api.inference
mock_inference_module = MagicMock()
mock_inference_module.ModelInference.get_instance.return_value = mock_inference_instance
sys.modules['src.api.inference'] = mock_inference_module

# Now import main FastAPI app and database components safely
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.database import Base, engine

# Initialize database tables for testing
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_get_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0"
    assert data["architecture"]["backbone"] == "GINEConv"

def test_get_model_metrics():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    # Accept both list of runs or fallback dict representation
    if isinstance(data, dict):
        assert "best_run" in data
        assert data["best_run"]["run_name"] == "tox21_gin5_production"
    else:
        assert isinstance(data, list)

def test_predict_single():
    payload = {
        "smiles": "CCO",
        "threshold": 0.5
    }
    response = client.post("/api/predict/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["canonical_smiles"] == "CCO"
    assert "predictions" in data
    assert data["predictions"]["NR-AhR"]["label"] == 1

def test_predict_single_invalid():
    # Setup mock to return invalid prediction
    mock_inference_instance.predict_single.return_value = {
        "smiles": "ZZZ",
        "canonical_smiles": None,
        "is_valid": False,
        "predictions": {},
        "molecular_properties": None,
        "inference_time_ms": 1,
        "model_version": "v1.0.0"
    }
    
    payload = {
        "smiles": "ZZZ",
        "threshold": 0.5
    }
    response = client.post("/api/predict/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["canonical_smiles"] is None

def test_start_training_run():
    payload = {
        "run_name": "test_run",
        "max_epochs": 10,
        "batch_size": 32
    }
    response = client.post("/api/train/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["run_name"] == "test_run"
    assert data["status"] == "pending"
    assert "run_id" in data

def test_get_training_runs():
    response = client.get("/api/train/runs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["run_name"] == "test_run"
