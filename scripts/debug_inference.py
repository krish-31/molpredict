import sys
import os
import torch
from rdkit import Chem
from torch_geometric.data import Batch

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model import MTGRLModel
from src.data.featurizer import MoleculeFeaturizer
from src.api.inference import ModelInference, DEFAULT_MODEL_CONFIG

def audit_checkpoint():
    print("=" * 60)
    print("1. MODEL CHECKPOINT AUDIT")
    print("=" * 60)
    checkpoint_path = "model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file '{checkpoint_path}' not found!")
        return False
        
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle wrapping if exists
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        print("Detected wrapper dict. Extracting 'model_state_dict'...")
        state_dict = state_dict["model_state_dict"]
        
    print(f"Checkpoint state_dict contains {len(state_dict)} tensors.")
    
    # Initialize the runtime model
    print("Initializing runtime model from DEFAULT_MODEL_CONFIG...")
    model = MTGRLModel.from_config(DEFAULT_MODEL_CONFIG)
    model.eval()
    model_state = model.state_dict()
    
    # Compare keys
    model_keys = set(model_state.keys())
    checkpoint_keys = set(state_dict.keys())
    
    missing_in_checkpoint = model_keys - checkpoint_keys
    extra_in_checkpoint = checkpoint_keys - model_keys
    common_keys = model_keys & checkpoint_keys
    
    print(f"Model keys: {len(model_keys)} | Checkpoint keys: {len(checkpoint_keys)}")
    print(f"Common keys matching: {len(common_keys)}")
    if missing_in_checkpoint:
        print(f"WARNING: {len(missing_in_checkpoint)} keys missing in checkpoint! Model will have random weights for these:")
        print(list(missing_in_checkpoint)[:10])
    else:
        print("SUCCESS: Zero keys missing from checkpoint.")
        
    if extra_in_checkpoint:
        print(f"INFO: {len(extra_in_checkpoint)} extra keys in checkpoint (ignored):")
        print(list(extra_in_checkpoint)[:10])
        
    # Check for shape mismatches
    shape_mismatches = []
    for k in common_keys:
        if model_state[k].shape != state_dict[k].shape:
            shape_mismatches.append((k, model_state[k].shape, state_dict[k].shape))
            
    if shape_mismatches:
        print(f"ERROR: {len(shape_mismatches)} shape mismatches found between model and checkpoint!")
        for name, m_shape, cp_shape in shape_mismatches:
            print(f"  - {name}: model shape {m_shape} vs checkpoint shape {cp_shape}")
    else:
        print("SUCCESS: All matching parameter shapes are identical.")
        
    return True

def audit_inference_pipeline():
    print("\n" + "=" * 60)
    print("2. INFERENCE PIPELINE & DIAGNOSTICS AUDIT")
    print("=" * 60)
    
    # Let's manually run step-by-step inference to trace predictions, logits, and sigmoids.
    engine = ModelInference.get_instance()
    if not engine.is_loaded:
        engine.load()
        
    model = engine.model
    featurizer = engine.featurizer
    
    test_molecules = {
        "Nitrobenzene": "O=[N+]([O-])c1ccccc1",
        "Benzene": "c1ccccc1",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
        "Disulfoton": "CCOP(=S)(OCC)SCSc1ccccc1"
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n--- Molecule: {name} (SMILES: {smiles}) ---")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ERROR: RDKit failed to parse SMILES!")
            continue
            
        data = featurizer.smiles_to_graph(smiles)
        if data is None:
            print("  ERROR: Featurization failed!")
            continue
            
        print(f"  Graph Stats: Nodes (atoms)={data.x.shape[0]}, Edges (bonds*2)={data.edge_index.shape[1]}")
        print(f"  Atom Features shape: {data.x.shape} | Edge Features shape: {data.edge_attr.shape}")
        
        # Verify node features aren't all zero/truncated incorrectly
        non_zero_feats = (data.x != 0).sum().item()
        total_feats = data.x.numel()
        print(f"  Non-zero node features: {non_zero_feats}/{total_feats} ({non_zero_feats/total_feats:.2%})")
        
        # Check active training layers
        training_layers = [m for m in model.modules() if getattr(m, 'training', False) and hasattr(m, 'reset_parameters')]
        # e.g., check dropout or batchnorm training states
        dropout_active = any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
        batchnorm_active = any(m.training for m in model.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)))
        print(f"  Model eval state check: eval_mode={not model.training} | Dropout active={dropout_active} | BatchNorm active={batchnorm_active}")
        
        # Manually run the forward pass to get raw logits and sigmoids
        batch = Batch.from_data_list([data]).to(engine.device)
        with torch.no_grad():
            # Get logits from backbone + task heads
            logits = model(batch)  # list of 12 tensors of shape [1, 1]
            raw_logits = [l.item() for l in logits]
            probs = [torch.sigmoid(l).item() for l in logits]
            
        print("  Task-by-Task Predictions:")
        task_names = DEFAULT_MODEL_CONFIG["task_names"]
        for idx, (task, logit, prob) in enumerate(zip(task_names, raw_logits, probs)):
            print(f"    - {task:<15} | Logit: {logit:+.4f} | Prob: {prob:.4f} ({prob:.1%})")
            
        # Analyze clustering
        avg_prob = sum(probs) / len(probs)
        std_prob = (sum((p - avg_prob) ** 2 for p in probs) / len(probs)) ** 0.5
        print(f"  Clustering statistics: Mean={avg_prob:.4f}, Std={std_prob:.4f}")

def main():
    audit_checkpoint()
    audit_inference_pipeline()

if __name__ == "__main__":
    main()
