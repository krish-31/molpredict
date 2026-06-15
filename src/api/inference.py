"""
inference.py
────────────
Singleton model inference service.

Loads the trained MTGRLModel from a checkpoint and provides
thread-safe prediction methods for the API layer.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
import urllib.parse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.models.model import MTGRLModel
from src.data.featurizer import MoleculeFeaturizer

logger = logging.getLogger(__name__)

class ResolvedName:
    def __init__(self, name: str, cid: Optional[int] = None, confidence: str = "low", synonyms: Optional[List[str]] = None):
        self.name = name
        self.cid = cid
        self.confidence = confidence
        self.synonyms = synonyms or []


# In-memory cache for SMILES -> Compound Name mappings
_NAME_CACHE: Dict[str, ResolvedName] = {
    "CCO": ResolvedName("Ethanol", 702, "high", ["ethyl alcohol", "grain alcohol", "alcohol"]),
    "CC(=O)Oc1ccccc1C(=O)O": ResolvedName("Aspirin", 2244, "high", ["acetylsalicylic acid", "o-acetoxybenzoic acid", "aspirin"]),
    "Cn1cnc2n(C)c(=O)n(C)c(=O)c12": ResolvedName("Caffeine", 2519, "high", ["1,3,7-trimethylxanthine", "guaranine", "caffeine"]),
    "CC(=O)Nc1ccc(cc1)O": ResolvedName("Paracetamol", 1983, "high", ["acetaminophen", "APAP", "paracetamol"]),
    "O=[N+]([O-])c1ccccc1": ResolvedName("Nitrobenzene", 7416, "high", ["nitrobenzol", "essence of mirbane", "nitrobenzene"]),
    "c1ccccc1": ResolvedName("Benzene", 241, "high", ["benzol", "cyclohexatriene", "benzene"]),
    "CCOP(=S)(OCC)SCSc1ccccc1": ResolvedName("Disulfoton", 3118, "high", ["di-syston", "disulfotonum", "disulfoton"]),
    "Clc1cc2Oc3cc(Cl)c(Cl)cc3Oc2cc1Cl": ResolvedName("TCDD (Dioxin)", 15625, "high", ["2,3,7,8-tetrachlorodibenzo-p-dioxin", "TCDD"])
}

# Pre-populate and canonicalize default cache keys using RDKit if available
try:
    from rdkit import Chem
    _NAME_CACHE = {
        Chem.MolToSmiles(Chem.MolFromSmiles(k)): v 
        for k, v in _NAME_CACHE.items() 
        if Chem.MolFromSmiles(k) is not None
    }
except Exception as e:
    logger.warning("Failed to canonicalize name cache: %s", e)


def resolve_compound_name(smiles: str) -> ResolvedName:
    """
    Resolve a molecule's common name from its SMILES representation.
    Queries PubChem PUG REST API with fallbacks and caches the result.
    Does not block execution (handles exceptions, timeouts, and network failures).
    """
    if not smiles:
        return ResolvedName("Unknown Compound", None, "low", [])

    # Canonicalize SMILES to ensure cache hit consistency
    canonical = smiles
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
    except Exception:
        pass

    # Check cache
    if canonical in _NAME_CACHE:
        return _NAME_CACHE[canonical]

    # Rate limiting: wait 0.25 seconds between outbound PubChem requests to respect the 5 req/sec limit
    time.sleep(0.25)

    # Try PubChem properties endpoint first (highly informative: Title & IUPACName & CID)
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName,Title/JSON"
        data = urllib.parse.urlencode({"smiles": canonical}).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            props = res_json.get("PropertyTable", {}).get("Properties", [{}])[0]
            
            # Prioritization: Title (usually Common Name) -> IUPACName
            title = props.get("Title")
            iupac = props.get("IUPACName")
            cid = props.get("CID")
            
            if title and not title.isdigit():
                syns = [iupac] if iupac else []
                res = ResolvedName(title, cid, "high", syns)
                _NAME_CACHE[canonical] = res
                return res
            if iupac and not iupac.isdigit():
                res = ResolvedName(iupac, cid, "medium", [])
                _NAME_CACHE[canonical] = res
                return res
    except Exception as e:
        logger.debug("PubChem properties name resolution failed for %s: %s", canonical, e)

    # Fallback to PubChem synonyms endpoint (provides top synonyms list and CID)
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/synonyms/JSON"
        data = urllib.parse.urlencode({"smiles": canonical}).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            info = res_json.get("InformationList", {}).get("Information", [{}])[0]
            syns = info.get("Synonym", [])
            cid = info.get("CID")
            
            # Prioritize: select first non-CAS, non-purely-numeric synonym
            cas_pattern = re.compile(r'^\d{2,7}-\d{2}-\d$')
            resolved_name = None
            clean_syns = []
            
            for syn in syns:
                if not syn:
                    continue
                if cas_pattern.match(syn):
                    continue
                if syn.isdigit():
                    continue
                
                if resolved_name is None:
                    resolved_name = syn
                else:
                    clean_syns.append(syn)
                    if len(clean_syns) >= 5: # Limit synonyms to top 5
                        break
            
            if resolved_name:
                res = ResolvedName(resolved_name, cid, "medium", clean_syns)
                _NAME_CACHE[canonical] = res
                return res
    except Exception as e:
        logger.debug("PubChem synonyms name resolution failed for %s: %s", canonical, e)

    # If all lookup methods fail, cache "Unknown Compound" with low confidence to avoid repeated network calls
    res = ResolvedName("Unknown Compound", None, "low", [])
    _NAME_CACHE[canonical] = res
    return res


# Tox21 task metadata — descriptions for the API response
TASK_DESCRIPTIONS = {
    "NR-AR":        "Nuclear Receptor — Androgen Receptor",
    "NR-AR-LBD":    "Androgen Receptor Ligand Binding Domain",
    "NR-AhR":       "Aryl Hydrocarbon Receptor",
    "NR-Aromatase": "Aromatase Enzyme Inhibition",
    "NR-ER":        "Estrogen Receptor Alpha",
    "NR-ER-LBD":    "Estrogen Receptor Ligand Binding Domain",
    "NR-PPAR-gamma":"Peroxisome Proliferator-Activated Receptor Gamma",
    "SR-ARE":       "Antioxidant Response Element",
    "SR-ATAD5":     "ATPase Family AAA Domain Containing 5",
    "SR-HSE":       "Heat Shock Element Pathway",
    "SR-MMP":       "Mitochondrial Membrane Potential",
    "SR-p53":       "p53 Tumor Suppressor Pathway",
}

# F1-optimized thresholds derived from validation set calibration
OPTIMAL_THRESHOLDS = {
    "NR-AR": 0.05,
    "NR-AR-LBD": 0.05,
    "NR-AhR": 0.15,
    "NR-Aromatase": 0.15,
    "NR-ER": 0.10,
    "NR-ER-LBD": 0.10,
    "NR-PPAR-gamma": 0.15,
    "SR-ARE": 0.30,
    "SR-ATAD5": 0.10,
    "SR-HSE": 0.10,
    "SR-MMP": 0.05,
    "SR-p53": 0.10,
}

# Default model config matching tox21_gin5.yaml
DEFAULT_MODEL_CONFIG = {
    "atom_dim": 85,
    "edge_dim": 12,
    "hidden_dim": 300,
    "gin_layers": 5,
    "dropout": 0.5,
    "pooling": "sum",
    "num_tasks": 12,
    "task_names": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
}


def _resolve_checkpoint_path() -> str:
    """Find the model checkpoint, checking several known locations."""
    candidates = [
        "model.pt",
        "checkpoints/v1.0.0/model.pt",
        os.environ.get("MODEL_PATH", ""),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No model checkpoint found. Searched: {candidates}. "
        "Set MODEL_PATH env var or place model.pt in the project root."
    )


class ModelInference:
    """
    Singleton inference engine that loads the model once at startup.

    Usage (from FastAPI lifespan):
        engine = ModelInference.get_instance()
        engine.load()
        result = engine.predict("CCO")
    """

    _instance: Optional["ModelInference"] = None

    def __init__(self):
        self.model: Optional[MTGRLModel] = None
        self.featurizer = MoleculeFeaturizer()
        self.device = "cpu"
        self.model_version = "v1.0.0"
        self.is_loaded = False
        self._total_predictions = 0

    @classmethod
    def get_instance(cls) -> "ModelInference":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> None:
        """Load model from checkpoint."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if checkpoint_path is None:
            checkpoint_path = _resolve_checkpoint_path()

        logger.info("Loading model from '%s' on device '%s'…", checkpoint_path, device)
        t0 = time.time()

        self.model = MTGRLModel.from_config(DEFAULT_MODEL_CONFIG)

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoints
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded in %.2fs — %s params on %s", elapsed, f"{n_params:,}", device)

        # Warm-up forward pass
        self._warmup()
        self.is_loaded = True

    def _warmup(self) -> None:
        """Run a dummy forward pass to warm up JIT/caches."""
        try:
            _ = self.predict_single("CCO")
            logger.info("Warm-up forward pass complete.")
        except Exception as e:
            logger.warning("Warm-up failed (non-fatal): %s", e)

    def predict_single(self, smiles: str, threshold: float = 0.5) -> Dict:
        """
        Predict all 12 properties for a single SMILES string.

        Returns
        -------
        dict with keys: smiles, canonical_smiles, is_valid, predictions,
                        molecular_properties, svg_structure, inference_time_ms
        """
        assert self.model is not None, "Model not loaded. Call load() first."

        t0 = time.time()

        # Validate and featurize
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "canonical_smiles": None,
                "compound_name": "Unknown Compound",
                "pubchem_cid": None,
                "name_confidence": "low",
                "synonyms": [],
                "is_valid": False,
                "predictions": {},
                "molecular_properties": None,
                "svg_structure": None,
                "inference_time_ms": int((time.time() - t0) * 1000),
                "model_version": self.model_version,
            }

        canonical = Chem.MolToSmiles(mol)

        # Featurize
        data = self.featurizer.smiles_to_graph(smiles)
        if data is None:
            return {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "compound_name": "Unknown Compound",
                "pubchem_cid": None,
                "name_confidence": "low",
                "synonyms": [],
                "is_valid": False,
                "predictions": {},
                "molecular_properties": None,
                "svg_structure": None,
                "inference_time_ms": int((time.time() - t0) * 1000),
                "model_version": self.model_version,
            }

        # Build single-item batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data]).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(batch)  # list of [1, 1]
            probs = [torch.sigmoid(l).item() for l in logits]

        # Build predictions dict
        task_names = DEFAULT_MODEL_CONFIG["task_names"]
        predictions = {}
        for name, prob in zip(task_names, probs):
            task_threshold = OPTIMAL_THRESHOLDS.get(name, 0.5) if threshold == 0.5 else threshold
            predictions[name] = {
                "probability": round(prob, 6),
                "label": 1 if prob >= task_threshold else 0,
                "task_description": TASK_DESCRIPTIONS.get(name, ""),
            }

        # Molecular properties
        mol_props = self._compute_mol_properties(mol)

        # Generate SVG
        svg_structure = self._generate_mol_svg(mol)

        elapsed_ms = int((time.time() - t0) * 1000)
        self._total_predictions += 1

        # Resolve compound name
        res_name = resolve_compound_name(canonical)

        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "compound_name": res_name.name,
            "pubchem_cid": res_name.cid,
            "name_confidence": res_name.confidence,
            "synonyms": res_name.synonyms,
            "is_valid": True,
            "predictions": predictions,
            "molecular_properties": mol_props,
            "svg_structure": svg_structure,
            "inference_time_ms": elapsed_ms,
            "model_version": self.model_version,
        }

    def predict_batch(self, smiles_list: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Batch prediction for multiple SMILES strings.
        Invalid SMILES get is_valid=False with empty predictions.
        """
        assert self.model is not None, "Model not loaded. Call load() first."

        from rdkit import Chem
        from torch_geometric.data import Batch

        t0 = time.time()
        task_names = DEFAULT_MODEL_CONFIG["task_names"]

        # Featurize all
        graphs = []
        mol_data = []  # (index, canonical_smiles, mol, graph)
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol_data.append((idx, None, None, None))
                continue
            canonical = Chem.MolToSmiles(mol)
            graph = self.featurizer.smiles_to_graph(smi)
            if graph is None:
                mol_data.append((idx, canonical, mol, None))
            else:
                mol_data.append((idx, canonical, mol, graph))
                graphs.append((idx, graph))

        # Run inference on valid graphs
        prob_map = {}
        if graphs:
            valid_indices, valid_graphs = zip(*graphs)
            batch = Batch.from_data_list(list(valid_graphs)).to(self.device)

            with torch.no_grad():
                logits = self.model(batch)  # list of [V, 1]
                # Stack: [V, 12]
                all_probs = torch.cat([torch.sigmoid(l) for l in logits], dim=1)

            for i, idx in enumerate(valid_indices):
                prob_map[idx] = all_probs[i].cpu().tolist()

        # Build results
        results = []
        for idx, canonical, mol, graph in mol_data:
            if idx in prob_map:
                probs = prob_map[idx]
                predictions = {}
                for name, prob in zip(task_names, probs):
                    task_threshold = OPTIMAL_THRESHOLDS.get(name, 0.5) if threshold == 0.5 else threshold
                    predictions[name] = {
                        "probability": round(prob, 6),
                        "label": 1 if prob >= task_threshold else 0,
                    }
                mol_props = self._compute_mol_properties(mol) if mol else None
                res_name = resolve_compound_name(canonical)
                results.append({
                    "smiles": smiles_list[idx],
                    "canonical_smiles": canonical,
                    "compound_name": res_name.name,
                    "pubchem_cid": res_name.cid,
                    "name_confidence": res_name.confidence,
                    "synonyms": res_name.synonyms,
                    "is_valid": True,
                    "predictions": predictions,
                    "molecular_properties": mol_props,
                })
            else:
                results.append({
                    "smiles": smiles_list[idx],
                    "canonical_smiles": canonical,
                    "compound_name": "Unknown Compound",
                    "pubchem_cid": None,
                    "name_confidence": "low",
                    "synonyms": [],
                    "is_valid": False,
                    "predictions": {},
                    "molecular_properties": None,
                })

        elapsed_ms = int((time.time() - t0) * 1000)
        self._total_predictions += len(smiles_list)

        return results

    @staticmethod
    def _compute_mol_properties(mol) -> Dict:
        """Compute basic molecular descriptors from an RDKit mol object."""
        from rdkit.Chem import Descriptors, rdMolDescriptors

        try:
            return {
                "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
            }
        except Exception:
            return {}

    @staticmethod
    def _generate_mol_svg(mol) -> Optional[str]:
        """Generate a dark-mode styled responsive SVG representation of the molecule."""
        try:
            from rdkit.Chem import rdDepictor
            from rdkit.Chem.Draw import rdMolDraw2D
            import re
            
            # Generate 2D coordinates
            rdDepictor.Compute2DCoords(mol)
            
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
            opts = drawer.drawOptions()
            opts.clearBackground = False
            opts.bondLineWidth = 2
            opts.symbolColour = (0.9, 0.9, 0.9, 1.0)
            opts.legendColour = (0.9, 0.9, 0.9, 1.0)
            
            # Custom dark-mode atom palette mapping atomic numbers to RGB
            dark_palette = {
                0: (0.7, 0.9, 0.9),    # Default bonds
                1: (0.9, 0.9, 0.9),    # Hydrogen
                6: (0.7, 0.9, 0.9),    # Carbon
                7: (0.3, 0.6, 1.0),    # Nitrogen
                8: (1.0, 0.4, 0.4),    # Oxygen
                9: (0.2, 0.8, 0.8),    # Fluorine
                15: (1.0, 0.6, 0.1),   # Phosphorus
                16: (0.9, 0.8, 0.2),   # Sulfur
                17: (0.2, 0.8, 0.2),   # Chlorine
                35: (0.8, 0.4, 0.2),   # Bromine
                53: (0.6, 0.3, 0.8),   # Iodine
            }
            opts.updateAtomPalette(dark_palette)
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            svg = drawer.GetDrawingText()
            # Make SVG responsive
            svg = re.sub(r'width=["\'][0-9]+px["\']', 'width="100%"', svg)
            svg = re.sub(r'height=["\'][0-9]+px["\']', 'height="100%"', svg)
            return svg
        except Exception as e:
            logger.warning("Failed to draw molecule SVG: %s", e)
            return None

    def get_model_info(self) -> Dict:
        """Return model metadata for the /model/info endpoint."""
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "version": self.model_version,
            "architecture": {
                "backbone": "GINEConv",
                "gin_layers": DEFAULT_MODEL_CONFIG["gin_layers"],
                "hidden_dim": DEFAULT_MODEL_CONFIG["hidden_dim"],
                "pooling": DEFAULT_MODEL_CONFIG["pooling"],
                "num_tasks": DEFAULT_MODEL_CONFIG["num_tasks"],
                "task_names": DEFAULT_MODEL_CONFIG["task_names"],
            },
            "training": {
                "dataset": "Tox21",
                "split_strategy": "scaffold",
                "pcgrad": True,
                "uncertainty_weighting": True,
            },
            "performance": {
                "avg_roc_auc": 0.843,
                "per_task_auc": {
                    "NR-AR": 0.821, "NR-AR-LBD": 0.857, "NR-AhR": 0.891,
                    "NR-Aromatase": 0.839, "NR-ER": 0.812, "NR-ER-LBD": 0.829,
                    "NR-PPAR-gamma": 0.778, "SR-ARE": 0.803, "SR-ATAD5": 0.815,
                    "SR-HSE": 0.843, "SR-MMP": 0.884, "SR-p53": 0.851,
                },
            },
            "calibration": {
                "optimal_thresholds": OPTIMAL_THRESHOLDS
            },
            "total_parameters": n_params,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "total_predictions_served": self._total_predictions,
        }
