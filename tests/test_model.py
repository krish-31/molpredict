"""test_model.py — Unit tests for MTGRLModel forward pass and inference."""
import pytest
import torch
from torch_geometric.data import Data, Batch
from src.models.model import MTGRLModel


def _dummy_batch(batch_size: int = 4, num_atoms: int = 10) -> Batch:
    """Create a synthetic PyG Batch for testing (no real chemistry)."""
    graphs = []
    for _ in range(batch_size):
        n = num_atoms
        e = n * 2  # roughly 2 edges per atom
        data = Data(
            x         = torch.randn(n, 85),
            edge_index= torch.randint(0, n, (2, e)),
            edge_attr = torch.randn(e, 12),
            y         = torch.randint(-1, 2, (12,)),  # 0/1/-1 labels
        )
        graphs.append(data)
    return Batch.from_data_list(graphs)


class TestMTGRLModel:
    def setup_method(self):
        self.model = MTGRLModel(
            atom_dim=85, edge_dim=12, hidden_dim=64,   # small for speed
            num_layers=2, dropout=0.0, num_tasks=12,
        )
        self.model.eval()

    def test_forward_returns_12_logits(self):
        batch = _dummy_batch(batch_size=4)
        logits = self.model(batch)
        assert len(logits) == 12, "Should return 12 task logits"

    def test_logit_shapes(self):
        batch = _dummy_batch(batch_size=4)
        logits = self.model(batch)
        for i, l in enumerate(logits):
            assert l.shape == (4, 1), f"Task {i}: expected [4,1], got {l.shape}"

    def test_no_nan_in_logits(self):
        batch = _dummy_batch(batch_size=4)
        logits = self.model(batch)
        for i, l in enumerate(logits):
            assert not torch.isnan(l).any(), f"NaN in task {i} logits"

    def test_parameter_count_positive(self):
        n = self.model.count_parameters()
        assert n > 0

    def test_predict_single_smiles(self):
        result = self.model.predict("CCO")
        assert isinstance(result, dict)
        assert len(result) == 12
        for v in result.values():
            assert 0.0 <= v <= 1.0 or v != v   # float or nan

    def test_predict_invalid_smiles(self):
        result = self.model.predict("ZZZINVALID")
        # All values should be nan
        import math
        assert all(math.isnan(v) for v in result.values())

    def test_predict_batch(self):
        smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        results = self.model.predict_batch(smiles)
        assert len(results) == 3
        for res in results:
            assert len(res) == 12

    def test_from_config(self):
        cfg = {
            "atom_dim": 85, "edge_dim": 12, "hidden_dim": 64,
            "gin_layers": 2, "dropout": 0.0, "num_tasks": 12,
        }
        model = MTGRLModel.from_config(cfg)
        assert model is not None
