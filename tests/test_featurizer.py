"""test_featurizer.py — Unit tests for MoleculeFeaturizer."""
import pytest
import torch
from src.data.featurizer import MoleculeFeaturizer, atom_features, bond_features


FEATURIZER = MoleculeFeaturizer()

VALID_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
    "CC(=O)Nc1ccc(cc1)O",      # Paracetamol
    "c1ccccc1",                 # Benzene
    "CCO",                      # Ethanol
]

INVALID_SMILES = ["", "not_a_smiles", "ZZZZZZ"]


class TestFeaturizer:
    def test_valid_smiles_returns_data(self):
        for smi in VALID_SMILES:
            data = FEATURIZER.smiles_to_graph(smi)
            assert data is not None, f"Expected Data for '{smi}'"

    def test_invalid_smiles_returns_none(self):
        for smi in INVALID_SMILES:
            data = FEATURIZER.smiles_to_graph(smi)
            assert data is None, f"Expected None for '{smi}'"

    def test_atom_feature_dim(self):
        for smi in VALID_SMILES:
            data = FEATURIZER.smiles_to_graph(smi)
            assert data.x.shape[1] == 85, (
                f"Atom dim should be 85, got {data.x.shape[1]} for '{smi}'"
            )

    def test_edge_feature_dim(self):
        for smi in VALID_SMILES:
            data = FEATURIZER.smiles_to_graph(smi)
            if data.num_edges > 0:
                assert data.edge_attr.shape[1] == 12, (
                    f"Edge dim should be 12, got {data.edge_attr.shape[1]} for '{smi}'"
                )

    def test_edge_index_shape(self):
        data = FEATURIZER.smiles_to_graph("c1ccccc1")
        assert data.edge_index.shape[0] == 2
        # Benzene: 6 bonds × 2 directions = 12 edges
        assert data.edge_index.shape[1] == 12

    def test_dtype_float(self):
        data = FEATURIZER.smiles_to_graph("CCO")
        assert data.x.dtype == torch.float32
        assert data.edge_attr.dtype == torch.float32

    def test_edge_index_dtype_long(self):
        data = FEATURIZER.smiles_to_graph("CCO")
        assert data.edge_index.dtype == torch.long

    def test_single_atom_molecule(self):
        # Single atom: no bonds
        data = FEATURIZER.smiles_to_graph("[Na+]")
        assert data is not None
        assert data.num_nodes == 1
        assert data.num_edges == 0

    def test_validate_dimensions(self):
        for smi in VALID_SMILES:
            data = FEATURIZER.smiles_to_graph(smi)
            assert FEATURIZER.validate_dimensions(data)
