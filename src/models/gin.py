"""
gin.py
──────
Graph Isomorphism Network (GIN) backbone with edge features (GINEConv).

Architecture per layer
──────────────────────
  h_v^(k) = MLP [ (1 + ε) · h_v^(k-1)  +  Σ_{u ∈ N(v)} (h_u^(k-1) + e_uv) ]
                              ↑                              ↑
                         learnable ε               edge feature added before agg

After all layers: global SUM pooling → graph embedding [B, hidden_dim]

Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks"
           Xu et al.  "How Powerful are Graph Neural Networks?"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP used inside each GINEConv."""
    return nn.Sequential(
        nn.Linear(in_dim, 2 * out_dim),
        nn.BatchNorm1d(2 * out_dim),
        nn.ReLU(),
        nn.Linear(2 * out_dim, out_dim),
    )


POOLING_MAP = {
    "sum":  global_add_pool,
    "mean": global_mean_pool,
    "max":  global_max_pool,
}


class GINBackbone(nn.Module):
    """
    Multi-layer GINEConv backbone.

    Parameters
    ----------
    atom_dim    : input node feature dim (85)
    edge_dim    : input edge feature dim (12)
    hidden_dim  : embedding width at every layer (300)
    num_layers  : number of message-passing rounds (5)
    dropout     : dropout probability applied after each layer (0.5)
    pooling     : 'sum' | 'mean' | 'max'
    residual    : add skip-connection between consecutive layers

    Output
    ------
    graph_emb : FloatTensor of shape [batch_size, hidden_dim]
    """

    def __init__(
        self,
        atom_dim: int = 85,
        edge_dim: int = 12,
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.5,
        pooling: str = "sum",
        residual: bool = True,
    ):
        super().__init__()
        assert num_layers >= 2, "Need at least 2 GIN layers."
        assert pooling in POOLING_MAP, f"pooling must be one of {list(POOLING_MAP)}"

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.residual = residual
        self.pool = POOLING_MAP[pooling]

        # Input projection: atom_dim → hidden_dim
        self.input_proj = nn.Linear(atom_dim, hidden_dim)

        # Edge feature projection: edge_dim → hidden_dim (required by GINEConv)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GINEConv layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for _ in range(num_layers):
            mlp = _make_mlp(hidden_dim, hidden_dim)
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.drop = nn.Dropout(p=dropout)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Batch object with fields:
               x          [N_total, atom_dim]
               edge_index [2, E_total]
               edge_attr  [E_total, edge_dim]
               batch      [N_total]  (node → graph mapping)

        Returns
        -------
        graph_emb : [B, hidden_dim]
        """
        x         = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float()
        batch      = data.batch

        # Project inputs to hidden_dim
        x         = self.input_proj(x)                    # [N, H]
        edge_attr = self.edge_proj(edge_attr)             # [E, H]

        # Message passing
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index, edge_attr)            # [N, H]
            h = bn(h)
            h = F.relu(h)
            h = self.drop(h)

            # Residual connection (skip first layer — dims may differ in custom setups)
            if self.residual and i > 0:
                x = x + h
            else:
                x = h

        # Global pooling → graph-level embedding
        graph_emb = self.pool(x, batch)                  # [B, H]
        return graph_emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
