from typing import Optional
import torch
from torch_geometric.nn import MLP, GINEConv
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

# /qfs/projects/stargazer/share/minar/model/RecGINE.py
"""
Recurrently-applied GINEConv GNN.

Implements a GNN that re-uses a single GINEConv layer recurrently for a number
of steps (num_layers). The API is kept compatible with torch_geometric's
BasicGNN-style models:

- __init__(in_channels, hidden_channels, out_channels, num_layers=3, ...)
- forward(x, edge_index, edge_attr=None, batch=None)

Behavior:
- If `batch` is provided, forward returns a graph-level tensor of shape
    (num_graphs, out_channels) (if out_channels is not None).
- If `batch` is None, forward returns node-level outputs of shape
    (num_nodes, out_channels) (if out_channels is not None).
- If out_channels is None, the model returns the raw node embeddings.
"""
import torch.nn as nn
import torch.nn.functional as F


class RecGINE(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            out_channels: int,
            edge_dim: Optional[int] = None,
            act: Optional[str] = 'relu',
            dropout: Optional[float] = None,
            **kwargs
    ):
            super().__init__()
            assert num_layers >= 1, "num_layers must be >= 1"

            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.num_layers = num_layers
            self.edge_dim = edge_dim
            self.act = act
            self.act_first = kwargs.get('act_first', False)
            self.norm = kwargs.get('norm', None)
            self.norm_kwargs = kwargs.get('norm_kwargs', None)
            self.residual = kwargs.get('residual', False)

            # Per-step activations and normalizations
            self.act = activation_resolver(self.act)
            self.norms = normalization_resolver(self.norm, hidden_channels, self.num_layers, self.norm_kwargs)
            
            # Input linear to project raw features to hidden dimensionality
            if in_channels != hidden_channels:
                self.input_lin = nn.Linear(in_channels, hidden_channels)
            else:
                self.input_lin = None

            # Single GINEConv reused recurrently
            conv_nn = MLP([hidden_channels, hidden_channels, hidden_channels],
                           act=self.act,
                           act_first=self.act_first,
                           norm=self.norm,
                           norm_kwargs=self.norm_kwargs,
            )
            self.conv = GINEConv(conv_nn, edge_dim=edge_dim)

            # Output head (optional)
            if out_channels != hidden_channels:
                self.out_lin = nn.Linear(hidden_channels, out_channels)
            else:
                self.out_lin = None

            self.reset_parameters()

    def reset_parameters(self):
        if self.input_lin is not None:
            if hasattr(self.input_lin, "reset_parameters"):
                self.input_lin.reset_parameters()
        # GINEConv has its own reset
        if hasattr(self.conv, "reset_parameters"):
                self.conv.reset_parameters()
        if self.norms is not None:
            for n in self.norms:
                if hasattr(n, "reset_parameters"):
                        n.reset_parameters()
        if self.out_lin is not None:
                self.out_lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x: node features [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: optional edge attributes [num_edges, edge_dim]
        batch: optional batch vector [num_nodes] for graph-level pooling
        """
        if self.input_lin is not None:
            x = self.input_lin(x)
        # recurrently apply the same conv layer
        for i in range(self.num_layers):
            x_in = x
            x = self.conv(x, edge_index, edge_attr)
            if self.residual:
                # If shapes differ for some reason, fall back to addition when possible
                if x.shape == x_in.shape:
                    x = x + x_in
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.act(x)
        if self.out_lin is not None:
            x = self.out_lin(x)

        return x