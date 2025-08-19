from typing import Final, Optional, List
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import MLP, GINConv, GINEConv
from torch_geometric.nn.models.basic_gnn import BasicGNN, MessagePassing
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset

class MinAggConv(MessagePassing):
    def __init__(
        self,
        agg_mlp,
        up_mlp,
        act = 'ReLU',
        **kwargs
    ):
        super().__init__(aggr='min')
        self.agg_mlp = agg_mlp
        self.up_mlp = up_mlp
        self.act = activation_resolver(act)
        self.reset_parameters()
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        tmp = torch.cat((x_j, edge_attr), 1)
        return self.act(self.agg_mlp(tmp))
    
    def update(self, aggr_out, x):
        tmp = torch.cat((aggr_out, x), 1)
        return self.up_mlp(tmp)

class MinAggGNN(BasicGNN):
    
    mlp_hidden_channels = 64
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        if 'edge_dim' in kwargs:
            edge_dim = kwargs.pop('edge_dim')
        else:
            edge_dim = 0
        agg_mlp = MLP(
            [in_channels + edge_dim, self.mlp_hidden_channels, self.mlp_hidden_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        up_mlp = MLP(
            [self.mlp_hidden_channels + in_channels, self.mlp_hidden_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return MinAggConv(agg_mlp, up_mlp, **kwargs)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        x = super().forward(x,
                          edge_index,
                          edge_weight,
                          edge_attr,
                          batch,
                          batch_size,
                          num_sampled_nodes_per_hop,
                          num_sampled_edges_per_hop)
        return x