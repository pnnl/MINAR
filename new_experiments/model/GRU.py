import torch_geometric.nn as pyg_nn
from inspect import signature
from loguru import logger
import torch
import torch.nn as nn
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

class GRU(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  edge_dim=1, aggr='sum', mlp=False,
                  **kwargs) -> MessagePassing:
        if mlp:
            mlp_edge = MLP(
                [in_channels+edge_dim, out_channels, out_channels],
                act=self.act,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            return GRUMLPConv(in_channels, out_channels, aggr=aggr, mlp_edge=mlp_edge, **kwargs)
        else:
            return GRUConv(in_channels, out_channels, aggr=aggr, **kwargs)

######################
# Modules from https://github.com/floriangroetschla/Recurrent-GNNs-for-algorithm-learning/blob/main/model.py
# Adapted to work with edge weights

class GRUConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr):
        super(GRUConv, self).__init__(aggr=aggr)
        logger.info(f"GRUConv: in_channels: {in_channels}, out_channels: {out_channels}")
        self.rnn = torch.nn.GRUCell(in_channels, out_channels)
        self.edge_weight_scaler = nn.Linear(1, in_channels)

    def forward(self, x, edge_index, edge_weight, last_hidden):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.rnn(out, last_hidden)
        return out

    def message(self, x_j, edge_weight):
        return F.relu(x_j + self.edge_weight_scaler(edge_weight.unsqueeze(-1)))

class GRUMLPConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, mlp_edge, aggr):
        super(GRUMLPConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(in_channels, out_channels)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index, last_hidden, edge_attr=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.rnn(out, last_hidden)
        return out

    def message(self, x_j, x_i, edge_attr=None):
        concatted = torch.cat((x_j, x_i), dim=-1)
        if edge_attr is not None:
            concatted = torch.cat((concatted, edge_attr.unsqueeze(-1)), dim=-1)
        return self.mlp_edge(concatted) 