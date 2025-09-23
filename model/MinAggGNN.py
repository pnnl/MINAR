from typing import Final, Optional, List
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN, MessagePassing
from torch_geometric.nn.resolver import (
    activation_resolver,
)
from typing import Callable, Final, List, Optional, Union

import torch
from torch import Tensor

class MinAggConv(MessagePassing):
    r'''
    The minimum-aggregated :class:`MinAggConv` operator from https://arxiv.org/abs/2503.19173.

    .. math::
        h'_v = \varphi_{\text{up}} \left(
            \min_{u \sim v} \{\sigma(\varphi_{\text{agg}}(h_u, e_{uv}))\}, h_v
        \right)

    where :math:`\varphi_{\text{agg}}` and :math:`\varphi_{\text{up}}` are MLPs.

    Args:
        agg_mlp (torch.nn.Module): neural network of shape
            `(in_channels+edge_dim, hidden_channels)` that aggregates neighboring
            node features and edge features to intermediate features
        up_mlp (torch.nn.Module): neural network `(hidden_channels, out_channels)`
            that transformas aggregated features.
        act (str or Callable, optional): The non-linear activation function :math:`\sigma`
            to apply after agg_mlp. (default: :obj:`"relu"`).

    Shapes:
        input: node features :math:`(|V|,d_{\text{in}})`,
            edge indices :math:`(2, |E|)`, edge features :math:`(|E|,d_{\text{edge}})`
            (optional)
        output: node features :math:`(|V|,d_{\text{out}})`
    '''
    def __init__(
        self,
        agg_mlp: torch.nn.Module,
        up_mlp: torch.nn.Module,
        act: Union[str, Callable, None] = "relu"
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
    r'''
    The minimum-aggregated message-passing neural network from https://arxiv.org/abs/2503.19173,
    using the :class:`~MinAggConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MinAggConv`.
    '''
    
    mlp_hidden_channels = 64
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def init_conv(self,
                  in_channels: int,
                  out_channels: int,
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
        r'''
        Forward Pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        '''
        x = super().forward(x,
                          edge_index,
                          edge_weight,
                          edge_attr,
                          batch,
                          batch_size,
                          num_sampled_nodes_per_hop,
                          num_sampled_edges_per_hop)
        return x