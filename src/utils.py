import networkx as nx
import torch
from torch import Tensor
from typing import Optional
import torch_geometric as pyg

def multiplicative_loss(
        input: Tensor,
        target: Tensor,
        reduction: str = "sum",
        weight: Optional[Tensor] = None,):

    errors = (torch.ones_like(input) - target / input).abs()
    if weight is not None:
        errors = errors * weight

    if reduction == "none":
        return errors
    elif reduction == "sum":
        return torch.sum(errors)
    elif reduction == "mean":
        return torch.sum(errors) / torch.sum(weight)

class MultiplicativeLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "sum") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return multiplicative_loss(input, target, reduction=self.reduction)
    
def _apply_model(model, data):
    if model.supports_edge_weight and model.supports_edge_attr:
        return model(data.x, data.edge_index, edge_weight=data.edge_weight, edge_attr=data.edge_attr)
    elif model.supports_edge_weight:
        return model(data.x, data.edge_index, edge_weight=data.edge_weight)
    elif model.supports_edge_attr:
        return model(data.x, data.edge_index, edge_attr=data.edge_attr)
    else:
        return model(data.x, data.edge_index)

def _place_hook(module, hook_str = 'register_forward_hook'):
    return hasattr(module, hook_str) and \
            len(list(module.children())) == 0 and \
            not isinstance(module, torch.nn.Identity) and \
            not isinstance(module, pyg.nn.aggr.Aggregation) and \
            not isinstance(module, torch.nn.Dropout)

def longest_path(G, sources, targets, top_sort=None, key='weight'):
    weights = nx.get_edge_attributes(G, key)
    if top_sort is None:
        top_sort = [list(generation) for generation in nx.topological_generations(G)]
    distances = {v : float('-inf') for v in G.nodes}
    predecessors = {v : None for v in G.nodes}
    for u in sources:
        distances[u] = 0
    for u in G.nodes:
        for edge in G.edges(u):
            v = edge[1]
            if distances[v] < distances[u] + weights[edge]:
                distances[v] = distances[u] + weights[edge]
                predecessors[v] = u
    tgt_distances = {tgt : distances[tgt] for tgt in targets}
    end, _ = max(tgt_distances.items(), key=lambda item: item[1])
    path = [end]
    while predecessors[path[-1]] is not None:
        path.append(predecessors[path[-1]])
    path.reverse()
    return path