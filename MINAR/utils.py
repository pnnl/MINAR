import networkx as nx
import torch
from torch import Tensor
from typing import Optional
import torch_geometric as pyg
def _apply_model(model: torch.nn.Module,
                 data: pyg.data.Data,
                 **kwargs) -> Tensor:
    if model.supports_edge_weight and model.supports_edge_attr:
        return model(data.x, data.edge_index, edge_weight=data.edge_weight, edge_attr=data.edge_attr, **kwargs)
    elif model.supports_edge_weight:
        return model(data.x, data.edge_index, edge_weight=data.edge_weight, **kwargs)
    elif model.supports_edge_attr:
        return model(data.x, data.edge_index, edge_attr=data.edge_attr, **kwargs)
    else:
        return model(data.x, data.edge_index, **kwargs)

def _place_hook(module: torch.nn.Module,
                hook_str:str = 'register_forward_hook') -> None:
    return hasattr(module, hook_str) and \
            len(list(module.children())) == 0 and \
            not isinstance(module, torch.nn.Identity) and \
            not isinstance(module, pyg.nn.aggr.Aggregation) and \
            not isinstance(module, torch.nn.Dropout)

def longest_path(G: nx.DiGraph,
                 sources: list[str],
                 targets: list[str],
                 top_sort: Optional[list[list[str]]]=None,
                 key: Optional[str]='weight'):
    '''
    Utility function to compute the longest path between sources and targets
    in a topologically sorted DAG.

    Args:
        G (networkx.DiGraph): DAG to compute longest paths in
        sources (list[str]): list of source nodes
        targets (list[str]): list of target nodes
        top_sort (list[list[str]], optional): topologically sorted layers
            of G. `top_sort` should be a list of lists containing each topological layer
            of G, topologically sorted, where each layer is a list of nodes. If `None`,
            this function will call `nx.topological_generations(G)`
            (Default: None)
        key (str, optional): edge attribute to use as the edge weight
            (Default: weight)
    '''
    weights = nx.get_edge_attributes(G, key)
    if top_sort is None:
        top_sort = [list(generation) for generation in nx.topological_generations(G)]
    distances = {v : float('-inf') for v in G.nodes}
    predecessors = {v : None for v in G.nodes}
    for u in sources:
        distances[u] = 0
    for layer in top_sort:
        for u in layer:
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