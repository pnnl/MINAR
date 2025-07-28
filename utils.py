import networkx as nx
import torch
import torch_geometric as pyg

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

def _place_hook(module, hook_str = 'register_forward_hook'):
    return hasattr(module, hook_str) and \
            len(list(module.children())) == 0 and \
            not isinstance(module, torch.nn.Identity) and \
            not isinstance(module, pyg.nn.aggr.Aggregation) and \
            not isinstance(module, torch.nn.Dropout)