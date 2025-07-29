import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.data import Data

beta = 100.

def path_graph(k, A, m = None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    if A is None:
        A = rng.uniform(1.,10.,k)
    if m is None:
        m = k
    assert len(A) == k
    forward_edge_index = torch.tensor(
        [[i for i in range(k)],
         [i for i in range(1,k+1)]]
    )
    if not isinstance(A, Tensor):
        forward_edge_attr = torch.tensor(A)
    else:
        forward_edge_attr = A
    edge_index = torch.cat((forward_edge_index, forward_edge_index.flip(0)), 1)
    edge_attr = torch.cat((forward_edge_attr, forward_edge_attr.flip(0))).reshape(-1,1)
    # beta = torch.sum(forward_edge_attr) + 1
    x = torch.full((k+1, 1), beta)
    x[0] = 0.
    y = x.clone()
    for i in range(m+1):
        y[i] = forward_edge_attr[:i].sum()
    y = y.flatten()
    reachable = torch.zeros(k+1, dtype=torch.bool)
    reachable[:m+1] = True
    data = Data(x = x, y = y.float(),
                edge_index = edge_index,
                edge_attr = edge_attr.float(),
                reachable = reachable)
    return data

def H_graph(K, m = None):
    if m is None:
        m = K
    forward_v_edge_index = torch.tensor(
        [[i for i in range(K)],
         [i for i in range(1,K+1)]]
    )
    forward_v_edge_attr = torch.zeros(K)
    forward_u_edge_index = torch.tensor(
        [[i for i in range(K+1,2*K+1)],
         [i for i in range(K+2,2*(K+1))]]
    )
    forward_u_edge_attr = torch.zeros(K)
    forward_v_to_u_edge_index = torch.tensor(
        [[i for i in range(K)],
         [i for i in range(K+2,2*(K+1))]]
    )
    forward_v_to_u_edge_attr = torch.ones(K)
    forward_u_to_v_edge_index = torch.tensor(
        [[i for i in range(K+1,2*K+1)],
         [i for i in range(1,K+1)]]
    )
    forward_u_to_v_edge_attr = torch.ones(K)

    forward_edge_index = torch.cat((forward_v_edge_index,
                                    forward_u_edge_index,
                                    forward_v_to_u_edge_index,
                                    forward_u_to_v_edge_index), 1)
    forward_edge_attr = torch.cat((forward_v_edge_attr,
                                    forward_u_edge_attr,
                                    forward_v_to_u_edge_attr,
                                    forward_u_to_v_edge_attr))
    edge_index = torch.cat((forward_edge_index, forward_edge_index.flip(0)), 1)
    edge_attr = torch.cat((forward_edge_attr, forward_edge_attr.flip(0))).reshape(-1,1)
    # beta = torch.sum(forward_edge_attr) + 1
    x = torch.full((2*(K+1), 1), beta)
    x[0] = 0.
    y = torch.ones_like(x)
    y[0] = 0.
    y = y.flatten()
    reachable = torch.ones_like(y, dtype=torch.bool)
    data = Data(x = x, y = y.float(),
                edge_index = edge_index,
                edge_attr = edge_attr.float(),
                reachable = reachable)
    return data

def _nx_to_test_data(G, K, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    nx.set_edge_attributes(G,
                           values = {e : float(rng.uniform(1.,10.)) for e in G.edges()},
                           name = 'weight')
    hops = nx.single_source_shortest_path_length(G, 0, cutoff=K)
    reachable = torch.tensor(
        [v in hops.keys() for v in range(G.number_of_nodes())]
    )

    distances = nx.single_source_bellman_ford_path_length(G, 0)
    # beta = sum(distances.values()) + 1

    data = pyg.utils.from_networkx(G)
    data.y = torch.tensor([distances[i] if reachable[i] else beta for i in range(G.number_of_nodes())])
    data.x = torch.full((G.number_of_nodes(),1), beta)
    data.x[0] = 0.0
    data.edge_attr = data.weight.reshape(-1,1)
    data.weight = None
    data.reachable = reachable
    return data

def cycle_test_graph(n, K, rng=None):
    G = nx.cycle_graph(n)
    return _nx_to_test_data(G, K, rng=rng)

def complete_test_graph(n, K, rng=None):
    G = nx.complete_graph(n)
    return _nx_to_test_data(G, K, rng=rng)

def ER_test_graph(n, p, K, rng=None):
    G = nx.erdos_renyi_graph(n, p, seed=rng)
    return _nx_to_test_data(G, K, rng=rng)