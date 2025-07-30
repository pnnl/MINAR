import re
import copy
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
import itertools
from collections import OrderedDict
from .EAPScores import compute_eap_scores, compute_eap_ig_scores
from .utils import longest_path, _place_hook, _apply_model

def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))

class ComputationGraph(nx.DiGraph):
    
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        input_dim = model.in_channels
        self.top_sort = [[f'input.{i}' for i in range(input_dim)]]
        self.layer_modules = []
        for v in self.top_sort[0]:
            self.add_node(v, layer = 0)
        l = 1
        for name, module in model.named_modules():
            if _place_hook(module) and 'act' not in name and 'trim' not in name:
                layer = []
                for i in range(module.out_channels):
                    v = f'{name}.{i}'
                    self.add_node(v, layer = l)
                    layer.append(v)
                    if hasattr(module, 'weight'):
                        for u in self.top_sort[-1]:
                            j = int(re.match('.*?([0-9]+)$', u).group(1))
                            self.add_edge(u, v, weight = float(module.weight[i,j]))
                self.top_sort.append(layer)
                self.layer_modules.append((name, module))
                l += 1
        
    def add_inputs(self, new_inputs, default_weight = 1):
        '''
        new_inputs should be a dictionary of the additional inputs to layer with
        name : layer or
        name : [layer] or
        name : [layer, weight]
        where weight is an array of length equal to the size of layer.
        Note that the new inputs will always be put in layer 0.
        If name : [layer] is passed, the weight values will all be default_weight
        (e.g. the assumption will be that the new input is added to the activations
        rather than concatenated)
        '''
        for name, values in new_inputs.items():

            if isinstance(values, list):
                layer = values[0]
            else:
                layer = values
            assert layer < len(self.top_sort) - 1
            
            if isinstance(values, list) and len(values) == 2:
                weight = values[1]
            else:
                weight = default_weight * np.ones(len(self.top_sort[layer]))

            self.add_node(name, layer = 0)
            self.top_sort[0].append(name)
            for i, v in enumerate(self.top_sort[layer]):
                self.add_edge(name, v, weight = float(weight[i]))

    def add_residual_connections(self, connections, default_weight = 1):
        '''
        connections should be a dictionary of the desired connections of the form
        source : destination or
        source : [destination, weight] where source can be a node or a layer,
        and destination can each be a node, a list of nodes, or a layer
        and weight is an array of size len(source) x len(destination)
        '''
        for source, values in connections.items():
            if isinstance(source, str):
                src_lst = [source]
            elif isinstance(source, int):
                src_lst = self.top_sort[source]

            if isinstance(values, list) and \
              len(values) == 2 and \
              np.asarray(values[1]).shape[0] == len(src_lst):
                destination = values[0]
                weight = np.asarray(values[1])
            else:
                destination = values
                weight = None
            if isinstance(destination, str):
                dst_lst = [destination]
            elif isinstance(destination, list):
                dst_lst = destination
            elif isinstance(destination, int):
                dst_lst = self.top_sort[destination]
            
            if weight is None:
                weight = default_weight * np.ones((len(src_lst), len(dst_lst)))

            for (i,j), (u,v) in enumerated_product(src_lst, dst_lst):
                self.add_edge(u,v, weight = weight[i,j])

    def calculate_scores(self, clean_data, corrupted_data, loss, which = 'EAP', **kwargs):
        if which == 'EAP':
            score_function = compute_eap_scores
        elif which == 'EAP-IG':
            score_function = compute_eap_ig_scores
        else:
            raise NotImplementedError()
        all_scores = []
        avg_scores = {}
        for data, data_corr in zip(clean_data, corrupted_data):
            all_scores.append(score_function(self.model, data, data_corr, loss, **kwargs))
        for key in all_scores[0].keys():
            avg_scores[key] = torch.mean(torch.stack([score_dict[key] for score_dict in all_scores]), 0)
        for key, score in avg_scores.items():
            for j in range(score.shape[1]):
                v = key + f'.{j}'
                for i, e in enumerate(self.in_edges(v)):
                    nx.set_edge_attributes(self, {e : {which : float(score[i,j].abs())}})

class Circuit(nx.DiGraph):

    def __init__(self, model,
                 G=None, K=10, key='weight',
                 as_view=True, use_abs=True):

        self.model = model
        self.model_state_dict = copy.deepcopy(model.state_dict())
        self.mask_handles = []
        if G is not None:
            self.G = G
        else:
            G = ComputationGraph(model)

        assert nx.get_edge_attributes(G, key) is not None

        circuit_edges = set()
        if use_abs:
            sorted_edges = iter(sorted(nx.get_edge_attributes(G, key).items(), key=lambda item: abs(item[1]), reverse=True))
        else:
            sorted_edges = iter(sorted(nx.get_edge_attributes(G, key).items(), key=lambda item: item[1], reverse=True))
        path = longest_path(G, G.top_sort[0], G.top_sort[-1], top_sort=G.top_sort, key = key)
        for j in range(1, len(path)):
            circuit_edges.add((path[j-1], path[j]))

        k = 0
        while k < K:
            edge_to_add, _ = next(sorted_edges)
            if edge_to_add not in circuit_edges:
                pre_path = longest_path(G, G.top_sort[0], [edge_to_add[0]], top_sort=G.top_sort, key=key)
                post_path = longest_path(G, [edge_to_add[1]], G.top_sort[-1], top_sort=G.top_sort, key=key)
                path = pre_path + post_path
                for j in range(1, len(path)):
                    circuit_edges.add((path[j-1], path[j]))
                k += 1
        
        super().__init__(G.to_directed(as_view=as_view).edge_subgraph(circuit_edges))

    def _apply_masks(self):
        def _mask_module(mask):
            def _mask_pre_hook(module, input):
                module.weight.data *= mask
            return _mask_pre_hook

        for l, (name, module) in enumerate(self.G.layer_modules):
            mask = torch.zeros_like(module.weight)
            layer = self.G.top_sort[l+1]
            input_dict = OrderedDict()
            for v in layer:
                for u in self.G.predecessors(v):
                    input_dict[u] = None
            inputs = list(input_dict.keys())
            for (j, u), (i, v) in itertools.product(enumerate(inputs), enumerate(layer)):
                mask[i,j] = self.has_edge(u,v)
            self.mask_handles.append(module.register_forward_pre_hook(_mask_module(mask)))
    
    def _clear_masks(self):
        self.model.load_state_dict(self.model_state_dict)
        while self.mask_handles:
            h = self.mask_handles.pop()
            h.remove()

    def forward(self, data):
        self._apply_masks()
        out = _apply_model(self.model, data)
        self._clear_masks()
        return out