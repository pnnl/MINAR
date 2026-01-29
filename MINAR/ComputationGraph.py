import re
import copy
import numpy as np
import torch
from torch import Tensor
import torch_geometric as pyg
import networkx as nx
import itertools
from collections import OrderedDict
from .EAPScores import *
from .utils import longest_path, _place_hook, _apply_model
from typing import Callable, Optional, Union

def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))

class ComputationGraph(nx.DiGraph):
    '''
    Class to construct neuron-level model computation graphs from a GNN model.

    Args:
        model (torch.nn.Module): GNN model to build computation graph from
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 module_name = None,
                 special_modules = None) -> None:
        super().__init__()
        
        self.model = model
        input_dim = model.in_channels
        self.layers = [[f'input.{i}' for i in range(input_dim)]]
        self.modules = {}
        for v in self.layers[0]:
            self.add_node(v, layer = 0)
        l = 1
        for name, module in model.named_modules():
            if _place_hook(module) and 'act' not in name and 'trim' not in name:
                if special_modules is None or name not in special_modules:
                    layer = []
                    if name is None or name == '':
                        name = module_name if module_name is not None else 'output'
                    for i in range(module.out_channels):
                        v = f'{name}.{i}'
                        self.add_node(v, layer = l)
                        layer.append(v)
                        if hasattr(module, 'weight'):
                            for u in self.layers[-1]:
                                j = int(re.match('.*?([0-9]+)$', u).group(1))
                                self.add_edge(u, v, weight = float(module.weight[i,j].detach()))
                    self.layers.append(layer)
                    self.modules[name] = module
                    l += 1
    
    def add_module(self, name, module,
                   module_inputs : Union[list, int, str],
                   module_outputs : Union[dict, int, str],
                   layer=None,
                   **kwargs) -> None:
        
        # Flag to indicate if module values are added to module_outputs rather than concatenated.
        # If `module_outputs` is an int or a str, we assume that this is the case.
        addition_output = False
        if isinstance(module_inputs, int):
            module_inputs = self.layers[module_inputs]
        elif isinstance(module_inputs, str):
            module_inputs = [module_inputs]
        if isinstance(module_outputs, int):
            module_outputs = {output_name : 1 for output_name in self.layers[module_outputs]}
            addition_output = True
        elif isinstance(module_outputs, str):
            module_outputs = {module_outputs : 1}
            addition_output = True

        module_G = ComputationGraph(module,
                                    special_modules=kwargs.get('special_modules', None),
                                    module_name=kwargs.get('module_name', name))
        
        assert len(module_inputs) == len(module_G.layers[0]), "module_inputs must match module's input count"
        mapping = {old: new for old, new in zip(module_G.layers[0], module_inputs)}
        nx.relabel_nodes(module_G, mapping, copy=False)
        module_G.layers[0] = module_inputs
        
        self.add_nodes_from(module_G.nodes)
        for v in module_G.nodes:
            self.nodes[v]['layer'] = self.nodes[v].get(layer if layer is not None else 0)
        self.add_edges_from(module_G.edges(data=True))

        for v, w in zip(module_G.layers[-1], module_outputs.keys()):
            if w not in self.nodes():
                self.add_node(w, layer = len(self.layers) - 1)
                self.layers[-1].append(w)
            self.add_edge(v, w, weight = float(module_outputs[w]))
            if addition_output:
                self[v][w]['addition'] = True
            if w in self.layers[0]:
                new_input = f'input.{w}'
                self.add_edge(new_input, w, weight = 1.0, addition=True)
        self.modules[name] = module
        
    def add_inputs(self,
                   new_inputs: dict,
                   default_weight: float = 1) -> None:
        '''
        Add additional inputs to the model (e.g. edge weights or attributes that are
        not reflected in the model's in_channels).

        Note that the new inputs will always be put in layer 0, but they can be connected to any layer
        in the computation graph.
        
        If `name : [layer]` is passed, the weight values will all be `default_weight`
        (e.g. the assumption will be that the new input is added to the activations
        rather than concatenated)

        Args:
            new_inputs (dict): `new_inputs` should be a dictionary of the additional inputs to layer with
                - `name : layer` or
                - `name : [layer]` or
                - `name : [layer, weight]`
                where `weight` is an array of length equal to the size of `layer`.
            default_weight (float, optional): (default: `1`)
        '''
        for name, values in new_inputs.items():

            if isinstance(values, list):
                layer = values[0]
            else:
                layer = values
            assert layer < len(self.layers)
            
            if isinstance(values, list) and len(values) == 2:
                weight = values[1]
            else:
                weight = default_weight * np.ones(len(self.layers[layer]))

            self.add_node(name, layer = 0)
            self.layers[0].append(name)
            for i, v in enumerate(self.layers[layer]):
                self.add_edge(name, v, weight = float(weight[i]))

    def add_residual_connections(self,
                                 connections: dict,
                                 default_weight: float = 1) -> None:
        '''
        Add residual connections to model computation graph.

        Args:
            connections (dict): a dictionary of the desired connections of the form:
                `source : destination` or
                `source : [destination, weight]` where `source` can be a node or a layer,
                    and `destination` can each be a node, a list of nodes, or a layer
                    and `weight` is an array of size `len(source)*len(destination)`
            default_weight (float, optional): (default: `1`)
        '''
        for source, values in connections.items():
            if isinstance(source, str):
                src_lst = [source]
            elif isinstance(source, int):
                src_lst = self.layers[source]

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
                dst_lst = self.layers[destination]
            
            if weight is None:
                weight = default_weight * np.ones((len(src_lst), len(dst_lst)))

            for (i,j), (u,v) in enumerated_product(src_lst, dst_lst):
                self.add_edge(u,v, weight = weight[i,j])

    def correct_layers(self) -> None:
        '''
        Utility function to correct layer assignments for all nodes in the graph.
        This runs a topological sort and assigns each node a layer equal to
        max(predecessor layers) + 1 (or 0 for source nodes). It then rebuilds
        self.layers as a list of node lists per layer.
        '''
        try:
            topo_order = list(nx.topological_sort(self))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains cycles; cannot perform topological sort to correct layers")

        layer_of = {}
        for v in topo_order:
            preds = list(self.predecessors(v))
            if not preds:
                layer = 0
            else:
                layer = max(layer_of[p] for p in preds) + 1
            layer_of[v] = layer
            self.nodes[v]['layer'] = layer

        if layer_of:
            max_layer = max(layer_of.values())
            layers = [[] for _ in range(max_layer + 1)]
            for v in topo_order:
                layers[layer_of[v]].append(v)
        else:
            layers = [[]]

        self.layers = layers

    def calculate_scores(self,
                         clean_data: pyg.data.Data,
                         corrupted_data: pyg.data.Data,
                         loss: Callable,
                         which: str = 'EAP',
                         **kwargs) -> None:
        '''
        Calculate edge attribution scores. 

        Args:
            clean_data (torch_geometric.data.Data): clean input data
            corrupted_data (torch_geometric.data.Data): corrupted input data
            loss (Callable): loss function to compute difference between clean and corrupted outputs
            which (str, optional): score method to use. Currently supported methods:
                `"weight_grad"`: Compute the gradient of the (clean) output with respect to the individual edge weight
                `"EAP"`: Edge Attribution Patching from Syed et al. "Attribution Patching Outperforms Automated Circuit Discovery."
                    (https://arxiv.org/abs/2310.10348)
                `"EAP-IG"`: Edge Attribution Patching with Integrated Gradients from Hanna et al.
                    "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms."
                    (https://arxiv.org/abs/2403.17806)
                (default: `"EAP"`)
            **kwargs (optional): Additional keyword arguments for score calculation
        '''
        if which == 'weight_grad':
            score_function = compute_weight_grad_scores
        elif which == 'EAP':
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
                for i, u in enumerate(self.predecessors(v)):
                    if i < score.shape[0]:
                        nx.set_edge_attributes(self, {(u,v) : {which : float(score[i,j].abs())}})
                    else:
                        m = re.match(r'.*?([0-9]+)$', u)
                        if m is None:
                            ind = 0
                        else:
                            ind = int(m.group(1))
                        nx.set_edge_attributes(self, {(u,v) : {which : float(score[ind,j].abs())}})
                for w in self.successors(v):
                    if self[v][w].get('addition', False):
                        nx.set_edge_attributes(self, {(v,w) : {which : 0.0}})
        for edge in self.edges:
            if which not in self.edges[edge]:
                nx.set_edge_attributes(self, {edge : {which : 0.0}})

class Circuit(nx.DiGraph):
    '''
    Class to represent subgraphs taken from the computation graph as a union of
    longest paths weighted by attribution scores.

    Args:
        model (torch.nn.Module): GNN model to build computation graph from and use for inference
        G (Optional[ComputationGraph]): base `ComputationGraph` object to select from. If
            `None`, `G` will be built using `model`.
        K (int, optional): number of edges to add after selecting initial paths (default: `10`)
        key (str, optional): which score to use as weight during circuit selection.
            Scores must be calculated before they can be used (default: `"weight"`)
        as_view (bool, optional): whether or not to use `G` in-place as a view.
            If `False`, a copy of `G` will be made (default: `True`)
        use_abs (bool, optional): whether or not to take the absolute value of edge scores
            when selecting edges for the circuit (default: `True`)
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 G: Optional[ComputationGraph]=None,
                 K: int=10,
                 key: str='weight',
                 as_view:bool=True,
                 use_abs:bool=True,
                 include_all_outputs:bool=True) -> None:

        self.model = model
        self.model_state_dict = copy.deepcopy(model.state_dict())
        self.mask_handles = []
        if G is not None:
            self.G = G
        else:
            G = ComputationGraph(model)

        assert nx.get_edge_attributes(G, key) is not None

        self.layers = [[] for layer in G.layers]

        circuit_edges = set()
        if use_abs:
            sorted_edges = iter(sorted(nx.get_edge_attributes(G, key).items(), key=lambda item: abs(item[1]), reverse=True))
        else:
            sorted_edges = iter(sorted(nx.get_edge_attributes(G, key).items(), key=lambda item: item[1], reverse=True))
        
        if include_all_outputs:
            for output in G.layers[-1]:
                self.layers[-1].append(output)
                path = longest_path(G, G.layers[0], [output], top_sort=G.layers, key = key)
                for j in range(1, len(path)):
                    circuit_edges.add((path[j-1], path[j]))
                    if path[j-1] not in self.layers[j-1]:
                        self.layers[j-1].append(path[j-1])

        k = 0
        while k < K:
            edge_to_add, _ = next(sorted_edges)
            if edge_to_add not in circuit_edges:
                pre_path = longest_path(G, G.layers[0], [edge_to_add[0]], top_sort=G.layers, key=key)
                post_path = longest_path(G, [edge_to_add[1]], G.layers[-1], top_sort=G.layers, key=key)
                path = pre_path + post_path
                for j in range(1, len(path)):
                    circuit_edges.add((path[j-1], path[j]))
                    l = G.nodes[path[j-1]]['layer']
                    if path[j-1] not in self.layers[l]:
                        self.layers[l].append(path[j-1])
                k += 1
        
        super().__init__(G.to_directed(as_view=as_view).edge_subgraph(circuit_edges))

    def _apply_masks(self, default_one: bool, invert: bool) -> None:
        '''
        Helper to build and register forward-pre hooks that mask module weights.

        Args:
            default_one (bool): whether to initialize masks with ones (True) or zeros (False)
            invert (bool): if True, use 1 - has_edge(u,v) for mask entries; otherwise use has_edge(u,v)
        '''
        def _mask_module(mask):
            def _mask_pre_hook(module, input):
                module.weight.data *= mask
            return _mask_pre_hook

        for name, module in self.G.modules.items():
            mask = torch.ones_like(module.weight) if default_one else torch.zeros_like(module.weight)
            inputs = OrderedDict()
            # locate the nodes in self.G that belong to this module by matching the module name prefix
            module_nodes = []
            prefix = f'{name}.'
            for v in self.G.nodes():
                if v.startswith(prefix):
                    module_nodes.append(v)
                    for u in self.G.predecessors(v):
                        if not self.G[u][v].get('addition', False):
                            inputs[u] = None
            module_nodes.sort(key=lambda x: int(x.split('.')[-1]))
            input_list = list(inputs.keys())
            for (j, u), (i, v) in itertools.product(enumerate(input_list), enumerate(module_nodes)):
                val = float(self.has_edge(u, v))
                if invert:
                    val = 1.0 - val
                mask[i, j] = val
            self.mask_handles.append(module.register_forward_pre_hook(_mask_module(mask)))
    
    def _clear_masks(self) -> None:
        '''
        Utility function to clear parameter masks from model
        '''
        self.model.load_state_dict(self.model_state_dict)
        while self.mask_handles:
            h = self.mask_handles.pop()
            h.remove()

    def forward(self,
                data: pyg.data.Data,
                **kwargs) -> Tensor:
        '''
        Apply GNN model using only the components included in the circuit subgraph.

        Args:
            data (torch_geometric.data.Data): input data for forward pass
            **kwargs (optional): additional keyword arguments to model

        Returns:
            Output of the model restricted to circuit components.
        '''
        self.model.eval()
        self._apply_masks(default_one=False, invert=False)
        out = _apply_model(self.model, data, **kwargs)
        self._clear_masks()
        return out
    
    def ablate_circuit(self,
                       data: pyg.data.Data,
                       **kwargs) -> Tensor:
        '''
        Ablate circuit components from model and apply to input data.

        Args:
            data (torch_geometric.data.Data): input data for forward pass
            **kwargs (optional): additional keyword arguments to model

        Returns:
            Output of the model with circuit components ablated.
        '''
        self.model.eval()
        self._apply_masks(default_one=True, invert=True)
        out = _apply_model(self.model, data, **kwargs)
        self._clear_masks()
        return out