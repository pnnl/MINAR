from operator import truth
import re
import gc
from typing import Optional, override
import torch
import torch_geometric as pyg
import networkx as nx

import sys

import torch_scatter
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')
from MINAR.ComputationGraph import Circuit, ComputationGraph
from MINAR.EAPScores import _ensure_requires_grad
from MINAR.utils import _place_hook
from EncodeProcessDecode import EncodeProcessDecode

from salsaclrs import specs
from salsaclrs.data import SALSACLRSDataLoader
from baselines.core.utils import stack_hidden

lambda_hidden = 0.1 # Use the same hidden loss coefficient as during training

def _apply_salsa_clrs_model(model, data, **kwargs):
    out, hints, hidden = model(data, **kwargs)
    return out, hints, hidden

def _apply_salsa_clrs_loss(out_corr, out_clean, data, clrsloss, **kwargs):
    out_corr_, hint_corr, hidden_corr = out_corr
    out_clean_, hint_clean, hidden_clean = out_clean
    data_compare = data.clone()
    for output in data.outputs:
        data_compare[output] = out_corr_[output]
        type_ = specs.SPECS[data.task][output][2]
        if type_ == "mask":
            data_compare[output] = data_compare[output].sigmoid()
        elif type_ == "mask_one":
            data_compare[output] = torch_scatter.scatter_softmax(data_compare[output], data.batch, dim=0)
        elif type_ == "pointer":
            data_compare[output] = torch_scatter.scatter_softmax(data_compare[output], data.edge_index[0], dim=0)
        else:
            raise NotImplementedError
    # Hints are not used in eval mode
    out_loss, _, _ = clrsloss(data_compare, out_clean_, hint_clean, hidden_clean)
    hidden_loss = torch.nn.functional.mse_loss(hidden_corr, hidden_clean)
    total_loss = out_loss + lambda_hidden * hidden_loss
    return total_loss

def compute_weight_grad_scores(
        model: EncodeProcessDecode,
        data: pyg.data.Data,
        data_corr: pyg.data.Data,
        loss: torch.nn.modules.loss._Loss) -> dict[str,torch.Tensor]:
    r'''
    Compute score for a computation edge :math:`(\psi_i, \psi_j)` with weight
    :math:`w_{ij}` by
    .. math::
        \frac{\partial}{\partial w_{ij}} L(x_{\text{clean}}, x_{\text{corrupted}}).

    Args:
        model (EncodeProcessDecode): EncodeProcessDecode module to compute scores for.
        data (torch_geometric.data.Data): clean input Data object.
        data_corr (torch_geometric.data.Data): corrupted input Data object.
        loss (torch.nn.modules.loss._Loss): loss function to compute scores with.

    Returns:
        Dictionary of edge names and score values.
    '''
    scores = {}
            
    out_corr = _apply_salsa_clrs_model(model, data_corr)
    out_clean = _apply_salsa_clrs_model(model, data)
    L = _apply_salsa_clrs_loss(out_corr, out_clean, data, loss)
    L.backward()

    for name, module in model.processor.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook') and hasattr(module, 'weight'):
            scores[name] = module.weight.grad.detach().clone().T / data.num_nodes
            
    # clean up gpu memory
    model.zero_grad(set_to_none=True)
    return scores

def compute_eap_scores(model: EncodeProcessDecode,
                       data: pyg.data.Data,
                       data_corr: pyg.data.Data,
                       loss: torch.nn.modules.loss._Loss) -> dict[str,torch.Tensor]:
    r'''
    Compute edge attribution patching (EAP) score for a computation edge
    :math:`(\psi_i, \psi_j)`.
     
    The EAP score is given by
    ..math::
        \operatorname{EAP}_{(i,j)}(x,x') = (z_i'-z_i)^\transpose \partials{\psi_j}
            L(\Psi(x),\Psi(x'))
    
    where :math:`x` is the clean data, :math:`x'` is the corrupted data, and
    :math:`z_i` and :math:`z_i'` are the activations of neuron :math:`\psi_i`
    on :math:`x` and :math:`x'`, respectively.

    Args:
        model (EncodeProcessDecode): EncodeProcessDecode module to compute scores for.
        data (torch_geometric.data.Data): clean input Data object.
        data_corr (torch_geometric.data.Data): corrupted input Data object.
        loss (torch.nn.modules.loss._Loss): loss function to compute scores with.

    Returns:
        Dictionary of edge names and score values.
    '''
    data.apply_(_ensure_requires_grad)
    data_corr.apply_(_ensure_requires_grad)
    activations = []
    corr_activations = []
    gradients = []
    names = []
    def _activation_hook(module, input, output):
        activations.append(output.detach().clone())
    def _corr_activation_hook(module, input, output):
        corr_activations.append(output.detach().clone())
    def _get_gradients(name):
        def _gradient_hook(module, grad_input, grad_output):
            gradients.append(grad_input[0].detach().clone())
            names.append(name)
        return _gradient_hook
    
    # get corrupted activations
    corr_handles = []
    for name, module in model.processor.named_modules():
        if _place_hook(module):
            corr_handles.append(module.register_forward_hook(_activation_hook))
    out_corr = _apply_salsa_clrs_model(model, data_corr)
    for h in corr_handles:
        h.remove()
    
    # set forward hooks for clean activations
    clean_handles = []
    for name, module in model.processor.named_modules():
        if _place_hook(module):
            clean_handles.append(module.register_forward_hook(_corr_activation_hook))

    # set backward hooks
    backward_handles = []
    for name, module in model.processor.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook'):
            backward_handles.append(module.register_full_backward_hook(_get_gradients(name)))

    # get activations and gradients
    out_clean = _apply_salsa_clrs_model(model, data)
    L = _apply_salsa_clrs_loss(out_corr, out_clean, data, loss)
    L.backward()
    for h in clean_handles:
        h.remove()
    for h in backward_handles:
        h.remove()
    gradients.reverse()
    names.reverse()

    # compute scores
    scores = {}
    tmp_zip = list(zip(names, activations, corr_activations, gradients))
    for i, (name, clean_act, corr_act, grad) in enumerate(tmp_zip):
        if 'act' not in name:
            if i+1 < len(tmp_zip) and 'act' in tmp_zip[i+1][0]:
                act = model.processor.get_submodule(tmp_zip[i+1][0])
                score_matrix = (act(corr_act) - act(clean_act)).T @ grad
            else:
                score_matrix = (corr_act - clean_act).T @ grad
            scores[name] = score_matrix.T.detach() / data.num_nodes
    
    # clean up gpu memory
    model.zero_grad(set_to_none=True)

    return scores

def compute_eap_ig_scores(model: EncodeProcessDecode,
                          data: pyg.data.Data,
                          data_corr: pyg.data.Data,
                          loss: torch.nn.modules.loss._Loss,
                          steps: int=5) -> dict[str,torch.Tensor]:
    r'''
    Compute edge attribution patching with integrated gradients (EAP-IG) score
    for a computation edge :math:`(\psi_i, \psi_j)`.
     
      The EAP-IG score is given by
    ..math::
        \operatorname{EAP-IG}_{(i,j)}(x,x')
            = (z_i'-z_i)^\transpose \frac{1}{m}\sum_{k=1}^m \partials{\psi_j} 
                L\left(\Psi(x),\Psi\left(x'+\frac{k}{m}(x-x')\right)\right).
    
    where :math:`k` is the number of steps, :math:`x` is the clean data,
    :math:`x'` is the corrupted data, and
    :math:`z_i` and :math:`z_i'` are the activations of neuron :math:`\psi_i`
    on :math:`x` and :math:`x'`, respectively.

    Args:
        model (torch.nn.Module): GNN module to compute scores for.
        data (torch_geometric.data.Data): clean input Data object.
        data_corr (torch_geometric.data.Data): corrupted input Data object.
        loss (torch.nn.modules.loss._Loss): loss function to compute scores with.
        steps (int, optional): How many steps to use for IG computation. Using
            `1` is equivalent to EAP. (Default: 5)
        sigmoid_target (bool, optional): whether or not to apply the sigmoid function
            to the target for `loss`. Useful for binary classification losses
            (e.g. `torch.nn.modules.loss.BCEWithLogitsLoss`). (Default: `False`)
            
    Returns:
        Dictionary of edge names and score values.
    '''
    data_steps = []
    for step in range(1, steps):
        tmp_data = data_corr.clone()
        for input in data.inputs:
            tmp_data[input] += (step / steps) * (data[input] - data_corr[input])
        if data.edge_weight is not None:
            tmp_data.edge_weight += (step / steps) * (data.edge_weight - data_corr.edge_weight)
        if data.edge_attr is not None:
            tmp_data.edge_attr += (step / steps) * (data.edge_attr - data_corr.edge_attr)
        data_steps.append(tmp_data)
    data_steps.append(data_corr)

    activations = []
    corr_activations = []
    gradients = []
    names = []
    def _get_activations(name):
        def _activation_hook(module, input, output):
            names.append(name)
            activations.append(output.detach().clone())
        return _activation_hook
    def _corr_activation_hook(module, input, output):
        corr_activations.append(output.detach().clone())
    def _gradient_hook(module, grad_input, grad_output):
        gradients[-1].append(grad_input[0].detach().clone())
    
    with torch.no_grad():
        # get corrupted activations
        corr_handles = []
        for name, module in model.processor.named_modules():
            if _place_hook(module):
                corr_handles.append(module.register_forward_hook(_corr_activation_hook))
        out_corr = _apply_salsa_clrs_model(model, data_corr)
        for h in corr_handles:
            h.remove()
        
        # set forward hooks for clean activations
        clean_handles = []
        for name, module in model.processor.named_modules():
            if _place_hook(module):
                clean_handles.append(module.register_forward_hook(_get_activations(name)))
        out_clean = _apply_salsa_clrs_model(model, data)
        for h in clean_handles:
            h.remove()

    # set backward hooks
    backward_handles = []
    for name, module in model.processor.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook'):
            backward_handles.append(module.register_full_backward_hook(_gradient_hook))

    # get activations and gradients
    for step in range(steps):
        gradients.append([])
        data_steps[step].apply_(_ensure_requires_grad)
        out_step = _apply_salsa_clrs_model(model, data_steps[step])
        L = _apply_salsa_clrs_loss(out_corr, out_step, data, loss)
        L.backward()
        del out_step
        gc.collect()
    for h in backward_handles:
        h.remove()
    for grad_list in gradients:
        grad_list.reverse()
    grad_avgs = [sum(col) / steps for col in zip(*gradients)]

    # compute scores
    scores = {}
    tmp_zip = list(zip(names, activations, corr_activations, grad_avgs))
    for i, (name, clean_act, corr_act, grad) in enumerate(tmp_zip):
        if 'act' not in name:
            if i+1 < len(tmp_zip) and 'act' in tmp_zip[i+1][0]:
                act = model.processor.get_submodule(tmp_zip[i+1][0])
                score_matrix = (act(corr_act) - act(clean_act)).T @ grad
            else:
                score_matrix = (corr_act - clean_act).T @ grad
            scores[name] = score_matrix.T.detach() / data.num_nodes
    
    # clean up gradients
    model.zero_grad(set_to_none=True)

    return scores

class SALSACLRSComputationGraph(ComputationGraph):
    def __init__(self, model, special_modules=None):

        self.EncodeProcessDecode = model
        super().__init__(model.processor, special_modules=special_modules)

    @override
    def calculate_scores(self,
                        algorithm,
                        clean_data,
                        corrupted_data,
                        loss,
                        which='EAP',
                        **kwargs):
        '''
        Calculate edge attribution scores for a SALSACLRS EncodeProcessDecode model. 

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

        clean_loader = SALSACLRSDataLoader(clean_data, batch_size=32, shuffle=False, num_workers=1)
        corrupted_loader = SALSACLRSDataLoader(corrupted_data, batch_size=32, shuffle=False, num_workers=1)
        for data, data_corr in zip(clean_loader, corrupted_loader):
            data.task = algorithm
            data_corr.task = algorithm
            if hasattr(data, 'weights'):
                data.edge_attr = data.weights.unsqueeze(1)
                data_corr.edge_attr = data_corr.weights.unsqueeze(1)
            else:
                data.edge_attr = torch.zeros((data.num_edges, 1), device=self.EncodeProcessDecode.device)
                data_corr.edge_attr = torch.zeros((data_corr.num_edges, 1), device=self.EncodeProcessDecode.device)
            all_scores.append(score_function(self.EncodeProcessDecode, data, data_corr, loss, **kwargs))
        for key in all_scores[0].keys():
            avg_scores[key] = torch.mean(torch.stack([score_dict[key] for score_dict in all_scores]), 0)
        for key, score in avg_scores.items():
            for j in range(score.shape[1]):
                v = key + f'.{j}'
                for i, u in enumerate(self.predecessors(v)):
                    if i < score.shape[0]:
                        nx.set_edge_attributes(self, {(u,v) : {f'{which}_{algorithm}' : float(score[i,j].abs())}})
                    else:
                        m = re.match(r'.*?([0-9]+)$', u)
                        if m is None:
                            ind = 0
                        else:
                            ind = int(m.group(1))
                        nx.set_edge_attributes(self, {(u,v) : {f'{which}_{algorithm}' : float(score[ind,j].abs())}})
                for w in self.successors(v):
                    if self[v][w].get('addition', False):
                        nx.set_edge_attributes(self, {(v,w) : {f'{which}_{algorithm}' : 0.0}})
        for edge in self.edges:
            if f'{which}_{algorithm}' not in self.edges[edge]:
                nx.set_edge_attributes(self, {edge : {f'{which}_{algorithm}' : 0.0}})

class SALSACLRSCircuit(Circuit):

    def __init__(self,
                 model: torch.nn.Module,
                 G: Optional[ComputationGraph]=None,
                 K: int=10,
                 key: str='EAP',
                 as_view:bool=True,
                 use_abs:bool=True,
                 include_all_outputs:bool=False) -> None:
        
        self.EncodeProcessDecode = model
        super().__init__(model.processor, G, K, key, as_view, use_abs, include_all_outputs)
    
    @override
    def forward(self, data, **kwargs):
        self.EncodeProcessDecode.eval()
        self._apply_masks(default_one=False, invert=False)
        out, hints, hidden = _apply_salsa_clrs_model(self.EncodeProcessDecode, data, **kwargs)
        self._clear_masks()
        return out, hints, hidden

    @override
    def ablate_circuit(self,
                    data: pyg.data.Data,
                    **kwargs) -> torch.Tensor:
        self.EncodeProcessDecode.eval()
        self._apply_masks(default_one=True, invert=True)
        out, hints, hidden = _apply_salsa_clrs_model(self.EncodeProcessDecode, data, **kwargs)
        self._clear_masks()
        return out, hints, hidden