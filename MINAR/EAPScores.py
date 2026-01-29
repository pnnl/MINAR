import torch
from torch import Tensor
import torch_geometric
from .utils import _place_hook, _apply_model

def _ensure_requires_grad(t):
    '''
    Utility function to enforce `requires_grad` on `t`.
    '''
    if hasattr(t, 'requires_grad') and torch.is_floating_point(t):
        t.requires_grad_()

def compute_weight_grad_scores(
        model: torch.nn.Module,
        data: torch_geometric.data.Data,
        data_corr: torch_geometric.data.Data,
        loss: torch.nn.modules.loss._Loss) -> dict[str,Tensor]:
    r'''
    Compute score for a computation edge :math:`(\psi_i, \psi_j)` with weight
    :math:`w_{ij}` by
    .. math::
        \frac{\partial}{\partial w_{ij}} L(x_{\text{clean}}, x_{\text{corrupted}}).

    Args:
        model (torch.nn.Module): GNN module to compute scores for.
        data (torch_geometric.data.Data): clean input Data object.
        data_corr (torch_geometric.data.Data): corrupted input Data object.
        loss (torch.nn.modules.loss._Loss): loss function to compute scores with.

    Returns:
        Dictionary of edge names and score values.
    '''
    scores = {}
            
    out_corr = _apply_model(model, data_corr)
    out_clean = _apply_model(model, data)
    L = loss(out_corr, out_clean)
    L.backward()

    for name, module in model.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook') and hasattr(module, 'weight'):
            scores[name] = module.weight.grad.detach().clone().T / data.num_nodes
            
    # clean up gpu memory
    model.zero_grad(set_to_none=True)
    return scores

def compute_eap_scores(model: torch.nn.Module,
                       data: torch_geometric.data.Data,
                       data_corr: torch_geometric.data.Data,
                       loss: torch.nn.modules.loss._Loss,
                       sigmoid_target: bool=False) -> dict[str,Tensor]:
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
        model (torch.nn.Module): GNN module to compute scores for.
        data (torch_geometric.data.Data): clean input Data object.
        data_corr (torch_geometric.data.Data): corrupted input Data object.
        loss (torch.nn.modules.loss._Loss): loss function to compute scores with.
        sigmoid_target (bool, optional): whether or not to apply the sigmoid function
            to the target for `loss`. Useful for binary classification losses
            (e.g. `torch.nn.modules.loss.BCEWithLogitsLoss`). (Default: `False`)

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
    for name, module in model.named_modules():
        if _place_hook(module):
            corr_handles.append(module.register_forward_hook(_activation_hook))
    out_corr = _apply_model(model, data_corr)
    for h in corr_handles:
        h.remove()
    
    # set forward hooks for clean activations
    clean_handles = []
    for name, module in model.named_modules():
        if _place_hook(module):
            clean_handles.append(module.register_forward_hook(_corr_activation_hook))

    # set backward hooks
    backward_handles = []
    for name, module in model.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook'):
            backward_handles.append(module.register_full_backward_hook(_get_gradients(name)))

    # get activations and gradients
    out_clean = _apply_model(model, data)
    if sigmoid_target:
        out_clean = out_clean.sigmoid()
    L = loss(out_corr, out_clean)
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
                act = model.get_submodule(tmp_zip[i+1][0])
                score_matrix = (act(corr_act) - act(clean_act)).T @ grad
            else:
                score_matrix = (corr_act - clean_act).T @ grad
            scores[name] = score_matrix.T / data.num_nodes
    
    # clean up gpu memory
    model.zero_grad(set_to_none=True)

    return scores

def compute_eap_ig_scores(model: torch.nn.Module,
                          data: torch_geometric.data.Data,
                          data_corr: torch_geometric.data.Data,
                          loss: torch.nn.modules.loss._Loss,
                          steps: int=5,
                          sigmoid_target: bool=False) -> dict[str,Tensor]:
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
    data.apply_(_ensure_requires_grad)
    data_corr.apply_(_ensure_requires_grad)
    data_steps = []
    for step in range(1, steps):
        tmp_data = data_corr.clone()
        tmp_data.x += (step / steps) * (data.x - data_corr.x)
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
    
    # get corrupted activations
    corr_handles = []
    for name, module in model.named_modules():
        if _place_hook(module):
            corr_handles.append(module.register_forward_hook(_corr_activation_hook))
    out_corr = _apply_model(model, data_corr).detach()
    for h in corr_handles:
        h.remove()
    
    # set forward hooks for clean activations
    clean_handles = []
    for name, module in model.named_modules():
        if _place_hook(module):
            clean_handles.append(module.register_forward_hook(_get_activations(name)))
    out_clean = _apply_model(model, data).detach()
    for h in clean_handles:
        h.remove()

    # set backward hooks
    backward_handles = []
    for name, module in model.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook'):
            backward_handles.append(module.register_full_backward_hook(_gradient_hook))

    # get activations and gradients
    for step in range(steps):
        gradients.append([])
        out_step = _apply_model(model, data_steps[step])
        if sigmoid_target:
            out_step = out_step.sigmoid()
        L = loss(out_corr, out_step)
        L.backward()
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
                act = model.get_submodule(tmp_zip[i+1][0])
                score_matrix = (act(corr_act) - act(clean_act)).T @ grad
            else:
                score_matrix = (corr_act - clean_act).T @ grad
            scores[name] = score_matrix.T / data.num_nodes
    
    # clean up gpu memory
    model.zero_grad(set_to_none=True)

    return scores