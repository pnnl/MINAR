import torch
from .utils import _place_hook, _apply_model

def _ensure_requires_grad(t):
    if hasattr(t, 'requires_grad') and torch.is_floating_point(t):
        t.requires_grad_()

def compute_eap_scores(model, data, data_corr, loss):
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
            scores[name] = score_matrix.T
    
    # clean up gpu memory
    model.zero_grad(set_to_none=True)

    return scores

def compute_eap_ig_scores(model, data, data_corr, loss, steps=5):
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
            scores[name] = score_matrix.T
    
    # clean up gpu memory
    model.zero_grad(set_to_none=True)

    return scores