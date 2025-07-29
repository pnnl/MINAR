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
    def _get_activations(name):
        def _activation_hook(module, input, output):
            activations.append(output.detach().clone())
        return _activation_hook
    def _get_corr_activations(name):
        def _corr_activation_hook(module, input, output):
            corr_activations.append(output.detach().clone())
        return _corr_activation_hook
    def _get_gradients(name):
        def _gradient_hook(module, grad_input, grad_output):
            gradients.append(grad_input[0].detach().clone())
            names.append(name)
        return _gradient_hook
    
    # get corrupted activations
    corr_handles = []
    for name, module in model.named_modules():
        if _place_hook(module):
            corr_handles.append(module.register_forward_hook(_get_corr_activations(name)))
    out_corr = _apply_model(model, data_corr)
    for h in corr_handles:
        h.remove()
    
    # set forward hooks for clean activations
    clean_handles = []
    for name, module in model.named_modules():
        if _place_hook(module):
            clean_handles.append(module.register_forward_hook(_get_activations(name)))

    # set backward hooks
    backward_handles = []
    for name, module in model.named_modules():
        if _place_hook(module, hook_str='register_full_backward_hook'):
            backward_handles.append(module.register_full_backward_hook(_get_gradients(name)))

    # get activations and gradients
    out_clean = _apply_model(model, data)
    L = loss(out_clean, out_corr)
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