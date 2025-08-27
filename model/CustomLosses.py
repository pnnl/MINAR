import torch
from torch import Tensor
from typing import Optional

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
    '''
    The continuous multiplicative loss from https://arxiv.org/abs/2503.19173 
    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = "sum") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return multiplicative_loss(input, target, reduction=self.reduction)
    
def joint_loss(input, target,
               criteria,
               reduction = "sum",
               weight = None
               ):
    errors = torch.stack([criteria[i](input[:,i], target[:,i]) for i in range(len(criteria))])
    if weight is not None:
        errors = errors * weight

    if reduction == "none":
        return errors
    elif reduction == "sum":
        return torch.sum(errors)
    elif reduction == "mean":
        return torch.sum(errors) / torch.sum(weight)

class JointLoss(torch.nn.modules.loss._Loss):
    def __init__(self, criteria, is_classification, weight=None, size_average=None, reduce=None, reduction: str = "sum") -> None:
        super().__init__(size_average, reduce, reduction)
        self.criteria = criteria
        self.is_classification = is_classification
        self.weight = weight

    def forward(self, input, target):
        tmp_target = torch.empty_like(target, device=target.device)
        for i, is_class in enumerate(self.is_classification):
            if is_class:
                tmp_target[:,i] = target[:,i].sigmoid()
            else:
                tmp_target[:,i] = target[:,i]

        return joint_loss(input, tmp_target, self.criteria, reduction=self.reduction, weight=self.weight)