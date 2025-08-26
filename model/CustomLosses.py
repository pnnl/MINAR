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