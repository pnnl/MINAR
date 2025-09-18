import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Optional, Union, List

def multiplicative_loss(
        input: Tensor,
        target: Tensor,
        reduction: str = "sum",
        weight: Optional[Tensor] = None,):
    '''
    The continuous multiplicative loss from https://arxiv.org/abs/2503.19173.

    See :class:`~MultiplicativeLoss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Multipliative loss.
    '''
    errors = (torch.ones_like(input) - target / input).abs()
    if weight is not None:
        errors = errors * weight

    if reduction == "none":
        return errors
    elif reduction == "sum":
        return torch.sum(errors)
    elif reduction == "mean":
        return torch.sum(errors) / torch.sum(weight)

class MultiplicativeLoss(_Loss):
    '''
    The continuous multiplicative loss from https://arxiv.org/abs/2503.19173.
    
    For an input :math:`x` and target :math:`y`, the sum-reduced loss is given by
    .. math::
        \sum_i \left| 1 - \frac{y_i}{x_i} \right|.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'sum'``
    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = "sum") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return multiplicative_loss(input, target, reduction=self.reduction)
    
def joint_loss(input: Tensor,
               target: Tensor,
               losses: List[_Loss],
               reduction: str = "sum",
               weight: Union[Tensor, List] = None
               ) -> Tensor:
    '''
    Function to apply different losses to different output coordinates.

    See :class:`~JointLoss` for details.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        losses (List[_Loss]): List of loss functions to apply.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                                   'sum': the output will be summed. 'none': no reduction will be applied.
                                   Default: 'mean'.
        weight (Tensor, optional): Weights for each sample. Default: None.
    
    Returns:
        Tensor: Joint losses (optionally reduced)
    '''
    errors = torch.stack([losses[i](input[:,i], target[:,i]) for i in range(len(losses))])
    if weight is not None:
        errors = errors * weight

    if reduction == "none":
        return errors
    elif reduction == "sum":
        return torch.sum(errors)
    elif reduction == "mean":
        return torch.sum(errors) / torch.sum(weight)

class JointLoss(_Loss):
    '''
    Class to apply different losses to different output coordinates.

    If `losses` contains :math:`k` losses :math:`[\ell_1, \dots, \ell_k]` and
    input :math:`x` and target :math:`y` both consist of :math:`k` coordinates
    :math:`[x_1, \dots, x_k]` and :math:`[y_1, \dots, y_k]`,
    the unweighted, unreduced loss can be described as
    .. math::
        \ell(x,y) = L = [\ell_1(x_1,y_1),\dots,\ell_k(x_k,y_k)].

    Args:
        losses (List[_Loss]): A list of loss functions, one per output coordinate.
        apply_sigmoid (Union[Tensor, List]): A flag for each loss describing whether or not
            to apply a sigmoid function to the target for binary classification tasks.
        weight (Union[Tensor, List]): How to weight each loss function in `losses`.
            If `None`, each loss will be equally weighted. (default: None)
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'sum'``
    '''
    def __init__(self,
                 losses: List[_Loss],
                 apply_sigmoid: Union[Tensor, List],
                 weight: Union[Tensor, List]=None,
                 size_average: Optional[bool]=None,
                 reduce: Optional[bool]=None,
                 reduction: str = "sum") -> None:
        super().__init__(size_average, reduce, reduction)
        self.losses = losses
        self.apply_sigmoid = apply_sigmoid
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        tmp_target = torch.empty_like(target, device=target.device)
        for i, is_class in enumerate(self.apply_sigmoid):
            if is_class:
                tmp_target[:,i] = target[:,i].sigmoid()
            else:
                tmp_target[:,i] = target[:,i]

        return joint_loss(input, tmp_target, self.losses, reduction=self.reduction, weight=self.weight)