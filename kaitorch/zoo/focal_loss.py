import torch

from ..typing import TorchTensor, TorchFloat


def focal_loss(
    x: TorchTensor[TorchFloat],
    target: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''A modified focal loss from CenterNet.

    l = -\frac{1}{N} \sum_{xyc}
        \begin{cases}
            (1 - t_{xyc})^{2}log(t_{xyc})          & \text{if } p_{xyc} = 1, \\
            (1 - p_{xyc})^{4}t_{xyc}^{2}log(1-t_{xyc}) & \text{otherwise}.
        \end{cases}

    #### Args:
    - x: the prediction results from a model.
    - target: It should share the same shape with the prediction results.

    #### Returns:
    - a loss value.

    '''
    mask_pos = target >= 1 - 1e-4  # target.eq(1)
    mask_neg = torch.logical_and(torch.logical_not(mask_pos), target >= 0)

    pred_neg = x[mask_neg]
    loss_neg = torch.sum(
        torch.log(1 - pred_neg)
        * torch.pow(pred_neg, 2)
        * torch.pow(1 - target[mask_neg], 4)
    )

    pred_pos = x[mask_pos]
    num_pos = mask_pos.sum()
    if 0 == num_pos:
        return -loss_neg
    return -(
        torch.sum(torch.log(pred_pos) * torch.pow(1 - pred_pos, 2)) + loss_neg
    ) / num_pos
