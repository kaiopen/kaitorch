r'''
Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without
Convolutions

'''
import torch
from torch import nn
import torch.nn.functional as F

from ..nn.conv import Conv2dBlock, conv2d
from ..typing import TorchTensor, TorchFloat


class SRA(nn.Module):
    r'''Spatial-Reduction Attention for 2D data.

    #### Args;
    - in_channels: the number of input channels.
    - reduction: reduction ratio.
    - num_head: number of heads in MHA (Multi-Head Attention).
    - dropout: dropout probability.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, in_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        reduction: int = 1,
        num_head: int = 8,
        dropout: float = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._num_head = num_head
        self._head_channels = in_channels // num_head
        self._scale = self._head_channels ** -0.5

        self._kv = nn.Sequential(
            Conv2dBlock(
                in_channels, in_channels, reduction, reduction,
                normalization='groupnorm',
                normalization_kw={
                    'num_groups': 1,
                    'num_channels': in_channels
                },  # same as layernorm
                mode='cn'
            ),
            conv2d(in_channels, in_channels * 2, 1, bias=True)
        )

        self._q = conv2d(in_channels, in_channels, 1, bias=True)

        self._drop = nn.Dropout(dropout)
        self._proj = conv2d(in_channels, in_channels, bias=True)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        batch_size, channels, h, w = x.shape

        kv = self._kv(x).reshape(
            batch_size, 2, self._num_head, self._head_channels, -1
        )  # (B, 2, NH, HC, L)

        return self._proj(
            torch.matmul(
                self._drop(
                    F.softmax(
                        torch.matmul(
                            self._q(x).reshape(
                                batch_size,
                                self._num_head,
                                self._head_channels,
                                -1
                            ).permute(0, 1, 3, 2),  # q (B, NH, HW, HC)
                            kv[:, 0]  # k (B, NH, HC, L)
                        ) * self._scale,
                        dim=-1
                    )  # atten (B, NH, HW, L)
                ),
                kv[:, 1].transpose(2, 3)  # v (B, NH, L, HC)
            ).transpose(2, 3).reshape(batch_size, channels, h, w)
        )
