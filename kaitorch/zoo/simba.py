r'''
SiMBA: Simplified Mamba-based Architecture for Vision and Multivariate Time
Series

'''
from typing import Sequence, Tuple, Union
import math
import random

import torch
from torch import fft
from torch import nn
import torch.nn.functional as F

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock, conv2d
from .droppath import DropPath
from .mamba import Mamba


def random_box(height: int, width: int):
    rate = math.sqrt(1 - random.random())
    h = int(height * rate) // 2
    w = int(width * rate) // 2
    x = random.randint(0, height)
    y = random.randint(0, width)

    return max(0, min(x - h, height)), max(0, min(y - w, width)), \
        max(0, min(x + h, height)), max(0, min(y + w, width))


class EMM(nn.Module):
    def __init__(
        self,
        num_block: int,
        channels: int,
        scale: float = 0.2,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._w_r = nn.Parameter(
            torch.randn(num_block, channels, channels) * scale
        )
        self._w_i = nn.Parameter(
            torch.randn(num_block, channels, channels) * scale
        )
        self._b_r = nn.Parameter(torch.randn(num_block, channels) * scale)
        self._b_i = nn.Parameter(torch.randn(num_block, channels) * scale)

    @staticmethod
    def mm(input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(
        self, real: TorchTensor[TorchFloat], imag: TorchTensor[TorchFloat]
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        # R^l = EMM(R^(l - 1) * w_r) - EMM(I^(l - 1) * w_i) + b_r
        # I^l = EMM(R^(l - 1) * w_i) + EMM(I^(l - 1) * w_r) + b_i
        return (
            self.mm(real, self._w_r) - self.mm(imag, self._w_i) + self._b_r,
            self.mm(real, self._w_i) + self.mm(imag, self._w_r) + self._b_i
        )


class EinFFT(nn.Module):
    def __init__(self, in_channels: int, num_block: int = 4, *args, **kwargs):
        super().__init__()
        self._n = num_block
        if 0 != in_channels % self._n:
            raise ValueError(
                '`in_channels` should be divided exactly by `num_block`.'
            )
        self._c = in_channels // self._n

        self._emm_0 = EMM(self._n, self._c)
        self._emm_1 = EMM(self._n, self._c)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        B, N, C = x.shape
        x = fft.fft2(x.view(B, N, self._n, self._c), dim=(1, 2), norm='ortho')
        real, imag = self._emm_0(x.real, x.imag)
        return fft.ifft2(
            torch.view_as_complex(
                F.softshrink(
                    torch.stack(
                        self._emm_1(F.relu(real), F.relu(imag)), dim=-1
                    ),
                    lambd=0.01
                )
            ),
            dim=(1, 2),
            norm='ortho'
        ).float().reshape(B, N, C)


class Block(nn.Module):
    def __init__(
        self, in_channels: int, drop: float = 0., *args, **kwargs
    ) -> None:
        super().__init__()
        self._attn = Mamba(in_channels, state_channels=64)
        self._mlp = EinFFT(in_channels)
        self._drop = DropPath(drop) if drop > 0 else nn.Identity()
        self._norm_attn = nn.LayerNorm(in_channels, eps=1e-6)
        self._norm_mlp = nn.LayerNorm(in_channels, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            h, w = m.kernel_size
            m.weight.data.normal_(
                0, math.sqrt(2.0 / ((h * w * m.out_channels) // m.groups))
            )
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = x + self._drop(self._attn(self._norm_attn(x)))
        return x + self._drop(self._mlp(self._norm_mlp(x)))


class SiMBA(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the numbers of the output channels of each down sampling.
        Due to the stem block doing twice down sampling, its length should be
        equal to one plus the number of stages.
    - num_category
    - depths: number of SiMBA blocks in each stage. Its length is the number of
        stages.
    - drop: drop path rate.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    For a four-stage network, the forward processes are
    `Stem -> NBN -> Down -> NBN -> Down -> NBN -> Down -> NB -> Cls -> N ->
    Heads`, where `Stem` is a stem block with 2 down sampling, `Down` is a
    down sampling, `B` is a sequence of stacked SiMBA blocks, `N` is a
    layer normalization, `Cls` is classification token, `Heads` are two
    heads.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: input tensor. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    For training:
    - Classification results. Its shape is `(B, num_category)`.
    - Auxiliary output. Its shape is
        `(B, H // scale * W // scale, num_category)`, where
        `scale = len(out_channels) = len(depths) + 1`.
    - Random mixed box. It is in the form of `(x of the top left point, y of
        the top left point, x of the bottom right point ,y of the bottom right
        point)`.
    Or
    - Classification results. Its shape is `(B, num_category)`.

    '''
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Sequence[int] = (32, 64, 128, 320, 448),
        num_category: int = 1000,
        depths: Sequence[int] = (3, 4, 6, 3),
        drop: float = 0.1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        num_stage = len(depths)

        if len(out_channels) != num_stage + 1:
            raise ValueError(
                'len(out_channels) should be equal to len(depths) + 1.'
            )

        h = out_channels[0]
        o = out_channels[1]
        out_channels = out_channels[1:]
        self._stem = nn.Sequential(
            Conv2dBlock(in_channels, h, 7, 2, 3, padding_mode=padding_mode),
            Conv2dBlock(h, h, 3, 1, 1, padding_mode=padding_mode),
            Conv2dBlock(h, h, 3, 1, 1, padding_mode=padding_mode),
            conv2d(
                h, o, 3, 2, 1,
                bias=True,
                padding_mode=padding_mode
            )
        )

        self._downs = nn.ModuleList()
        for i in range(1, num_stage):
            c = o
            o = out_channels[i]
            self._downs.append(
                conv2d(c, o, 3, 2, 1, padding_mode=padding_mode)
            )

        rates = [x.item() for x in torch.linspace(0, drop, sum(depths))]
        num_stage -= 1
        j = 0
        self._blocks = nn.ModuleList()
        for i in range(num_stage):
            c = out_channels[i]
            blocks = nn.Sequential(nn.LayerNorm(c, eps=1e-6))
            for _ in range(depths[i]):
                blocks.append(Block(c, rates[j]))
                j += 1
            blocks.append(nn.LayerNorm(c, eps=1e-6))
            self._blocks.append(blocks)

        c = out_channels[num_stage]
        self._block = nn.Sequential(nn.LayerNorm(c, eps=1e-6))
        for _ in range(depths[num_stage]):
            self._block.append(Block(c, rates[j]))
            j += 1
        self._token = Block(c)
        self._norm = nn.LayerNorm(c, eps=1e-6)

        self._head = nn.Linear(c, num_category)
        self._aux = nn.Linear(c, num_category)

        self._scale = 2 ** num_stage
        self._range = range(num_stage)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            h, w = m.kernel_size
            m.weight.data.normal_(
                0, math.sqrt(2.0 / ((h * w * m.out_channels) // m.groups))
            )
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: TorchTensor[TorchFloat]) -> Union[
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchFloat],
            Tuple[int, int, int, int]
        ],
        TorchTensor[TorchFloat]
    ]:
        batch_size = len(x)
        x = self._stem(x)  # Stem (B, C, H, W)
        _, c, h, w = x.shape

        # Do mix token. See TokenLabeling for detail.
        if self.training:
            # Random box
            x_1, y_1, x_2, y_2 = random_box(
                h // self._scale, w // self._scale
            )
            _x_1 = x_1 * self._scale
            _x_2 = x_2 * self._scale
            _y_1 = y_1 * self._scale
            _y_2 = y_2 * self._scale
            x[:, :, _x_1: _x_2, _y_1: _y_2] = x.flip(0)[
                :, :, _x_1: _x_2, _y_1: _y_2
            ]

        for i in self._range:
            # Flatten -> NBN -> Reshape -> Down
            x = self._downs[i](
                self._blocks[i](
                    x.flatten(2).transpose(1, 2)
                ).reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
            )  # (B, C, H, W)
            _, c, h, w = x.shape

        # Flatten -> NB
        x = self._block(x.flatten(2).transpose(1, 2))  # (B, HW, C)

        # CLS -> N
        x = self._norm(
            self._token(torch.cat((x.mean(dim=1, keepdim=True), x), dim=1))
        )  # (B, 1 + HW, C)

        cls = self._head(x[:, 0])  # (B, num_cat)
        aux = self._aux(x[:, 1:])  # (B, HW, num_cat)

        if self.training:
            _, l, c = aux.shape
            aux = aux.view(batch_size, h, w, c)
            aux[:, x_1: x_2, y_1: y_2] = aux.flip(0)[:, x_1: x_2, y_1: y_2]
            return cls, aux.view(batch_size, l, c), (x_1, y_1, x_2, y_2)

        return cls + 0.5 * aux.max(1)[0]
