r'''
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time
object detectors

'''
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock
from .cspnet import SPPCSPC
from .panet import UpsampleE
from .yolov2 import ReOrg


class MaxPool(nn.Module):
    r'''
    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`

    #### Returns:
    - Feature map. Its shape is `(B, out_channels // 2 * 2, H // 2, W // 2)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        out_channels //= 2
        self._i0o1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i0o2 = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return torch.cat((self._i0o1(x), self._i0o2(x)), dim=1)


class MaxPoolE(nn.Module):
    r'''Max pool and concatenate with an extral feature map.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map that will be sampled. Its shape should be
        `(B, in_channels[0], H, W)`.
    - e: extra feature map. Its shape should be
        `(B, in_channels[1], H // 2, W // 2)`.

    #### Returns:
    - Feature map. Its shape is
        `(B, out_channels // 4 * 2 + in_channels[1], H // 2, W // 2)`.

    '''
    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        out_channels //= 4
        self._i0o1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i0o2 = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

    def forward(
        self, x: TorchTensor[TorchFloat], e: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return torch.cat((self._i0o1(x), self._i0o2(x), e), dim=1)


class DownsampleE(nn.Module):
    r'''Downsample and concatenate with an extra input.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map that will be sampled. Its shape should be
        `(B, in_channels, H, W)`,
    - e: feature map. Its shape should be `(B, C, H // 2, W // 2)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels + C, H // 2, W // 2)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._ds = Conv2dBlock(
            in_channels, out_channels, 3, 2, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(
        self, x: TorchTensor[TorchFloat], e: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return torch.cat((self._ds(x), e), dim=1)


class Shortcut(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._conv_0 = Conv2dBlock(
            in_channels[0], out_channels, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._conv_1 = Conv2dBlock(
            in_channels[1], out_channels, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(
        self, x_0: TorchTensor[TorchFloat], x_1: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return self._conv_0(x_0) + self._conv_1(x_1)


class ELAN4(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i3o4 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 4, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        return self._icoo(
            torch.cat((self._i0o1(x), f_2, f_3, self._i3o4(f_3)), dim=1)
        )


class ELAN5(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i3o4 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i4o5 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 5, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        f_4 = self._i3o4(f_3)
        return self._icoo(
            torch.cat((self._i0o1(x), f_2, f_3, f_4, self._i4o5(f_4)), dim=1)
        )


class ELAN6(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i3o4 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i4o5 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i5o6 = nn.Sequential(
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 6, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        f_4 = self._i3o4(f_3)
        f_5 = self._i4o5(f_4)
        return self._icoo(
            torch.cat(
                (self._i0o1(x), f_2, f_3, f_4, f_5, self._i5o6(f_5)), dim=1
            )
        )


class ELANTiny(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = Conv2dBlock(
            hidden_channels, hidden_channels, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i3o4 = Conv2dBlock(
            hidden_channels, hidden_channels, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 4, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        return self._icoo(
            torch.cat((self._i0o1(x), f_2, f_3, self._i3o4(f_3)), dim=1)
        )


class ELANH6(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        half = hidden_channels // 2
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = Conv2dBlock(
            hidden_channels, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i3o4 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i4o5 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i5o6 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 4, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        f_4 = self._i3o4(f_3)
        f_5 = self._i4o5(f_4)
        return self._icoo(
            torch.cat(
                (self._i0o1(x), f_2, f_3, f_4, f_5, self._i5o6(f_5)), dim=1
            )
        )


class ELANH8(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        half = hidden_channels // 2
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = Conv2dBlock(
            hidden_channels, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i3o4 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i4o5 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i5o6 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i6o7 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i7o8 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 5, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        f_4 = self._i3o4(f_3)
        f_5 = self._i4o5(f_4)
        f_6 = self._i5o6(f_5)
        f_7 = self._i6o7(f_6)
        return self._icoo(
            torch.cat(
                (self._i0o1(x), f_2, f_3, f_4, f_5, f_6, f_7, self._i7o8(f_7)),
                dim=1
            )
        )


class ELANH10(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        half = hidden_channels // 2
        self._i0o1 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i0o2 = Conv2dBlock(
            in_channels, hidden_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i2o3 = Conv2dBlock(
            hidden_channels, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i3o4 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i4o5 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i5o6 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i6o7 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i7o8 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i8o9 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._i9o10 = Conv2dBlock(
            half, half, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._icoo = Conv2dBlock(
            hidden_channels * 6, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f_2 = self._i0o2(x)
        f_3 = self._i2o3(f_2)
        f_4 = self._i3o4(f_3)
        f_5 = self._i4o5(f_4)
        f_6 = self._i5o6(f_5)
        f_7 = self._i6o7(f_6)
        f_8 = self._i7o8(f_7)
        f_9 = self._i8o9(f_8)
        return self._icoo(
            torch.cat(
                (
                    self._i0o1(x),
                    f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9,
                    self._i9o10(f_9)
                ),
                dim=1
            )
        )


class YOLOv7(nn.Module):
    r'''YOLOv7 backbone.

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Input feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature maps. Their shapes are `(B, 512, H // 8, W // 8)`,
        `(B, 1024, H // 16, W // 16)`, `(B, 1024, H // 32, W // 32)`.

    '''
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer_0 = nn.Sequential(
            Conv2dBlock(
                in_channels, 32, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                32, 64, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                64, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_1 = nn.Sequential(
            Conv2dBlock(
                64, 128, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                128, 64, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_2 = nn.Sequential(
            MaxPool(
                256, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                256, 128, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_3 = nn.Sequential(
            MaxPool(
                512, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                512, 256, 1024,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_4 = nn.Sequential(
            MaxPool(
                1024, 1024,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                1024, 256, 1024,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> Tuple[
        TorchTensor[TorchFloat], TorchTensor[TorchFloat],
        TorchTensor[TorchFloat], TorchTensor[TorchFloat]
    ]:
        p_3 = self._layer_2(self._layer_1(self._layer_0(x)))
        p_4 = self._layer_3(p_3)
        return p_3, p_4, self._layer_4(p_4)


class Neck(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Feature maps. Their shapes should be `(B, in_channels[0], H, W)`,
        `(B, in_channels[1], H // 2, W // 2)`,
        `(B, in_channels[2], H // 4, W // 4)`.

    #### Returns:
    - Feature maps in different scales. Their shapes are
        `(B, 128, H, W)`, `(B, 256, H // 2, W // 2)`,
        `(B, 512, H // 4, W // 4)`.
    - Auxiliary feature maps in different scales. Their shapes are
        `(B, 128, H, W)`, `(B, 256, H // 2, W // 2)`,
        `(B, 512, H // 4, W // 4)`.

    '''
    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (512, 1024, 1024),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        c_3, c_4, c_5 = in_channels

        self._sppcspc = SPPCSPC(
            c_5, 512,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )  # u_5

        self._up_i5o4 = nn.Sequential(
            UpsampleE(
                (512, c_4), 512,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                512, 256, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # u_4

        self._up_i4o3 = nn.Sequential(
            UpsampleE(
                (256, c_3), 256,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                256, 128, 128,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # u_3

        self._ds_i3o4 = nn.Sequential(
            MaxPoolE(
                (128, 256), 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                512, 256, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # d_4

        self._ds_i4o5 = nn.Sequential(
            MaxPoolE(
                (256, 512), 1024,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                1024, 512, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # d_5

    def forward(
        self,
        x: Tuple[
            TorchTensor[TorchFloat], TorchTensor[TorchFloat],
            TorchTensor[TorchFloat], TorchTensor[TorchFloat],
            TorchTensor[TorchFloat], TorchTensor[TorchFloat]
        ]
    ) -> Tuple[
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchFloat],
            TorchTensor[TorchFloat]
        ],
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchFloat],
            TorchTensor[TorchFloat]
        ]
    ]:
        p_3, p_4, p_5 = x
        u_5 = self._sppcspc(p_5)
        u_4 = self._up_i5o4(u_5, p_4)
        u_3 = self._up_i4o3(u_4, p_3)
        d_4 = self._ds_i3o4(u_3, u_4)
        return (u_3, d_4, self._ds_i4o5(d_4, u_5)), (u_3, u_4, u_5)


class YOLOv7w6(nn.Module):
    r'''YOLOv7-w6 backbone.

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Input feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature maps. Their shapes are `(B, 256, H // 8, W // 8)`,
        `(B, 512, H // 16, W // 16)`, `(B, 768, H // 32, W // 32)`,
        `(B, 1024, H // 64, W // 64)`,

    '''
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer_0 = nn.Sequential(
            ReOrg(),
            Conv2dBlock(
                in_channels * 4, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_1 = nn.Sequential(
            Conv2dBlock(
                64, 128, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                128, 64, 128,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_2 = nn.Sequential(
            Conv2dBlock(
                128, 256, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                256, 128, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_3 = nn.Sequential(
            Conv2dBlock(
                256, 512, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                512, 256, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_4 = nn.Sequential(
            Conv2dBlock(
                512, 768, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                768, 384, 768,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._layer_5 = nn.Sequential(
            Conv2dBlock(
                768, 1024, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELAN4(
                1024, 512, 1024,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> Tuple[
        TorchTensor[TorchFloat], TorchTensor[TorchFloat],
        TorchTensor[TorchFloat], TorchTensor[TorchFloat]
    ]:
        p_3 = self._layer_2(self._layer_1(self._layer_0(x)))
        p_4 = self._layer_3(p_3)
        p_5 = self._layer_4(p_4)
        return p_3, p_4, p_5, self._layer_5(p_5)


class Neckw6(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Feature maps. Their shapes should be `(B, in_channels[0], H, W)`,
        `(B, in_channels[1], H // 2, W // 2)`,
        `(B, in_channels[2], H // 4, W // 4)`.
        `(B, in_channels[3], H // 8, W // 8)`.

    #### Returns:
    - Feature maps in different scales. Their shapes are
        `(B, 128, H, W)`, `(B, 256, H // 2, W // 2)`,
        `(B, 384, H // 4, W // 4)`, `(B, 512, H // 8, W // 8)`.
    - Auxiliary feature maps in different scales. Their shapes are
        `(B, 128, H, W)`, `(B, 256, H // 2, W // 2)`,
        `(B, 384, H // 4, W // 4)`, `(B, 512, H // 8, W // 8)`.

    '''
    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (256, 512, 768, 1024),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        c_3, c_4, c_5, c_6 = in_channels

        self._sppcspc = SPPCSPC(
            c_6, 512,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )  # u_6

        self._up_i6o5 = nn.Sequential(
            UpsampleE(
                (512, c_5), 768,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                768, 384, 384,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # u_5

        self._up_i5o4 = nn.Sequential(
            UpsampleE(
                (384, c_4), 512,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                512, 256, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # u_4

        self._up_i4o3 = nn.Sequential(
            UpsampleE(
                (256, c_3), 256,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                256, 128, 128,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # u_3

        self._ds_i304 = nn.Sequential(
            DownsampleE(
                128, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                512, 256, 256,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # d_4

        self._ds_i4o5 = nn.Sequential(
            DownsampleE(
                256, 384,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                768, 384, 384,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # d_5

        self._ds_i5o6 = nn.Sequential(
            DownsampleE(
                384, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            ELANH6(
                1024, 512, 512,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )  # d_6

    def forward(
        self,
        x: Tuple[
            TorchTensor[TorchFloat], TorchTensor[TorchFloat],
            TorchTensor[TorchFloat], TorchTensor[TorchFloat]
        ]
    ) -> Tuple[
        Tuple[
            TorchTensor[TorchFloat], TorchTensor[TorchFloat],
            TorchTensor[TorchFloat], TorchTensor[TorchFloat]],
        Tuple[
            TorchTensor[TorchFloat], TorchTensor[TorchFloat],
            TorchTensor[TorchFloat], TorchTensor[TorchFloat]
        ]
    ]:
        p_3, p_4, p_5, p_6 = x
        u_6 = self._sppcspc(p_6)
        u_5 = self._up_i6o5(u_6, p_5)
        u_4 = self._up_i5o4(u_5, p_4)
        u_3 = self._up_i4o3(u_4, p_3)
        d_4 = self._ds_i304(u_3, u_4)
        d_5 = self._ds_i4o5(d_4, u_5)
        return (u_3, d_4, d_5, self._ds_i5o6(d_5, u_6)), (u_3, u_4, u_5, u_6)
