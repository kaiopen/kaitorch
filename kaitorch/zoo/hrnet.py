r'''
Deep High-Resolution Representation Learning for Human Pose Estimation

'''
from typing import Any, Dict, List, Sequence, Union

from torch import nn

from ..nn.conv import Conv2dBlock
from ..typing import TorchTensor, TorchFloat
from .resnet import BasicBlock, Bottleneck


BLOCK = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
}


class Transition(nn.Module):
    r'''Transition from HRNet.

    #### Args:
    - in_channels: sequence of input channels. Given the length of it is `N`.
    - out_channels: sequence of output channels. Given the length of it is `M`.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.

    #### Methods:
    - forward

    ## forward
    #### Args:
    -x: a sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    #### Returns:
    - A list of feature maps. If `M >= N`, the length of the list is `M`. The
        shapes of the feature maps are `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{M - 1}, H // 2^M, W // 2^M)`
        where `C_0`, `C_1`, ..., `C_(M-1)` are in `out_channels`. If `M < N`,
        the length of the list is `N`. The shapes of the feature maps are
        `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{M - 1}, H // 2^M, W // 2^M)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],  # (18, 36)
        out_channels: Sequence[int],  # (18, 36, 72)
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        num_in = len(in_channels)
        self._range_in = range(num_in)
        self._range_out = range(num_in, len(out_channels))

        self._trans = nn.ModuleList()
        for c_in, c_out in zip(in_channels, out_channels):
            self._trans.append(
                nn.Identity()
                if c_in == c_out
                else nn.Sequential(
                    Conv2dBlock(
                        c_in, c_out, 1, 1, 0,
                        padding_mode=padding_mode,
                        activation=activation,
                        activation_kw=activation_kw
                    ),
                    Conv2dBlock(
                        c_out, c_out, 3, 1, 1,
                        padding_mode=padding_mode,
                        activation=activation,
                        activation_kw=activation_kw
                    )
                )
            )

        # Downsamples for new branches.
        c_in = in_channels[-1]
        for dst in self._range_out:
            ds = nn.Sequential()
            for _ in range(num_in, dst):
                ds.append(
                    Conv2dBlock(
                        c_in, c_in, 3, 2, 1,
                        padding_mode=padding_mode,
                        activation=activation,
                        activation_kw=activation_kw
                    )
                )
            ds.append(
                Conv2dBlock(
                    c_in, out_channels[dst], 3, 2, 1,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )
            self._trans.append(ds)

    def forward(
        self, x: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        last = x[-1]
        for i in self._range_out:
            x.append(self._trans[i](last))

        for i in self._range_in:
            x[i] = self._trans[i](x[i])

        return x


class Branches(nn.Module):
    r'''

    #### Args:
    - in_channels: number of channels. Given the length of it is `N`.
    - blocks: names of blocks for each branch.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    #### Returns:
    - List of feature maps. Its length is `N`. The shapes of the feature maps
        are `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)` where `C_0`, `C_1`, ...,
        `C_(N-1)` are in `in_channels`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],
        blocks: Sequence[Sequence[str]],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        num_branch = len(in_channels)
        self._range_branch = range(num_branch)

        self._branches = nn.ModuleList()
        for c, _blocks in zip(in_channels, blocks):
            branch = nn.Sequential()
            for block in _blocks:
                branch.append(
                    BLOCK[block](
                        c, c,
                        padding_mode=padding_mode,
                        activation=activation,
                        activation_kw=activation_kw
                    )
                )
            self._branches.append(branch)

    def forward(
        self, x: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        for i in self._range_branch:
            x[i] = self._branches[i](x[i])
        return x


class Fusion(nn.Module):
    r'''

    #### Args:
    - in_channels: number of channels. Given the length of it is `N`.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    ### Returns:
    - List of feature maps. Its length is `N`. The shapes of the feature maps
        are `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)` where `C_0`, `C_1`, ..., `C_(N-1)`
        are in `in_channels`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        num_branch = len(in_channels)  # 3

        self._range_branch = range(num_branch)  # Iterator [0, 1, 2]

        self._fus = nn.ModuleList()
        for dst in self._range_branch:  # [0, 1, 2]
            layers = nn.ModuleList()
            for src in self._range_branch:  # [0, 1, 2]
                if src == dst:
                    layers.append(None)  # placeholder
                elif src > dst:
                    layers.append(
                        nn.Sequential(
                            Conv2dBlock(
                                in_channels[src], in_channels[dst], 1, 1,
                                mode='cn'
                            ),
                            nn.Upsample(
                                scale_factor=2 ** (src - dst),
                                mode='nearest'
                            )
                        )
                    )
                else:
                    ds = nn.Sequential()
                    c_in = in_channels[src]
                    for _ in range(src, dst - 1):
                        ds.append(
                            Conv2dBlock(
                                c_in, c_in, 3, 2, 1,
                                padding_mode=padding_mode,
                                activation=activation,
                                activation_kw=activation_kw
                            )
                        )
                    ds.append(
                        Conv2dBlock(
                            c_in, in_channels[dst], 3, 2, 1,
                            padding_mode=padding_mode,
                            activation=activation,
                            activation_kw=activation_kw,
                            mode='cn'
                        )
                    )
                    layers.append(ds)
            self._fus.append(layers)

        self._indices_fusion = [
            [src for src in self._range_branch if src != dst]
            for dst in self._range_branch
        ]  # [[1, 2], [0, 2], [0, 1]]

    def forward(
        self, x: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        outs = []
        for dst in self._range_branch:  # [0, 1, 2]
            out = x[dst]
            _fus = self._fus[dst]
            for src in self._indices_fusion[dst]:
                out = out + _fus[src](x[src])
            outs.append(out)
        return outs


class Stem(nn.Module):
    r'''

    #### Args:
    - in_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    ### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H // 4, W // 4)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._stem = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 3, 2, 1,
                padding_mode=padding_mode,
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
        return self._stem(x)


class SingleBranchStage(nn.Module):
    r'''

    #### Args:
    - in_channels
    - out_channels: the number of output channels.
    - blocks: names of blocks.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - List of a feature map. Its length is `1`. The shape of the feature map is
        `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: Sequence[str],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()

        self._branch = nn.Sequential(
            BLOCK[blocks[0]](
                in_channels, out_channels,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        for block in blocks[1:]:
            self._branch.append(
                BLOCK[block](
                    out_channels, out_channels,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> List[TorchTensor[TorchFloat]]:
        return [self._branch(x)]


class MultiBranchesStage(nn.Module):
    r'''A stage of the HRNet.

    #### Args:
    - in_channels: sequence of input channels. Given the length of it is `N`.
    - out_channels: the number of output channels. Given the length of it is
        `M`.
    - num_module: number of modules. A module is consist of parallel branches
        and a fusion block.
    - blocks: names of blocks used in branches.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    ### Returns:
    - List of feature maps. If `M >= N`, the length of the list is `M`. The
        shapes of the feature maps are `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{M - 1}, H // 2^M, W // 2^M)`
        where `C_0`, `C_1`, ..., `C_(M-1)` are in `out_channels`. If `M < N`,
        the length of the list is `N`. The shapes of the feature maps are
        `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{M - 1}, H // 2^M, W // 2^M)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        num_module: int,
        blocks: Sequence[Sequence[str]],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._trans = Transition(
            in_channels, out_channels, padding_mode, activation, activation_kw
        )

        ms = []
        for _ in range(num_module):
            ms += [
                Branches(
                    out_channels, blocks,
                    padding_mode, activation, activation_kw
                ),
                Fusion(out_channels, padding_mode, activation, activation_kw)
            ]
        self._ms = nn.Sequential(*ms)

    def forward(
        self, x: Sequence[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        return self._ms(self._trans(x))


class HRNet(nn.Module):
    def __init__(
        self,
        stem: Dict[str, Any],
        stages: Sequence[Dict[str, Any]],
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._stem = Stem(**stem)

        self._stages = nn.Sequential(
            SingleBranchStage(**stages[0])
        )
        for stage in stages[1:]:
            self._stages.append(MultiBranchesStage(**stage))

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> List[TorchTensor[TorchFloat]]:
        return self._stages(self._stem(x))
