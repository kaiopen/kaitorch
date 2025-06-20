from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.fft import fft2, ifft2
import torch.nn.functional as F

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import Conv2dBlock, conv2d


class FrequencyPerception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        c = hidden_channels or in_channels // 16
        self._weight = nn.Sequential(
            Conv2dBlock(
                in_channels, c, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                c, in_channels, 1,
                bias=True,
                activation='sigmoid',
                activation_kw={},
                mode='ca'
            )
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = fft2(x.float())  # amp
        return torch.abs(ifft2(self._weight(x.real) * x))


class JDPM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._proj = Conv2dBlock(
            in_channels, hidden_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )

        self._aconv_3 = nn.Sequential(
            Conv2dBlock(hidden_channels, hidden_channels, 1, mode='cn'),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3,
                padding=3,
                dilation=3,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._aconv_5 = nn.Sequential(
            Conv2dBlock(hidden_channels, hidden_channels, 1, mode='cn'),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3,
                padding=5,
                dilation=5,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._aconv_7 = nn.Sequential(
            Conv2dBlock(hidden_channels, hidden_channels, 1, mode='cn'),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3,
                padding=7,
                dilation=7,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._aconv_9 = nn.Sequential(
            Conv2dBlock(hidden_channels, hidden_channels, 1, mode='cn'),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3,
                padding=9,
                dilation=9,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

        self._freq = nn.Sequential(
            FrequencyPerception(
                hidden_channels,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels,
                activation=activation,
                activation_kw=activation_kw,
                mode='na'
            )
        )

        self._reduce = Conv2dBlock(
            hidden_channels * 5, hidden_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )

        c = hidden_channels // 2
        self._out = nn.Sequential(
            Conv2dBlock(
                hidden_channels, c, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            nn.Conv2d(c, 1, 1)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = self._proj(x)

        x_3 = self._aconv_3(x)
        # TODO: inplace when eval
        x_3 = x_3 + self._freq(x_3)

        x_5 = self._aconv_5(x + x_3)
        x_5 = x_5 + self._freq(x_5)

        x_7 = self._aconv_7(x + x_5)
        x_7 = x_7 + self._freq(x_7)

        x_9 = self._aconv_9(x + x_7)
        x_9 = x_9 + self._freq(x_9)

        return self._out(
            self._reduce(torch.cat((x_3, x_5, x_7, x_9, x), 1)) + x
        )


class FSA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_head: int = 8,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._num_head = num_head

        self._temp = nn.Parameter(torch.ones(num_head, 1, 1))

        c = in_channels // 16
        self._weight = nn.Sequential(
            Conv2dBlock(
                in_channels, c, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                c, in_channels, 1,
                bias=True,
                activation='sigmoid',
                activation_kw={},
                mode='ca'
            )
        )

        self._proj = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    @staticmethod
    def _complex_norm(x, dim: int = -1):
        return torch.complex(F.softmax(x.real, dim), F.softmax(x.imag, dim))

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        batch_size, _, h, w = x.shape
        x = fft2(x.float())

        v = x.reshape(batch_size, self._num_head, -1, h * w)
        qk = F.normalize(v, dim=-1)  # (B, num_head, C, HW)
        return self._proj(
            torch.cat(
                (
                    torch.abs(
                        ifft2(
                            torch.matmul(
                                self._complex_norm(
                                    torch.matmul(qk, qk.transpose(2, 3))
                                    * self._temp
                                ),  # (B, num_head, C, C)
                                v  # (B, num_head, C, HW)
                            )  # (B, num_head, C, HW)
                        )
                    ).reshape(batch_size, -1, h, w),
                    torch.abs(ifft2(self._weight(x.real) * x))
                ),
                dim=1
            )
        )


class QKV(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        c = in_channels * 3
        self._pos = nn.Conv2d(in_channels, c, 1)  # position

        o = in_channels // 2 * 3
        self._dconv_3 = conv2d(
            c, o, 3, 1, 1,
            groups=o,
            padding_mode=padding_mode
        )
        self._dconv_5 = conv2d(
            c, o, 5, 1, 2,
            groups=o,
            padding_mode=padding_mode
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = self._pos(x)
        return torch.cat((self._dconv_3(x), self._dconv_5(x)), 1)


class SSA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_head: int = 8,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._num_head = num_head
        self._qkv = QKV(in_channels, padding_mode)

        self._temp = nn.Parameter(torch.ones(num_head, 1, 1))

        c = in_channels // 2
        self._dconv_3 = conv2d(
            in_channels, c, 3, 1, 1,
            groups=c,
            padding_mode=padding_mode
        )
        self._dconv_5 = conv2d(
            in_channels, c, 5, 1, 2,
            groups=c,
            padding_mode=padding_mode
        )
        self._proj = nn.Conv2d(c * 4, in_channels, 1, bias=False)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        batch_size, _, h, w = x.shape
        qkv = self._qkv(x).reshape(batch_size, 3, self._num_head, -1, h * w)
        qk = F.normalize(qkv[:, :2], dim=-1)
        # v = qkv[:, 2]
        return self._proj(
            torch.cat(
                (
                    torch.matmul(
                        F.softmax(
                            torch.matmul(
                                qk[:, 0], qk[:, 1].transpose(2, 3)
                            ) * self._temp,
                            dim=-1
                        ),  # (B, num_head, C, C)
                        qkv[:, 2]  # (B, num_head, C, HW)
                    ).reshape(batch_size, -1, h, w),
                    self._dconv_3(x),
                    self._dconv_5(x)
                ),
                dim=1
            )
        )


class EFFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._freq_0 = FrequencyPerception(
            in_channels,
            activation=activation,
            activation_kw=activation_kw
        )
        self._dconv_0 = conv2d(
            in_channels, in_channels, 3, 1, 1,
            groups=in_channels,
            padding_mode=padding_mode
        )

        c = in_channels * 2
        self._freq_1 = FrequencyPerception(
            c, in_channels // 16,
            activation=activation,
            activation_kw=activation_kw
        )
        self._dconv_1 = conv2d(
            c, c, 3, 1, 1,
            groups=in_channels,
            padding_mode=padding_mode
        )

        self._proj = nn.Conv2d(in_channels * 4, in_channels, 1, bias=False)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        f = self._freq_0(x)
        s = self._dconv_0(x)

        # TODO: inplace when eval
        x = torch.cat((F.gelu(f) * f, F.gelu(s) * s), dim=1)
        return self._proj(
            torch.cat((self._freq_1(x), self._dconv_1(x)), dim=1)
        )


class Former(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_head: int = 8,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._norm_0 = nn.LayerNorm(in_channels)
        self._ssa = SSA(in_channels, num_head, padding_mode)
        self._fsa = FSA(in_channels, num_head, activation, activation_kw)
        self._norm_1 = nn.LayerNorm(in_channels)
        self._ffn = EFFN(in_channels, padding_mode, activation, activation_kw)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x_0 = self._norm_0(
            x.flatten(2).transpose(1, 2)
        ).transpose(1, 2).view(x.shape).contiguous()

        x = x + self._ssa(x_0) + self._fsa(x_0)  # TODO: inplace when eval
        return x + self._ffn(
            self._norm_1(
                x.flatten(2).transpose(1, 2)
            ).transpose(1, 2).view(x.shape).contiguous()
        )


class ETB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_head: int = 8,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._proj = Conv2dBlock(
            in_channels, out_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )
        self._former = Former(
            out_channels, num_head, padding_mode, activation, activation_kw
        )
        self._reduce = Conv2dBlock(
            out_channels * 2, out_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = self._proj(x)
        # TODO: inplace when eval
        return self._reduce(torch.cat((x, self._former(x)), dim=1)) + x


class DRP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_aux: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._conv = nn.Sequential(
            Conv2dBlock(
                in_channels * (num_aux + 1), in_channels, 1,
                mode='cn'
            ),
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                activation=activation,
                activation_kw=activation_kw,
                padding_mode=padding_mode,
                mode='cncna'
            ),
        )

        self._spat = nn.Sequential(
            Conv2dBlock(
                in_channels, in_channels, 1,
                mode='cn'
            ),
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                mode='cncna'
            )
        )
        self._freq = nn.Sequential(
            FrequencyPerception(
                in_channels,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                in_channels,
                activation=activation,
                activation_kw=activation_kw,
                mode='na'
            )
        )

        self._out = nn.Sequential(
            Conv2dBlock(
                in_channels * 2, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            nn.Conv2d(in_channels, 1, 1)
        )

    @staticmethod
    def _get_reverse_attention(
        x: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return 2 - torch.sigmoid(x) \
            - torch.sigmoid(torch.abs(fft2(x.float())))

    def forward(
        self,
        x: TorchTensor[TorchFloat],
        auxs: Sequence[TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        size = x.shape[2:]
        ups = []
        exs = [x,]
        ras = []
        for aux in auxs:
            aux = F.interpolate(
                aux, size=size, mode='bilinear', align_corners=True
            )
            ups.append(aux)  # (B, 1, H, W)
            exs.append(aux.expand_as(x))  # (B, C, H, W)
            ras.append(self._get_reverse_attention(aux))  # (B, 1, H, W)

        y = self._conv(torch.cat(exs, dim=1))
        return self._out(
            torch.cat(
                (
                    sum(ras).expand_as(x) * x,
                    self._spat(y) + self._freq(y)
                ),
                dim=1
            )
        ) + sum(ups)


class FSEL(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        hidden_channels: int = 128,
        num_head: int = 8,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ):
        super().__init__()
        self._c = hidden_channels
        self._range = list(range(1, len(in_channels)))

        self._jdpm = JDPM(
            in_channels[-1], hidden_channels,
            padding_mode, activation, activation_kw
        )

        self._up = nn.Sequential(
            nn.PixelShuffle(2),
            Conv2dBlock(hidden_channels // 4, hidden_channels, 1, mode='cn'),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

        self._etbs = nn.ModuleList()
        for c in in_channels[::-1]:
            self._etbs.append(
                ETB(
                    c + hidden_channels, hidden_channels,
                    num_head, padding_mode, activation, activation_kw
                )
            )
        self._drps = nn.ModuleList()
        self._drps.append(nn.Identity())
        for i in self._range:
            self._drps.append(
                DRP(
                    hidden_channels,
                    num_aux=i,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )

    def forward(
        self, x: List[TorchTensor[TorchFloat]]
    ) -> Tuple[TorchTensor[TorchFloat], List[TorchTensor[TorchFloat]]]:
        x = list(x)
        x.reverse()

        r = self._range
        etbs = self._etbs
        drps = self._drps

        p = x[0]
        e = self._jdpm(p)

        p = etbs[0](torch.cat((p, e.expand(-1, self._c, -1, -1)), dim=1))
        x[0] = p

        for i in r:
            p = etbs[i](torch.cat((x[i], self._up(p)), dim=1))
            x[i] = p

        x[0] = drps[1](x[0], [e,])
        for i in r:
            x[i] = drps[i](x[i], x[0: i])

        x.reverse()
        x.append(e)
        return x[0], x[1:]
