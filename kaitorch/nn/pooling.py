from typing import Optional, Sequence, Union

from torch import nn

from ..data.utils import tuple_2
from .padding import PAD2D


def avg_pool2d(
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = None,
    padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    padding_mode: Union[str, Sequence[str]] = 'zeros'  # (M_H, M_W)
):
    r'''

    #### Args:
    - kernel_size: the size of the window. Its length should be 2 if it is a
        sequence.
    - stride: the stride of the window. Its length should be 2 if it is a
        sequence.
    - padding: the padding added to dimensions `H` and `W` of the input. Its
        length should be less than or equal to 2 if it is a sequence.
    - ceil_mode: Whether to use ceiling instead of flooring to compute the
        output shape.
    - count_include_pad: Whether to include the zero-padding in the averaging
        calculation.
    - divisor_override: If specified, it will be used as divisor, otherwise
        size of the pooling region will be used.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Returns:
    - 2D average pooling.

    '''
    if isinstance(padding_mode, (tuple, list)):
        len_m = len(padding_mode)
        if 2 == len_m:
            padding = tuple_2(padding)
            if len(padding) > 2:
                raise ValueError(
                    'The length of the sequence `padding` should be less than'
                    ' or equal to 2.'
                )
            m_0 = padding_mode[0]
            m_1 = padding_mode[1]
            if m_0 == m_1:
                padding_mode = m_0
            else:
                if 'zeros' != m_1:
                    layers = [
                        PAD2D[padding_mode[1]]((padding[1], padding[1], 0, 0))
                    ]
                    if 'zeros' != m_0:
                        # m_1 != 'zeros' and m_0 != 'zeros'
                        layers.append(
                            PAD2D[padding_mode[0]](
                                (0, 0, padding[0], padding[0])
                            ),
                            nn.AvgPool2d(
                                kernel_size, stride,
                                ceil_mode=ceil_mode,
                                divisor_override=divisor_override
                            )
                        )
                    else:
                        # m_1 != 'zeros' and m_0 == 'zeros'
                        layers.append(
                            nn.AvgPool2d(
                                kernel_size, stride,
                                padding=(padding[0], 0),
                                ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad,
                                divisor_override=divisor_override
                            )
                        )
                    return nn.Sequential(*layers)
                if 'zeros' != m_0:
                    # m_1 == 'zeros' and m_0 != 'zeros'
                    return nn.Sequential(
                        PAD2D[padding_mode[0]]((0, 0, padding[0], padding[0])),
                        nn.AvgPool2d(
                            kernel_size, stride,
                            padding=(0, padding[1]),
                            ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad,
                            divisor_override=divisor_override
                        )
                    )
        elif 1 == len_m:
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'The length of the sequence `padding_mode` should be less than'
                ' or equal to 2.'
            )
    if 'zeros' != padding_mode:
        return nn.Sequential(
            PAD2D[padding_mode](padding),
            nn.AvgPool2d(
                kernel_size, stride,
                ceil_mode=ceil_mode,
                divisor_override=divisor_override
            )
        )
    return nn.AvgPool2d(
        kernel_size, stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )


def max_pool2d(
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = None,
    padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
    dilation: Union[int, Sequence[int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
    padding_mode: Union[str, Sequence[str]] = 'zeros'  # (M_H, M_W)
):
    r'''

    #### Args:
    - kernel_size: the size of the window. Its length should be 2 if it is a
        sequence.
    - stride: the stride of the window. Its length should be 2 if it is a
        sequence.
    - padding: the padding added to dimensions `H` and `W` of the input. Its
        length should be less than or equal to 2 if it is a sequence.
    - return_indices: Whether to return the max indices along with the outputs.
    - ceil_mode: Whether to use ceiling instead of flooring to compute the
        output shape.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Returns:
    - 2D max pooling.

    '''
    if isinstance(padding_mode, (tuple, list)):
        len_m = len(padding_mode)
        if 2 == len_m:
            padding = tuple_2(padding)
            if len(padding) > 2:
                raise ValueError(
                    'The length of the sequence `padding` should be less than'
                    ' or equal to 2.'
                )
            m_0 = padding_mode[0]
            m_1 = padding_mode[1]
            if m_0 == m_1:
                padding_mode = m_0
            else:
                if 'zeros' != m_1:
                    layers = [
                        PAD2D[padding_mode[1]]((padding[1], padding[1], 0, 0))
                    ]
                    if 'zeros' != m_0:
                        # m_1 != 'zeros' and m_0 != 'zeros'
                        layers += [
                            PAD2D[padding_mode[0]](
                                (0, 0, padding[0], padding[0])
                            ),
                            nn.MaxPool2d(
                                kernel_size, stride,
                                dilation=dilation,
                                return_indices=return_indices,
                                ceil_mode=ceil_mode
                            )
                        ]
                    else:
                        # m_1 != 'zeros' and m_0 == 'zeros'
                        layers.append(
                            nn.MaxPool2d(
                                kernel_size, stride,
                                padding=(padding[0], 0),
                                dilation=dilation,
                                return_indices=return_indices,
                                ceil_mode=ceil_mode
                            )
                        )
                    return nn.Sequential(*layers)
                if 'zeros' != m_0:
                    # m_1 == 'zeros' and m_0 != 'zeros'
                    return nn.Sequential(
                        PAD2D[padding_mode[0]]((0, 0, padding[0], padding[0])),
                        nn.MaxPool2d(
                            kernel_size, stride,
                            padding=(0, padding[1]),
                            dilation=dilation,
                            return_indices=return_indices,
                            ceil_mode=ceil_mode
                        )
                    )
        elif 1 == len_m:
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'The length of the sequence `padding_mode` should be less than'
                ' or equal to 2.'
            )
    if 'zeros' != padding_mode:
        return nn.Sequential(
            PAD2D[padding_mode](padding),
            nn.MaxPool2d(
                kernel_size, stride,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode
            )
        )
    return nn.MaxPool2d(
        kernel_size, stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode
    )
