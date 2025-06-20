r'''
All Tokens Matter: Token Labeling for Training Better Vision Transformers

'''

from typing import Optional, Tuple, Union

import torch
from torchvision.ops import roi_align
import torchvision.transforms.v2.functional as tvF

from ..typing import TorchTensor, TorchFloat, TorchReal, real
from ..data.utils import tuple_2
from .cutmix import CutMix, cutmix_, reverse_


def get_box(
    size_map: Tuple[int, int],
    size_img: Tuple[int, int],
    box: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[real, real, real, real]:
    r'''Convert a box onto the top-K map.

    ### Args:
    - size_map: the size of the top-K map. It should be in the form of
        `(H, W)`.
    - size_img: the size of the image that not be cropped and resized. It
        should be in the form of `(H, W)`.
    - box: the cropping box. It should be in the form of `(y, x, h, w)`.

    ### Returns:
    - A box in the form of `(x_min, y_min, x_max, y_max)`.

    '''
    h_map, w_map = size_map

    if box is None:
        return (0, 0, h_map, w_map)

    i, j, h, w = box
    h_img, w_img = size_img
    scale_y = h_map / h_img
    scale_x = w_map / w_img
    i *= scale_y
    j *= scale_x
    h *= scale_y
    w *= scale_x
    h += i
    w += j
    return (i, j, h, w)


def get_label(
    map: TorchTensor[TorchFloat],
    box: Tuple[real, real, real, real],
    size: Union[int, Tuple[int, int]]
) -> TorchTensor[TorchFloat]:
    r'''

    ### Args:
    - map: a feature map. Its shape should be `(C, H, W)`.
    - box: a region of interest. It should be in the form of
        `(x_min, y_min, x_max, y_max)`.
    - size: the output size.

    ### Returns:
    - A label. Its shape is `(C, *size)` if the `size` is a tuple, or
        `(C, size, size)`. Its values are in [0, 1).

    '''
    return torch.nn.functional.softmax(
        roi_align(
            map.unsqueeze(0),
            boxes=[torch.as_tensor(box, dtype=map.dtype).unsqueeze_(0),],
            output_size=size,
            aligned=True
        ).squeeze_(0),
        dim=0
    )


def get_map(
    topk: Tuple[TorchTensor[TorchReal], TorchTensor[TorchReal]],
    num_category: int
) -> TorchTensor[TorchFloat]:
    r'''

    ### Args:
    - topk: a Top-K result of a feature map. The first item is values. The
        second item is indices. The shape of a item should be `(K, H, W)`.
    - num_category

    ### Returns:
    - A feature map. Its shape is `(num_category, H, W)`.

    '''
    v, inds = topk
    return torch.full(
        (num_category, *v.shape[1:]), 0, dtype=torch.float
    ).scatter_(0, inds.long(), v.float())


class TokenLabeling:
    r'''

    ### Args:
    - num_category: the number of categories.
    - size: the output size of top-K maps.

    ### Methods:
    - __call__
    - get_label
    - get_map

    ## __call__
    ### Args:
    - topk: a top-K map. Its shape should be `(2, K, H, W)`.
    - size_img: the size of the image that not be cropped and resized. It
        should be in the form of `(H, W)`.
    - box: the cropping box. It should be in the form of `(y, x, h, w)`.

    ### Returns:
    - A label. Its shape is `(C, *size)`. Its values are in [0, 1).
    - A label. Its shape is `(C,)`. Its values are in [0, 1).

    '''
    def __init__(
        self,
        num_category: int,
        size: Union[Tuple[int, int], int] = (7, 7)
    ) -> None:
        self._num_cat = num_category
        self._size = tuple_2(size)

    def __call__(
        self,
        topk: TorchTensor[TorchReal],
        size_img: Tuple[int, int],
        box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        return self.get_label(
            self.get_map(topk), get_box(tvF.get_size(topk), size_img, box)
        )

    def get_label(
        self,
        map: TorchTensor[TorchReal],
        box: Tuple[real, real, real, real]
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        r'''

        ### Args:
        - map: a feature map. Its shape should be `(C, H, W)`.
        - box: a region of interest. It should be in the form of
            `(x_min, y_min, x_max, y_max)`.

        ### Returns:
        - A label. Its shape is `(C, *size)`. Its values are in [0, 1).
        - A label. Its shape is `(C,)`. Its values are in [0, 1).

        '''
        return get_label(map, box, self._size), \
            get_label(map, box, (1, 1)).squeeze_()

    def get_map(
        self, topk: Tuple[TorchTensor[TorchReal], TorchTensor[TorchReal]]
    ) -> TorchTensor[TorchFloat]:
        r'''

        ### Args:
        - topk: a Top-K result of a feature map. The first item is values. The
            second item is indices. The shape of a item should be `(K, H, W)`.

        ### Returns:
        - A feature map. Its shape is `(num_category, H, W)`.

        '''
        return get_map(topk, self._num_cat)


class MixToken(CutMix):
    def __init__(
        self, alpha: float = 1., num_category: int = 1000, scale: int = 8
    ) -> None:
        super().__init__(alpha, num_category)
        self._scale = scale

    def __call__(
        self, x: TorchTensor[TorchReal], label: TorchTensor[TorchReal]
    ) -> Tuple[TorchTensor[TorchReal], TorchTensor[TorchReal]]:
        h, w = x.shape[2:]
        s = self._scale
        box, lam = self.random_box(h // s, w // s)

        x_0, y_0, x_1, y_1 = box
        return cutmix_(x, (x_0 * s, y_0 * s, x_1 * s, y_1 * s)), \
            self.mix_label(label, lam)
