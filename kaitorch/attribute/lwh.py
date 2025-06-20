from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class LWH(Attribute):
    r'''

    #### Args:
    - lwh: sizes of cuboids. Its shape should be `(N >= 0, 3)`, `(3,)` or
        `(0,)`. The size of a cuboid should be in the form of
        `(length, width, height)`.

    #### Properties:
    - device
    - lwh: sizes of the cuboids. Its shape is `(N >= 0, 3)`. The size of a
        cuboid is in the form of `(length, width, height)`.
    - lwh_: sizes of the cuboids. Its shape is `(N >= 0, 3)`. The size of a
        cuboid is in the form of `(length, width, height)`.

    #### Methods:
    - __iter__
    - __getitem__: Slice the necessary data.
    - __len__
    - __next__
    - append_: Append new data to the existed data.
    - copy: Copy the necessary data.
    - copy_all: Copy all of the data.
    - cpu_
    - cuda_
    - filter_: Filter the necessary data.
    - flip_around_x_axis_: Flip the data around the X axis.
    - flip_around_y_axis_: Flip the data around the Y axis.
    - is_empty: Whether there is no data.
    - merge_: Merge the two.
    - rotate_around_z_axis_: Rotate the data around the Z axis.
    - scale_: Scale the data.
    - slice_all: Slice all of the data.
    - update_lwh_: Update the sizes.

    #### Static Methods:
    - format_lwh: Make sure the shape of `lwh` is `(N >= 0, 3)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(self, lwh: TorchTensor[TorchReal], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lwh_ = self.format_lwh(lwh)

        if self._device is None:
            self._device = self.lwh_.device
        elif self.lwh_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.lwh_)
        elif len(self.lwh_) != self._len:
            self.__error_num(self._len)

    @property
    def lwh(self) -> TorchTensor[TorchReal]:
        r'''
        Sizes of the cuboids. Its shape is `(N >= 0, 3)`. The size of a cuboid
        is in the form of `(length, width, height)`.

        This is a copy of the data stored.

        '''
        return self.lwh_.clone()

    @staticmethod
    def __error_num(num: int):
        if 0 == num:
            raise ValueError('`lwh` with shape `(0,)` wanted.')
        elif 1 == num:
            raise ValueError(
                '`lwh` with shape `(1, 3)` or `(3,)` wanted.'
            )
        raise ValueError(f'`lwh` with shape `({num}, 3)` wanted.')

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(lwh=self.lwh_[i])

    def append_(self, lwh: TorchTensor[TorchReal], *args, **kwargs) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - lwh: sizes of cuboids. Its shape should be `(N >= 0, 3)`, `(3,)` or
            `(0,)`. The size of a cuboid should be in the form of
            `(length, width, height)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if lwh.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        lwh = self.format_lwh(lwh)

        if -1 == _len:
            _len = len(lwh)
            self._len += _len
        elif len(lwh) != _len:
            self.__error_num(_len)

        self.lwh_ = torch.cat((self.lwh_, lwh))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(lwh=self.lwh)

    def cpu_(self):
        super().cpu_()
        self.lwh_ = self.lwh_.cpu()
        self._device = self.lwh_.device

    def cuda_(self):
        super().cuda_()
        self.lwh_ = self.lwh_.cuda()
        self._device = self.lwh_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.lwh_ = self.lwh_[i]
        self._len = len(self.lwh_)

    def flip_around_x_axis_(self) -> None:
        r'''Flip the data around the X axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_x_axis_()

    def flip_around_y_axis_(self) -> None:
        r'''Flip the data around the Y axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_y_axis_()

    @staticmethod
    def format_lwh(lwh: TorchTensor[TorchReal]) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `lwh` is `(N >= 0, 3)`.

        #### Args:
        - lwh: sizes of cuboids. Its shape should be `(N >= 0, 3)`, `(3,)` or
            `(0,)`. The size of a cuboid should be in the form of
            `(length, width, height)`.

        #### Returns:
        - Sizes of cuboids. Its shape is `(N >= 0, 3)`. The size of a cuboid is
            in the form of `(length, width, height)`.

        '''
        if 1 == lwh.ndim and (0 == (len_lwh := len(lwh)) or 3 == len_lwh):
            return lwh.reshape(-1, 3)

        if 2 != lwh.ndim or 3 != lwh.shape[1]:
            raise ValueError(
                '`lwh` with shape `(N >= 0, 3)`, `(3,)` or `(0,)` wanted.'
            )
        return lwh

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(lwh=obj.lwh_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.lwh_ = torch.cat((self.lwh_, obj.lwh_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def scale_(self, scale: TorchReal) -> None:
        r'''Scale the data.

        Warning: This is an inplace method.

        #### Args:
        - scale

        '''
        self.lwh_ *= scale

    def update_lwh_(self, lwh: TorchTensor[TorchReal]) -> None:
        r'''Update the sizes.

        Warning: This is an inplace method.

        #### Args:
        - lwh: sizes of cuboids. Its shape should be `(N >= 0, 3)`, `(3,)` or
            `(0,)`. The size of a cuboid should be in the form of
            `(length, width, height)`.

        '''
        self.lwh_ = self.format_lwh(lwh)

        if self.lwh_.device != self._device:
            self._error_device()

        if len(self.lwh_) != self._len:
            self.__error_num(self._len)
