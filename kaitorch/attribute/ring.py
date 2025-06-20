from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class Ring(Attribute):
    r'''

    #### Args:
    - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

    #### Properties:
    - device
    - ring: rings. Its shape is `(N >= 0, 1)`.
    - ring_: rings. Its shape is `(N >= 0, 1)`.

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
    - slice_all: Slice all of the data.
    - update_ring_: Update the rings.

    #### Static Methods:
    - format_ring: Make sure the shape of `ring` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, ring: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ring_ = self.format_ring(ring)

        if self._device is None:
            self._device = self.ring_.device
        elif self.ring_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.ring_)
        elif len(self.ring_) != self._len:
            self.__error_num(self._len)

    @property
    def ring(self) -> TorchTensor[TorchReal]:
        r'''
        Rings. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.ring_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`ring` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(ring=self.ring_[i])

    def append_(
        self,
        ring: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if ring.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        ring = self.format_ring(ring)

        if -1 == _len:
            _len = len(ring)
            self._len += _len
        elif len(ring) != _len:
            self.__error_num(_len)

        self.ring_ = torch.cat((self.ring_, ring))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(ring=self.ring)

    def cpu_(self):
        super().cpu_()
        self.ring_ = self.ring_.cpu()
        self._device = self.ring_.device

    def cuda_(self):
        super().cuda_()
        self.ring_ = self.ring_.cuda()
        self._device = self.ring_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.ring_ = self.ring_[i]
        self._len = len(self.ring_)

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
    def format_ring(
        ring: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `ring` is `(N >= 0, 1)`.

        #### Args:
        - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        #### Returns:
        - Rings. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == ring.ndim:
            return ring.reshape(-1, 1)

        if 2 != ring.ndim or 1 != ring.shape[1]:
            raise ValueError(
                '`ring` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return ring

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(ring=obj.ring_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.ring_ = torch.cat((self.ring_, obj.ring_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def update_ring_(self, ring: TorchTensor[TorchReal]) -> None:
        r'''Update the rings.

        Warning: This is an inplace method.

        #### Args:
        - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        '''
        self.ring_ = self.format_ring(ring)

        if self.ring_.device != self._device:
            self._error_device()

        if len(self.ring_) != self._len:
            self.__error_num(self._len)
