from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from ..data import add_radii, minus_radii, PI
from .attribute import Attribute


class Yaw(Attribute):
    r'''

    #### Args:
    - yaw: yaws in radius. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

    #### Properties:
    - device
    - yaw: yaws in radius. Its shape is `(N >= 0, 1)`.
    - yaw_: yaws in radius. Its shape is `(N >= 0, 1)`.

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
    - update_yaw_: Update the yaws.

    #### Static Methods:
    - format_yaw: Make sure the shape of `yaw` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, yaw: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.yaw_ = self.format_yaw(yaw)

        if self._device is None:
            self._device = self.yaw_.device
        elif self.yaw_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.yaw_)
        elif len(self.yaw_) != self._len:
            self.__error_num(self._len)

    @property
    def yaw(self) -> TorchTensor[TorchReal]:
        r'''
        Yaws in radius. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.yaw_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`yaw` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(yaw=self.yaw_[i])

    def append_(
        self,
        yaw: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - yaw: yaws in radius. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if yaw.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        yaw = self.format_yaw(yaw)

        if -1 == _len:
            _len = len(yaw)
            self._len += _len
        elif len(yaw) != _len:
            self.__error_num(_len)

        self.yaw_ = torch.cat((self.yaw_, yaw))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(yaw=self.yaw)

    def cpu_(self):
        super().cpu_()
        self.yaw_ = self.yaw_.cpu()
        self._device = self.yaw_.device

    def cuda_(self):
        super().cuda_()
        self.yaw_ = self.yaw_.cuda()
        self._device = self.yaw_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.yaw_ = self.yaw_[i]
        self._len = len(self.yaw_)

    def flip_around_x_axis_(self) -> None:
        r'''Flip the data around the X axis.

        Warning: This is an inplace method.

        '''
        self.yaw_ = -self.yaw_

    def flip_around_y_axis_(self) -> None:
        r'''Flip the data around the Y axis.

        Warning: This is an inplace method.

        '''
        self.yaw_ = minus_radii(PI, self.yaw_)

    @staticmethod
    def format_yaw(
        yaw: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `yaw` is `(N >= 0, 1)`.

        #### Args:
        - yaw: yaws in radius. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Yaws in radius. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == yaw.ndim:
            return yaw.reshape(-1, 1)

        if 2 != yaw.ndim or 1 != yaw.shape[1]:
            raise ValueError(
                '`yaw` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return yaw

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(yaw=obj.yaw_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.yaw_ = torch.cat((self.yaw_, obj.yaw_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        self.yaw_ = add_radii(self.yaw_, radius)

    def update_yaw_(self, yaw: TorchTensor[TorchReal]) -> None:
        r'''Update the yaws.

        Warning: This is an inplace method.

        #### Args:
        - yaw: yaws in radius. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        '''
        self.yaw_ = self.format_yaw(yaw)

        if self.yaw_.device != self._device:
            self._error_device()

        if len(self.yaw_) != self._len:
            self.__error_num(self._len)
