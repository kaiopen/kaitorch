from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class Timestamp(Attribute):
    r'''

    #### Args:
    - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

    #### Properties:
    - device
    - timestamp: timestamps. Its shape is `(N >= 0, 1)`.
    - timestamp_: timestamps. Its shape is `(N >= 0, 1)`.

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
    - synchronize_with_: Synchronize with the input.
    - update_timestamp_: Update the timestamps.

    #### Static Methods:
    - format_timestamp: Make sure the shape of `timestamp` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, timestamp: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.timestamp_ = self.format_timestamp(timestamp)

        if self._device is None:
            self._device = self.timestamp_.device
        elif self.timestamp_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.timestamp_)
        elif len(self.timestamp_) != self._len:
            self.__error_num(self._len)

    @property
    def timestamp(self) -> TorchTensor[TorchReal]:
        r'''
        Timestamps. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.timestamp_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`timestamp` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(timestamp=self.timestamp_[i])

    def append_(
        self,
        timestamp: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if timestamp.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        timestamp = self.format_timestamp(timestamp)

        if -1 == _len:
            _len = len(timestamp)
            self._len += _len
        elif len(timestamp) != _len:
            self.__error_num(_len)

        self.timestamp_ = torch.cat((self.timestamp_, timestamp))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(timestamp=self.timestamp)

    def cpu_(self):
        super().cpu_()
        self.timestamp_ = self.timestamp_.cpu()
        self._device = self.timestamp_.device

    def cuda_(self):
        super().cuda_()
        self.timestamp_ = self.timestamp_.cuda()
        self._device = self.timestamp_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.timestamp_ = self.timestamp_[i]
        self._len = len(self.timestamp_)

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
    def format_timestamp(
        timestamp: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `timestamp` is `(N >= 0, 1)`.

        #### Args:
        - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Timestamps. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == timestamp.ndim:
            return timestamp.reshape(-1, 1)

        if 2 != timestamp.ndim or 1 != timestamp.shape[1]:
            raise ValueError(
                '`timestamp` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return timestamp

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(timestamp=obj.timestamp_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.timestamp_ = torch.cat((self.timestamp_, obj.timestamp_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def synchronize_with_(self, obj) -> None:
        r'''Synchronize with the input.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        self.timestamp_[:] = obj.timestamp_[0]

    def update_timestamp_(self, timestamp: TorchTensor[TorchReal]) -> None:
        r'''Update the timestamps.

        Warning: This is an inplace method.

        #### Args:
        - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        '''
        self.timestamp_ = self.format_timestamp(timestamp)

        if self.timestamp_.device != self._device:
            self._error_device()

        if len(self.timestamp_) != self._len:
            self.__error_num(self._len)
