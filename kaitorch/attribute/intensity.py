from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class Intensity(Attribute):
    def __init__(
        self, intensity: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - intensity: intensities. Its shape should be `(N >= 0, 1)` or
                `(N >= 0,)`.

        ### Properties:
            - device
            - intensity: intensities. Its shape is `(N >= 0, 1)`.
            - intensity_: intensities. Its shape is `(N >= 0, 1)`.

        ### Methods:
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
            - update_intensity_: Update the intensities.

        ### Static Methods:
            - format_intensity: Make sure the shape of `intensity` is
                `(N >= 0, 1)`.

        ### Class Methods:
            - from_similar: New data from the input.

        '''
        super().__init__(*args, **kwargs)
        self.intensity_ = self.format_intensity(intensity)

        if self._device is None:
            self._device = self.intensity_.device
        elif self.intensity_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.intensity_)
        elif len(self.intensity_) != self._len:
            self.__error_num(self._len)

    @property
    def intensity(self) -> TorchTensor[TorchReal]:
        r'''
        Intensities. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.intensity_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`intensity` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        ### Args:
            - i: index, slice, mask or indices.

        ### Returns:
            - A view of self.

        '''
        return self.__class__(intensity=self.intensity_[i])

    def append_(
        self,
        intensity: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        ### Args:
            - intensity: intensities. Its shape should be `(N >= 0, 1)` or
                `(N >= 0,)`.

        ### Returns:
            - Number of the appended boxes.

        '''
        if intensity.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        intensity = self.format_intensity(intensity)

        if -1 == _len:
            _len = len(intensity)
            self._len += _len
        elif len(intensity) != _len:
            self.__error_num(_len)

        self.intensity_ = torch.cat((self.intensity_, intensity))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        ### Returns:
            - A copy of self.

        '''
        return self.__class__(intensity=self.intensity)

    def cpu_(self):
        super().cpu_()
        self.intensity_ = self.intensity_.cpu()
        self._device = self.intensity_.device

    def cuda_(self):
        super().cuda_()
        self.intensity_ = self.intensity_.cuda()
        self._device = self.intensity_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        ### Args:
            - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.intensity_ = self.intensity_[i]
        self._len = len(self.intensity_)

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
    def format_intensity(
        intensity: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `intensity` is `(N >= 0, 1)`.

        ### Args:
            - intensity: intensities. Its shape should be `(N >= 0, 1)` or
                `(N >= 0,)`.

        ### Returns:
            - Intensities. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == intensity.ndim:
            return intensity.reshape(-1, 1)

        if 2 != intensity.ndim or 1 != intensity.shape[1]:
            raise ValueError(
                '`intensity` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return intensity

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        ### Args:
            - obj

        ### Returns:
            - Data sharing the storage memory with the input.

        '''
        return cls(intensity=obj.intensity_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        ###Args:
            - obj

        '''
        super().merge_(obj)
        self.intensity_ = torch.cat((self.intensity_, obj.intensity_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        ### Args:
            - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def update_intensity_(self, intensity: TorchTensor[TorchReal]) -> None:
        r'''Update the intensities.

        Warning: This is an inplace method.

        ### Args:
            - intensity: intensities. Its shape should be `(N >= 0, 1)` or
                `(N >= 0,)`.

        '''
        self.intensity_ = self.format_intensity(intensity)

        if self.intensity_.device != self._device:
            self._error_device()

        if len(self.intensity_) != self._len:
            self.__error_num(self._len)
