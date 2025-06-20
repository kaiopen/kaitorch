from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchBool, TorchReal
from .attribute import Attribute


class Ignored(Attribute):
    r'''Whether the data is ignored.

    #### Args:
    - ignored: ignoring attributes. Its shape should be `(N >= 0, 1)` or
        `(N >= 0,)`.

    #### Properties:
    - device
    - ignored: ignoring attributes. Its shape is `(N >= 0, 1)`.
    - ignored_: ignoring attributes. Its shape is `(N >= 0, 1)`.

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
    - update_ignored_: Update the ignoring attributes.

    #### Static Methods:
    - format_ignored: Make sure the shape of `ignored` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, ignored: TorchTensor[TorchBool], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ignored_ = self.format_ignored(ignored)

        if self._device is None:
            self._device = self.ignored_.device
        elif self.ignored_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.ignored_)
        elif len(self.ignored_) != self._len:
            self.__error_num(self._len)

    @property
    def ignored(self) -> TorchTensor[TorchBool]:
        r'''
        Ignoring attributes. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.ignored_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`ignored` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(ignored=self.ignored_[i])

    def append_(
        self,
        ignored: TorchTensor[TorchBool],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - ignored: ignoring attributes. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if ignored.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        ignored = self.format_ignored(ignored)

        if -1 == _len:
            _len = len(ignored)
            self._len += _len
        elif len(ignored) != _len:
            self.__error_num(_len)

        self.ignored_ = torch.cat((self.ignored_, ignored))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(ignored=self.ignored)

    def cpu_(self):
        super().cpu_()
        self.ignored_ = self.ignored_.cpu()
        self._device = self.ignored_.device

    def cuda_(self):
        super().cuda_()
        self.ignored_ = self.ignored_.cuda()
        self._device = self.ignored_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.ignored_ = self.ignored_[i]
        self._len = len(self.ignored_)

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
    def format_ignored(
        ignored: TorchTensor[TorchBool]
    ) -> TorchTensor[TorchBool]:
        r'''Make sure the shape of `ignored` is `(N >= 0, 1)`.

        #### Args:
        - ignored: ignoring attributes. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Ignoring attributes. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == ignored.ndim:
            return ignored.reshape(-1, 1)

        if 2 != ignored.ndim or 1 != ignored.shape[1]:
            raise ValueError(
                '`ignored` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return ignored

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(ignored=obj.ignored_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.ignored_ = torch.cat((self.ignored_, obj.ignored_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def update_ignored_(self, ignored: TorchTensor[TorchBool]) -> None:
        r'''Update the ignoring attributes.

        Warning: This is an inplace method.

        #### Args:
        - ignored: ignoring attributes. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        '''
        self.ignored_ = self.format_ignored(ignored)

        if self.ignored_.device != self._device:
            self._error_device()

        if len(self.ignored_) != self._len:
            self.__error_num(self._len)
