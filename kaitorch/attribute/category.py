from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class Category(Attribute):
    r'''

    #### Args:
    - category: categories. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

    #### Properties:
    - category: categories. Its shape is `(N >= 0, 1)`.
    - category_: categories. Its shape is `(N >= 0, 1)`.
    - device

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
    - update_category_: Update the categories.

    #### Static Methods:
    - format_category: Make sure the shape of `category` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, category: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.category_ = self.format_category(category)

        if self._device is None:
            self._device = self.category_.device
        elif self.category_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.category_)
        elif len(self.category_) != self._len:
            self.__error_num(self._len)

    @property
    def category(self) -> TorchTensor[TorchReal]:
        r'''
        Categories. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.category_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(
            f'`category` with shape `({num}, 1)` or `({num},)` wanted.'
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(category=self.category_[i])

    def append_(
        self,
        category: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - category: categories. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if category.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        category = self.format_category(category)

        if -1 == _len:
            _len = len(category)
            self._len += _len
        elif len(category) != _len:
            self.__error_num(_len)

        self.category_ = torch.cat((self.category_, category))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(category=self.category)

    def cpu_(self):
        super().cpu_()
        self.category_ = self.category_.cpu()
        self._device = self.category_.device

    def cuda_(self):
        super().cuda_()
        self.category_ = self.category_.cuda()
        self._device = self.category_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.category_ = self.category_[i]
        self._len = len(self.category_)

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
    def format_category(
        category: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `category` is `(N >= 0, 1)`.

        #### Args:
        - category: categories. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Categories. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == category.ndim:
            return category.reshape(-1, 1)

        if 2 != category.ndim or 1 != category.shape[1]:
            raise ValueError(
                '`category` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return category

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(category=obj.category_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.category_ = torch.cat((self.category_, obj.category_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def update_category_(self, category: TorchTensor[TorchReal]) -> None:
        r'''Update the categories.

        Warning: This is an inplace method.

        #### Args:
        - category: categories. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        '''
        self.category_ = self.format_category(category)

        if self.category_.device != self._device:
            self._error_device()

        if len(self.category_) != self._len:
            self.__error_num(self._len)
