from typing import Sequence, Union
import torch

from ..typing import TorchTensor, TorchReal
from .attribute import Attribute


class ID(Attribute):
    r'''

    #### Args:
    - id: IDs. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

    #### Properties:
    - device
    - id: IDs. Its shape is `(N >= 0, 1)`.
    - id_: IDs. Its shape is `(N >= 0, 1)`.

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
    - update_id_: Update the IDs.

    #### Static Methods:
    - format_id: Make sure the shape of `id` is `(N >= 0, 1)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, id: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.id_ = self.format_id(id)

        if self._device is None:
            self._device = self.id_.device
        elif self.id_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.id_)
        elif len(self.id_) != self._len:
            self.__error_num(self._len)

    @property
    def id(self) -> TorchTensor[TorchReal]:
        r'''
        IDs. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.id_.clone()

    @staticmethod
    def __error_num(num: int):
        raise ValueError(f'`id` with shape `({num}, 1)` or `({num},)` wanted.')

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(id=self.id_[i])

    def append_(
        self,
        id: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - id: IDs. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        if id.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        id = self.format_id(id)

        if -1 == _len:
            _len = len(id)
            self._len += _len
        elif len(id) != _len:
            self.__error_num(_len)

        self.id_ = torch.cat((self.id_, id))
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(id=self.id)

    def cpu_(self):
        super().cpu_()
        self.id_ = self.id_.cpu()
        self._device = self.id_.device

    def cuda_(self):
        super().cuda_()
        self.id_ = self.id_.cuda()
        self._device = self.id_.device

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.id_ = self.id_[i]
        self._len = len(self.id_)

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
    def format_id(
        id: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `id` is `(N >= 0, 1)`.

        #### Args:
        - id: IDs. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        #### Returns:
        - IDs. Its shape is `(N >= 0, 1)`.

        '''
        if 1 == id.ndim:
            return id.reshape(-1, 1)

        if 2 != id.ndim or 1 != id.shape[1]:
            raise ValueError(
                '`id` with shape `(N >= 0, 1)` or `(N >= 0,)` wanted.'
            )
        return id

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(id=obj.id_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.id_ = torch.cat((self.id_, obj.id_))

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)

    def update_id_(self, id: TorchTensor[TorchReal]) -> None:
        r'''Update the IDs.

        Warning: This is an inplace method.

        #### Args:
        - id: IDs. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

        '''
        self.id_ = self.format_id(id)

        if self.id_.device != self._device:
            self._error_device()

        if len(self.id_) != self._len:
            self.__error_num(self._len)
