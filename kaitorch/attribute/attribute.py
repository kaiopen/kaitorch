from typing import Optional, Sequence, Union

from ..typing import TorchDevice, TorchReal


class Attribute:
    r'''Abstract class for box.

    #### Properties:
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

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(self, *args, **kwargs) -> None:
        self._len = -1
        self._device: Optional[TorchDevice] = None
        self.__i: int = 0

    @property
    def device(self) -> Union[TorchDevice, None]:
        return self._device

    def _error_device(self):
        raise ValueError(f'Device `{self._device}` wanted.')

    def __iter__(self):
        return self

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        raise NotImplementedError

    def __len__(self) -> int:
        return self._len

    def __next__(self):
        if self.__i < self._len:
            data = self[self.__i]
            self.__i += 1
            return data
        self.__i = 0
        raise StopIteration

    def append_(self, *args, **kwargs) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace methods.

        #### Returns:
        - Number of the appended data.

        '''
        return -1

    def copy(self):
        r'''Copy the necessary data.

        ### Returns:
            - A copy of self.

        '''
        raise NotImplementedError

    def copy_all(self):
        r'''Copy all of the data.

        #### Returns:
        - A copy of self.

        '''
        return self.copy()

    def cpu_(self):
        return

    def cuda_(self):
        return

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        return

    def flip_around_x_axis_(self) -> None:
        r'''Flip the data around the X axis.

        Warning: This is an inplace method.

        '''
        return

    def flip_around_y_axis_(self) -> None:
        r'''Flip the data around the Y axis.

        Warning: This is an inplace method.

        '''
        return

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return NotImplementedError

    def is_empty(self) -> bool:
        r'''Whether there is no data.

        #### Return:
        - Empty or not.

        '''
        return 0 == self._len

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        if obj.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(obj)
        else:
            self._len += len(obj)

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        return

    def slice_all(self, i):
        r'''Slice all of the data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self[i]
