from typing import Optional, Sequence, Union
import torch

from ..typing import TorchTensor, TorchFloat, TorchReal
from ..data import rho_from_xy, rotate_point_2d, theta_from_xy
from .attribute import Attribute


class XY(Attribute):
    def __init__(
        self, xyf: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - xyf: 2D coordinates in the rectangular coordinate system. Its
                shape should be `(N >= 0, 2 [+ C])`, `(2 [+ C],)` or
                `(0, [C])`. A coordinate should be in the form of `(x, y)`.

        ### Properties:
            - device
            - rho: rhos in the polar coordinate system. Its shape is
                `(N >= 0, 1)`.
            - rho_: rhos in the polar coordinate system. Its shape is
                `(N >= 0, 1)`.
            - rt: 2D coordinates in the polar coordinate system. Its shape is
                `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`.
                And the `theta` is in radius.
            - rt_: 2D coordinates in the polar coordinate system. Its shape is
                `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`.
                And the `theta` is in radius.
            - theta: thetas in radius in the polar cooridnate system. Its shape
                is `(N >= 0, 1)`.
            - theta_: thetas in radius in the polar cooridnate system. Its
                shape is `(N >= 0, 1)`.
            - xy: 2D coordinates in the rectangular coordinate system. Its
                shape is `(N >= 0, 2)`. A coordinate is in the form of
                `(x, y)`.
            - xy_: 2D coordinates in the rectangular coordinate system. Its
                shape is `(N >= 0, 2)`. A coordinate is in the form of
                `(x, y)`.

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
            - scale_: Scale the data.
            - slice_all: Slice all of the data.
            - update_xy_: Update the coordinates.

        ### Static Methods:
            - format_xy: Make sure the shape of `xyf` is `(N >= 0, 2)`.

        ### Class Methods:
            - from_similar: New data from the input.

        '''
        super().__init__(*args, **kwargs)
        self.xy_ = self.format_xy(xyf)

        if self._device is None:
            self._device = self.xy_.device
        elif self.xy_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.xy_)
        elif len(self.xy_) != self._len:
            self.__error_num(self._len)

        self._rho_: Optional[TorchTensor[TorchReal]] = None
        self._theta_: Optional[TorchTensor[TorchReal]] = None

        self._rt_: Optional[TorchTensor[TorchFloat]] = None
        # If rt is not None, rho and theta must not be None.
        # If rho is None, rt must be None.
        # If theta is None, rt must be None.

        # rho, theta, rt share the same storage:
        # If rt is not None, both rho and theta share the storage with rt.

    @property
    def rho_(self) -> TorchTensor[TorchReal]:
        if self._rho_ is None:
            self._rho_ = rho_from_xy(self.xy_).unsqueeze_(-1)
        return self._rho_

    @property
    def theta_(self) -> TorchTensor[TorchReal]:
        if self._theta_ is None:
            self._theta_ = theta_from_xy(self.xy_).unsqueeze_(-1)
        return self._theta_

    @property
    def rt_(self) -> TorchTensor[TorchFloat]:
        if self._rt_ is None:
            if self._rho_ is None:
                self._rho_ = rho_from_xy(self.xy_).unsqueeze_(-1)

            if self._theta_ is None:
                self._theta_ = theta_from_xy(self.xy_).unsqueeze_(-1)

            self._rt_ = torch.cat((self._rho_, self._theta_), dim=-1)
            self._rho_ = self._rt_[:, 0: 1]
            self._theta_ = self._rt_[:, 1: 2]
        return self._rt_

    @property
    def rho(self) -> TorchTensor[TorchFloat]:
        r'''
        Rhos in the polar coordinate system. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.rho_.clone()

    @property
    def theta(self) -> TorchTensor[TorchFloat]:
        r'''
        Thetas in radius in the polar cooridnate system. Its shape is
        `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.theta_.clone()

    @property
    def rt(self) -> TorchTensor[TorchFloat]:
        r'''
        2D coordinates in the polar coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`. And the
        `theta` is in radius.

        This is a copy of the data stored.

        '''
        return self.rt_.clone()

    @property
    def xy(self) -> TorchTensor[TorchReal]:
        r'''
        2D coordinates in the rectangular coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(x, y)`.

        This is a copy of the data stired.

        '''
        return self.xy_.clone()

    @staticmethod
    def __error_num(num: int):
        if 0 == num:
            raise ValueError('`xyf` with shape `(0, [C])` wanted.')
        elif 1 == num:
            raise ValueError(
                '`xyf` with shape `(1, 2 [+ C])` or `(2 [+ C],)` wanted.'
            )
        raise ValueError(f'`xyf` with shape `({num}, 2 [+ C])` wanted.')

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        ### Args:
            - i: index, slice, mask or indices.

        ### Returns:
            - A view of self.

        '''
        return self.__class__(xy=self.xy_[i])

    def __reset_(self):
        r'''Clear and reset additional data.

        '''
        self._rho_ = None
        self._theta_ = None

        self._rt_ = None

    def append_(self, xyf: TorchTensor[TorchReal], *args, **kwargs) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        ### Args:
            - xyf: 2D coordinates in the rectangular coordinate system. Its
                shape should be `(N >= 0, 2 [+ C])`, `(2 [+ C],)` or
                `(0, [C])`. A coordinate should be in the form of `(x, y)`.

        ### Returns:
            - Number of the appended boxes.

        '''
        if xyf.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        xyf = self.format_xy(xyf)

        if -1 == _len:
            _len = len(xyf)
            self._len += _len
        elif len(xyf) != _len:
            self.__error_num(_len)

        self.xy_ = torch.cat((self.xy_, xyf))
        self.__reset_()
        return _len

    def copy(self):
        r'''Copy the necessary data.

        ### Returns:
            - A copy of self.

        '''
        return self.__class__(xy=self.xy)

    def copy_all(self):
        r'''Copy all of the data.

        ### Returns:
            - A copy of self.

        '''
        c = super().copy_all()

        if self._rt_ is not None:
            rt = self._rt_.clone()
            c._rt_ = rt
            c._rho_ = rt[:, 0: 1]
            c._theta_ = rt[:, 1: 2]
        else:
            if self._rho_ is not None:
                c._rho_ = self._rho_.clone()
            if self._theta_ is not None:
                c._theta_ = self._theta_.clone()
        return c

    def cpu_(self):
        super().cpu_()
        self.xy_ = self.xy_.cpu()
        self._device = self.xy_.device
        self.__reset_()

    def cuda_(self):
        super().cuda_()
        self.xy_ = self.xy_.cuda()
        self._device = self.xy_.device
        self.__reset_()

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        ### Args:
            - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.xy_ = self.xy_[i]
        self._len = len(self.xy_)
        self.__reset_()

    def flip_around_x_axis_(self) -> None:
        r'''Flip the data around the X axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_x_axis_()
        self.xy_[:, 1] = -self.xy_[:, 1]
        self.__reset_()

    def flip_around_y_axis_(self) -> None:
        r'''Flip the data around the Y axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_y_axis_()
        self.xy_[:, 0] = -self.xy_[:, 0]
        self.__reset_()

    @staticmethod
    def format_xy(xyf: TorchTensor[TorchReal]) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `xyf` is `(N >= 0, 2)`.

        ### Args:
            - xyf: 2D coordinates in the rectangular coordiante system. Its
                shape should be `(N >= 0, 2 [+ C])`, `(2 [+ C],)` or
                `(0, [C])`. A coordinate should be in the form of `(x, y)`.

        ### Returns:
            - 2D coordinates in the rectangular coordinate system. Its shape
                is `(N >= 0, 2)`. A coordinate is in the form of `(x, y)`.

        '''
        if 1 == xyf.ndim:
            if (s0 := len(xyf)) >= 2 or 0 == s0:
                return xyf[:2].reshape(-1, 2)
            raise ValueError(
                '`xyf` with shape `(0, [C])` or `(2 [+ C])` wanted.'
            )

        if 2 != xyf.ndim or xyf.shape[1] < 2:
            raise ValueError(
                '`xyf` with shape `(N >= 0, 2 [+ C])`, `(2 [+ C],)` or '
                + '`(0, [C])` wanted.'
            )
        return xyf[:, :2]

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        ### Args:
            - obj

        ### Returns:
            - Data sharing the storage memory with the input.

        '''
        return cls(xy=obj.xy_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        ### Args:
            - obj

        '''
        super().merge_(obj)
        self.xy_ = torch.cat((self.xy_, obj.xy_))
        self.__reset_()

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        ### Args:
            - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)
        self.xy_ = rotate_point_2d(self.xy_, radius)
        self.__reset_()

    def scale_(self, scale: TorchReal) -> None:
        r'''Scale the data.

        Warning: This is an inplace method.

        ### Args:
            - scale

        '''
        self.xy_ *= scale
        self.__reset_()

    def slice_all(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice all of the data.

        ### Args:
            - i: index, slice, mask or indices.

        ### Returns:
            - A view of self.

        '''
        c = super().slice_all(i)

        if self._rho_ is not None:
            c._rho_ = self._rho_[i]
        if self._theta_ is not None:
            c._theta_ = self._theta_[i]

        if self._rt_ is not None:
            c._rt_ = self._rt_[i]
        return c

    def update_xy_(self, xyf: TorchTensor[TorchReal]) -> None:
        r'''Update the coordinates.

        Warning: This is an inplace method.

        ### Args:
            - xyf: 2D coordinates in the rectangular coordinate system. Its
                shape should be `(N >= 0, 2 [+ C])`, `(2 [+ C],)` or
                `(0, [C])`. A coordinate should be in the form of `(x, y)`.

        '''
        self.xy_ = self.format_xy(xyf)

        if self.xy_.device != self._device:
            self._error_device()

        if len(self.xy_) != self._len:
            self.__error_num(self._len)

        self.__reset_()
