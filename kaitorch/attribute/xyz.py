from typing import Optional, Sequence, Union
import torch

from ..typing import TorchTensor, TorchFloat, TorchReal
from ..data import distance_from_xyz, phi_from_xyz, phi_from_rtz, \
    rho_from_xy, rotate_point_2d, theta_from_xy
from .attribute import Attribute


class XYZ(Attribute):
    r'''

    #### Args:
    - xyzf: 3D coordinates in the 3D rectangular coordinate system. Its shape
        should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A coordinate
        should be in the form of `(x, y, z)`.

    #### Properties:
    - device
    - distance: distance to the origin. Its shape is `(N >= 0, 1)`.
    - distance_: distance to the origin. Its shape is `(N >= 0, 1)`.
    - dtp: 3D coordinates in the spherical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(distance, theta, phi)`.
        And both the `theta` and the `phi` are in radius.
    - dtp_: 3D coordinates in the spherical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(distance, theta, phi)`.
        And both the `theta` and the `phi` are in radius.
    - phi: phis in radius in the spherical coordinate system. Its shape is
        `(N >= 0, 1)`.
    - phi_: phis in radius in the spherical coordinate system. Its shape is
        `(N >= 0, 1)`.
    - rho: rhos in the polar coordinate system. Its shape is `(N >= 0, 1)`.
    - rho_: rhos in the polar coordinate system. Its shape is `(N >= 0, 1)`.
    - rt: 2D coordinates in the polar coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`. And the
        `theta` is in radius.
    - rt_: 2D coordinates in the polar coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`. And the
        `theta` is in radius.
    - rtz: 3D coordinates in the cylindrical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(rho, theta, z)`. And
        the `theta` is in radius.
    - rtz_: 3D coordinates in the cylindrical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(rho, theta, z)`. And
        the `theta` is in radius.
    - theta: thetas in radius in the polar coordinate system. Its shape is
        `(N >= 0, 1)`.
    - theta_: thetas in radius in the polar coordinate system. Its shape is
        `(N >= 0, 1)`.
    - xy: 2D coordinates in the rectangular coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(x, y)`.
    - xy_: 2D coordinates in the rectangular coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(x, y)`.
    - xyz: 3D coordinates in the 3D rectangular coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.
    - xyz_: 3D coordinates in the 3D rectangular coordinate system. Its shape
        is `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.
    - z: zs in the 3D rectangular coordinate system of the cylindrical
        coordinate system. Its shape is `(N >= 0, 1)`.
    - z_: zs in the 3D rectangular coordinate system of the cylindrical
        coordinate system. Its shape is `(N >= 0, 1)`.

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
    - update_xyz_: Update the coordinates.

    #### Static Methods:
    - format_xyz: Make sure the shape of `xyzf` is `(N >= 0, 3)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self, xyzf: TorchTensor[TorchReal], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.xyz_ = self.format_xyz(xyzf)

        if self._device is None:
            self._device = self.xyz_.device
        elif self.xyz_.device != self._device:
            self._error_device()

        if -1 == self._len:
            self._len = len(self.xyz_)
        elif len(self.xyz_) != self._len:
            self.__error_num(self._len)

        self._distance_: Optional[TorchTensor[TorchReal]] = None
        self._phi_: Optional[TorchTensor[TorchReal]] = None
        self._rho_: Optional[TorchTensor[TorchReal]] = None
        self._theta_: Optional[TorchTensor[TorchReal]] = None

        self._dtp_: Optional[TorchTensor[TorchFloat]] = None
        self._rtz_: Optional[TorchTensor[TorchFloat]] = None
        # If dtp is not None, distance, theta and phi must not be None.
        # If rtz is not None, rho and theta must not be None.
        # If rho is None, rtz must be None.
        # If theta is None, rtz and dtp must be None.
        # If distance is None, dtp must be None.
        # If phi is None, dtp must be None.

        # rho, theta, distance, phi, rtz, dtp share the same storage:
        # If rtz is not None, rho share the storage with rtz.
        # If dtp is not None, distance and phi share the storage with dtp.
        # If rtz is not None or dtp is not None, theta share the storage with
        # rtz or dtp.

    @property
    def distance_(self) -> TorchTensor[TorchReal]:
        if self._distance_ is None:
            self._distance_ = distance_from_xyz(self.xyz_).unsqueeze_(-1)
        return self._distance_

    @property
    def phi_(self) -> TorchTensor[TorchReal]:
        if self._phi_ is None:
            if self._rtz_ is None:
                self._phi_ = phi_from_xyz(self.xyz_).unsqueeze_(-1)
            else:
                self._phi_ = phi_from_rtz(self._rtz_).unsqueeze_(-1)
        return self._phi_

    @property
    def rho_(self) -> TorchTensor[TorchReal]:
        if self._rho_ is None:
            self._rho_ = rho_from_xy(self.xyz_).unsqueeze_(-1)
        return self._rho_

    @property
    def theta_(self) -> TorchTensor[TorchReal]:
        if self._theta_ is None:
            self._theta_ = theta_from_xy(self.xyz_).unsqueeze_(-1)
        return self._theta_

    @property
    def z_(self) -> TorchTensor[TorchReal]:
        return self.xyz_[:, 2: 3]

    @property
    def xy_(self) -> TorchTensor[TorchReal]:
        return self.xyz_[:, 0: 2]

    @property
    def dtp_(self) -> TorchTensor[TorchFloat]:
        if self._dtp_ is None:
            if self._distance_ is None:
                self._distance_ = distance_from_xyz(self.xyz_).unsqueeze_(-1)

            if self._theta_ is None:
                self._theta_ = theta_from_xy(self.xyz_).unsqueeze_(-1)

            if self._phi_ is None:
                if self._rtz_ is None:
                    self._phi_ = phi_from_xyz(self.xyz_).unsqueeze_(-1)
                else:
                    self._phi_ = phi_from_rtz(self._rtz_).unsqueeze_(-1)

            self._dtp_ = torch.cat(
                (self._distance_, self._theta_, self._phi_), dim=-1
            )
            self._distance_ = self._dtp_[:, 0: 1]
            self._theta_ = self._dtp_[:, 1: 2]
            self._phi_ = self._dtp_[:, 2: 3]
        return self._dtp_

    @property
    def rt_(self) -> TorchTensor[TorchFloat]:
        if self._rtz_ is not None:
            return self._rtz_[:, :2]
        return torch.cat((self.rho_, self.theta_), dim=-1)

    @property
    def rtz_(self) -> TorchTensor[TorchFloat]:
        if self._rtz_ is None:
            if self._rho_ is None:
                self._rho_ = rho_from_xy(self.xyz_).unsqueeze_(-1)

            if self._theta_ is None:
                self._theta_ = theta_from_xy(self.xyz_).unsqueeze_(-1)

            self._rtz_ = torch.cat(
                (self._rho_, self._theta_, self.xyz_[:, 2: 3]), dim=-1
            )
            self._rho_ = self._rtz_[:, 0: 1]
            self._theta_ = self._rtz_[:, 1: 2]
        return self._rtz_

    @property
    def distance(self) -> TorchTensor[TorchFloat]:
        r'''
        Distances to the origin. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.distance_.clone()

    @property
    def phi(self) -> TorchTensor[TorchFloat]:
        r'''
        Phis in radius in the spherical coordinate system. Its shape is
        `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.phi_.clone()

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
    def z(self) -> TorchTensor[TorchReal]:
        r'''
        Zs in the 3D rectangular coordinate system of the cylindrical
        coordinate system. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.z_.clone()

    @property
    def rt(self) -> TorchTensor[TorchFloat]:
        r'''
        2D coordinates in the polar coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(rho, theta)`. And the
        `theta` is in radius.

        This is a copy of the data stored.

        '''
        if self._rtz_ is not None:
            return self._rtz_[:, :2].clone()
        return torch.cat((self.rho_, self.theta_), dim=-1)

    @property
    def dtp(self) -> TorchTensor[TorchFloat]:
        r'''
        3D coordinates in the spherical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(distance, theta, phi)`.
        And both the `theta` and the `phi` are in radius.

        This is a copy of the data stored.

        '''
        return self.dtp_.clone()

    @property
    def rtz(self) -> TorchTensor[TorchFloat]:
        r'''
        3D coordinates in the cylindrical coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(rho, theta, z)`. And
        the `theta` is in radius.

        This is a copy of the data stored.

        '''
        return self.rtz_.clone()

    @property
    def xy(self) -> TorchTensor[TorchReal]:
        r'''
        2D coordinates in the rectangular coordinate system. Its shape is
        `(N >= 0, 2)`. A coordinate is in the form of `(x, y)`.

        This is a copy of the data stired.

        '''
        return self.xy_.clone()

    @property
    def xyz(self) -> TorchTensor[TorchReal]:
        r'''
        3D coordinates in the 3D rectangular coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.

        This is a copy of the data stored.

        '''
        return self.xyz_.clone()

    @staticmethod
    def __error_num(num: int):
        if 0 == num:
            raise ValueError('`xyzf` with shape `(0, [C])` wanted.')
        elif 1 == num:
            raise ValueError(
                '`xyzf` with shape `(1, 3 [+ C])` or `(3 [+ C],)` wanted.'
            )
        raise ValueError(f'`xyzf` with shape `({num}, 3 [+ C])` wanted.')

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(xyzf=self.xyz_[i])

    def __reset_(self):
        r'''Clear and reset additional data.

        '''
        self._distance_ = None
        self._phi_ = None
        self._rho_ = None
        self._theta_ = None

        self._dtp_ = None
        self._rtz_ = None

    def append_(self, xyzf: TorchTensor[TorchReal], *args, **kwargs) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - xyzf: 3D coordinates in the 3D rectangular coordinate system. Its
            shape should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A
            coordinate should be in the form of `(x, y, z)`.

        ### Returns:
        - Number of the appended boxes.

        '''
        if xyzf.device != self._device:
            self._error_device()

        _len = super().append_(*args, **kwargs)
        xyzf = self.format_xyz(xyzf)

        if -1 == _len:
            _len = len(xyzf)
            self._len += _len
        elif len(xyzf) != _len:
            self.__error_num(_len)

        self.xyz_ = torch.cat((self.xyz_, xyzf))
        self.__reset_()
        return _len

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(xyzf=self.xyz)

    def copy_all(self):
        r'''Copy all of the data.

        #### Returns:
        - A copy of self.

        '''
        c = super().copy_all()

        if self._dtp_ is not None:
            c._dtp_ = self._dtp_.clone()
            c._distance_ = c._dtp_[:, 0: 1]
            c._theta_ = c._dtp_[:, 1: 2]
            c._phi_ = c._dtp_[:, 2: 3]
        else:
            if self._distance_ is not None:
                c._distance_ = self._distance_.clone()
            if self._theta_ is not None:
                c._theta_ = self._theta_.clone()
            if self._phi_ is not None:
                c._phi_ = self._phi_.clone()

        if self._rtz_ is not None:
            c._rtz_ = self._rtz_.clone()
            c._rho_ = c._rtz_[:, 0: 1]
            c._theta_ = c._rtz_[:, 1: 2]
        else:
            if self._rho_ is not None:
                c._rho_ = self._rho_.clone()
            if c._theta_ is None and self._theta_ is not None:
                c._theta_ = self._theta_.clone()
        return c

    def cpu_(self):
        super().cpu_()
        self.xyz_ = self.xyz_.cpu()
        self._device = self.xyz_.device
        self.__reset_()

    def cuda_(self):
        super().cuda_()
        self.xyz_ = self.xyz_.cuda()
        self._device = self.xyz_.device
        self.__reset_()

    def filter_(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Filter the necessary data.

        Warning: This is an inplace method.

        #### Args:
        - i: index, slice, mask or indices.

        '''
        super().filter_(i)
        self.xyz_ = self.xyz_[i]
        self._len = len(self.xyz_)
        self.__reset_()

    def flip_around_x_axis_(self) -> None:
        r'''Flip the data around the X axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_x_axis_()
        self.xyz_[:, 1] = -self.xyz_[:, 1]
        self.__reset_()

    def flip_around_y_axis_(self) -> None:
        r'''Flip the data around the Y axis.

        Warning: This is an inplace method.

        '''
        super().flip_around_y_axis_()
        self.xyz_[:, 0] = -self.xyz_[:, 0]
        self.__reset_()

    @staticmethod
    def format_xyz(xyzf: TorchTensor[TorchReal]) -> TorchTensor[TorchReal]:
        r'''Make sure the shape of `xyzf` is `(N >= 0, 3)`.

        #### Args:
        - xyzf: 3D coordinates in the 3D rectangular coordiante system. Its
            shape should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A
            coordinate should be in the form of `(x, y, z)`.

        #### Returns:
        - 3D coordinates in the 3D rectangular coordinate system. Its shape is
            `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.

        '''
        if 1 == xyzf.ndim:
            if (s0 := len(xyzf)) >= 3 or 0 == s0:
                return xyzf[:3].reshape(-1, 3)
            raise ValueError(
                '`xyzf` with shape `(0, [C])` or `(3 [+ C])` wanted.'
            )

        if 2 != xyzf.ndim or xyzf.shape[1] < 3:
            raise ValueError(
                '`xyzf` with shape `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or '
                + '`(0, [C])` wanted.'
            )
        return xyzf[:, :3]

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(xyzf=obj.xyz_)

    def merge_(self, obj) -> None:
        r'''Merge the two.

        Warning: This is an inplace method.

        #### Args:
        - obj

        '''
        super().merge_(obj)
        self.xyz_ = torch.cat((self.xyz_, obj.xyz_))
        self.__reset_()

    def rotate_around_z_axis_(self, radius: TorchReal) -> None:
        r'''Rotate the data around the Z axis.

        Warning: This is an inplace method.

        #### Args:
        - radius: radius to rotate by in radius.

        '''
        super().rotate_around_z_axis_(radius)
        self.xyz_[:, :2] = rotate_point_2d(self.xyz_, radius)
        self.__reset_()

    def scale_(self, scale: TorchReal) -> None:
        r'''Scale the data.

        Warning: This is an inplace method.

        #### Args:
        - scale

        '''
        self.xyz_ *= scale
        self.__reset_()

    def slice_all(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice all of the data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        c = super().slice_all(i)

        if self._distance_ is not None:
            c._distance_ = self._distance_[i]
        if self._phi_ is not None:
            c._phi_ = self._phi_[i]
        if self._rho_ is not None:
            c._rho_ = self._rho_[i]
        if self._theta_ is not None:
            c._theta_ = self._theta_[i]

        if self._dtp_ is not None:
            c._dtp_ = self._dtp_[i]
        if self._rtz_ is not None:
            c._rtz_ = self._rtz_[i]
        return c

    def update_xyz_(self, xyzf: TorchTensor[TorchReal]) -> None:
        r'''Update the coordinates.

        Warning: This is an inplace method.

        #### Args:
        - xyzf: 3D coordinates in the 3D rectangular coordinate system. Its
            shape should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A
            coordinate should be in the form of `(x, y, z)`.

        '''
        self.xyz_ = self.format_xyz(xyzf)

        if self.xyz_.device != self._device:
            self._error_device()

        if len(self.xyz_) != self._len:
            self.__error_num(self._len)

        self.__reset_()
