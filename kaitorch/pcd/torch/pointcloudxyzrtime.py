from typing import Sequence, Union

from ...typing import TorchTensor, TorchReal
from ...attribute import Ring, Timestamp, XYZ


class PointCloudXYZRTime(XYZ, Ring, Timestamp):
    r'''

    #### Args:
    - xyzf: 3D coordinates in the 3D rectangular coordinate system. Its shape
        should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A coordinate
        should be in the form of `(x, y, z)`.
    - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.
    - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.

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
    - ring: rings. Its shape is `(N >= 0, 1)`.
    - ring_: rings. Its shape is `(N >= 0, 1)`.
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
    - timestamp: timestamps. Its shape is `(N >= 0, 1)`.
    - timestamp_: timestamps. Its shape is `(N >= 0, 1)`.
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
    - reset_: Clear and reset the additional data.
    - rotate_around_z_axis_: Rotate the data around the Z axis.
    - scale_: Scale the data.
    - slice_all: Slice all of the data.
    - synchronize_with_: Synchronize with the input.
    - update_ring_: Update the rings.
    - update_timestamp_: Update the timestamps.
    - update_xyz_: Update the coordinates.

    #### Static Methods:
    - format_ring: Make sure the shape of `ring` is `(N >= 0, 1)`.
    - format_timestamp: Make sure the shape of `timestamp` is `(N >= 0, 1)`.
    - format_xyz: Make sure the shape of `xyzf` is `(N >= 0, 3)`.

    #### Class Methods:
    - from_similar: New data from the input.

    '''
    def __init__(
        self,
        xyzf: TorchTensor[TorchReal],
        ring: TorchTensor[TorchReal],
        timestamp: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> None:
        super().__init__(
            xyzf=xyzf, ring=ring, timestamp=timestamp, *args, **kwargs
        )

    def __getitem__(self, i: Union[int, slice, Sequence[Union[int, bool]]]):
        r'''Slice the necessary data.

        #### Args:
        - i: index, slice, mask or indices.

        #### Returns:
        - A view of self.

        '''
        return self.__class__(
            xyzf=self.xyz_[i], ring=self.ring_[i], timestamp=self.timestamp_[i]
        )

    def append_(
        self,
        xyzf: TorchTensor[TorchReal],
        ring: TorchTensor[TorchReal],
        timestamp: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> int:
        r'''Append new data to the existed data.

        Warning: This is an inplace method.

        #### Args:
        - xyzf: 3D coordinates in the 3D rectangular coordinate system. Its
            shape should be `(N >= 0, 3 [+ C])`, `(3 [+ C],)` or `(0, [C])`. A
            coordinate should be in the form of `(x, y, z)`.
        - ring: rings. Its shape should be `(N >= 0, 1)` or `(N >= 0,)`.
        - timestamp: timestamps. Its shape should be `(N >= 0, 1)` or
            `(N >= 0,)`.

        #### Returns:
        - Number of the appended boxes.

        '''
        return super().append_(
            xyzf=xyzf, ring=ring, timestamp=timestamp, *args, **kwargs
        )

    def copy(self):
        r'''Copy the necessary data.

        #### Returns:
        - A copy of self.

        '''
        return self.__class__(
            xyzf=self.xyz,  ring=self.ring, timestamp=self.timestamp
        )

    @classmethod
    def from_similar(cls, obj):
        r'''New data from the input.

        #### Args:
        - obj

        #### Returns:
        - Data sharing the storage memory with the input.

        '''
        return cls(xyzf=obj.xyz_, ring=obj.ring_, timestamp=obj.timestamp_)
