import torch

from ..typing import TorchTensor, TorchReal, TorchFloat
from .angle import PI


def distance_from_xyz(
    xyzf: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''
    Distance in the spherical coordinate system calculated from 3D rectangular
    coordinates.

    #### Args:
    - xyzf: coordinates in the 3D rectangular coordinate system. Its shape
        should be `([*,] 3 [+ C])`. A point should be in the form of
        `(x, y, z [, features])`.

    #### Returns:
    - Distance in the spherical coordinate system. Its shape is `([*,])`.

    '''
    return torch.linalg.norm(xyzf[..., :3], dim=-1)


def phi_from_xyz(xyzf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Phi calculated from 3D rectangular coordinates.

    #### Args:
    - xyzf: coordinates in the 3D rectangular coordinate system. Its shape
        should be `([*,] 3 [+ C])`. A point should be in the form of
        `(x, y, z [, features])`.

    #### Returns:
    - Phi in radius. Its shape is `([*,])`.

    '''
    phis = torch.atan2(xyzf[..., 2], torch.linalg.norm(xyzf[..., :2], dim=-1))
    phis[PI == phis] = -PI
    return phis


def phi_from_rtz(rtzf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Phi calculated from cylindrical coordinates.

    #### Args:
    - rtzf: coordinates in the cylindrical coordinate system. Its shape should
        be `([*,] 3 [+ C])`.A point should be in the form of
        `(rho_c, theta_c, z [, features])`.

    #### Returns:
    - Phi in radius. Its shape is `([*,])`.

    '''
    phis = torch.atan2(rtzf[..., 2], rtzf[..., 0])
    phis[phis == PI] = -PI
    return phis


def rho_from_xy(xyf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''
    Rho in the polar coordinate system calculated from rectangular coordinates.

    #### Args:
    - xyf: coordinates in the rectangular coordinate system. Its shape should
        be `([*,] 2 [+ C])`. A point should be in the form of
        `(x, y [, features])`.

    #### Returns:
    - Rho in the polar coordinate system. Its shape is `([*,])`.

    '''
    return torch.linalg.norm(xyf[..., :2], dim=-1)


def theta_from_xy(xyf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Theta calculated from rectangular coordinates.

    #### Args:
    - xyf: coordinates in the rectangular coordinate system. Its shape should
        be `([*,] 2 [+ C])`. A point should be in the form of
        `(x, y [, features])`.

    #### Returns:
    - Theta in radius. Its shape is `([*,])`.

    '''
    thetas = torch.atan2(xyf[..., 1], xyf[..., 0])
    thetas[PI == thetas] = -PI
    return thetas


# RECTANGULAR TO X
def xy_to_rt(xyf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Convert rectangular coordinates to polar coordinates.

    #### Args:
    - xyf: coordinates in the rectangular coordinate system. Its shape should
        be `([*,] 2 [+ C])`. A point should be in the form of
        `(x, y [, features])`.

    #### Returns:
    - Coordinates in the polar coordinate system. Its shape is `([*,] 2)`. The
        coordinates of a point is in the form of `(rho, theta)`. `rho` and
        `theta` in radius are calculated as follows.
                rho = sqrt(x ^ 2 + y ^ 2)
                theta = arctan(y, x)

    '''
    return torch.stack((rho_from_xy(xyf), theta_from_xy(xyf)), dim=-1)


def xyz_to_dtp(xyzf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Convert 3D rectangular coordinates to spherical coordinates.

    #### Args:
    - xyzf: coordinates in the 3D rectangular coordinate system. Its shape
        should be `([*,] 3 [+ C])`. A point should be in the form of
        `(x, y, z [, features])`.

    #### Returns:
    - Coordinates in the spherical coordinate system. Its shape is `([*,] 3)`.
        The coordinates of a point is in the form of `(distance, theta, phi)`.
        `rho`, `theta` and `phi` are calculated as follows. `theta` and `phi`
        are in radius.
                distance = sqrt(x ^ 2 + y ^ 2 + z ^ 2)
                theta = arctan(y / x)
                phi = arctan(z / sqrt(x ^ 2 + y ^ 2))

    '''
    return torch.stack(
        (
            distance_from_xyz(xyzf),
            theta_from_xy(xyzf),
            phi_from_xyz(xyzf)
        ),
        dim=-1
    )


def xyz_to_rtz(xyzf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Convert 3D rectangular coordinates to cylindrical coordinates.

    #### Args:
    - xyzf: coordinates in the 3D rectangular coordinate system. Its shape
        should be `([*,] 3 [+ C])`. A point should be in the form of
        `(x, y, z [, features])`.

    #### Returns:
    - Coordinates in the cylindrical coordinate system. Its shape is
        `([*,] 3)`. The coordinates of a point is in the form of
        `(rho, theta, z)`. `rho` and `theta` in radius are calculated as
        follows.
                rho = sqrt(x ^ 2 + y ^ 2)
                theta = arctan(y, x)

    '''
    return torch.stack(
        (
            rho_from_xy(xyzf),
            theta_from_xy(xyzf),
            xyzf[..., 2]
        ),
        dim=-1
    )


# POLAR TO X
def rt_to_xy(rtf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Convert polar coordinates to rectangular coordinates.

    #### Args:
    - rtf: coordinates in the polar coordinate system. Its shape should be
        `([*,] 2 [+ C])`. A point should be in the form of
        `(rho, theta [, features])`.

    #### Returns:
    - Coordinates in the rectangular coordinate system. Its shape is
        `([*,] 2)`. The coordinates of a point is in the form of `(x, y)`. `x`
        and `y` are calculated as follows.
                x = cos(theta) * rho
                y = sin(theta) * rho

    '''
    rhos = rtf[..., 0]
    thetas = rtf[..., 1]
    return torch.stack(
        (
            torch.cos(thetas) * rhos,
            torch.sin(thetas) * rhos
        ),
        dim=-1
    )


def rtz_to_dtp(rtzf: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Convert cylindrical coordinates to spherical coordinates.

    #### Args:
    - rtzf: coordinates in the cylindrical coordinate system. Its shape should
        be `([*,] 3 [+ C])`.A point should be in the form of
        `(rho, theta_c, z [, features])`.

    #### Returns:
    - Coordinates in the spherical coordinate system. Its shape is `([*,] 3)`.
        The coordinates of a point is in the form of
        `(distance, theta_s, phi)`. `distance`, `theta_s` and `phi` are
        calculated as follows. `theta_s` and `phi` are in radius.
                distance = sqrt(rho ^ 2 + z ^ 2)
                theta_s = theta_c
                phi = arctan(z / rho)

    '''
    return torch.stack(
        (
            torch.linalg.norm(rtzf[..., [0, 2]], dim=-1),
            rtzf[..., 1],
            phi_from_rtz(rtzf),
        ),
        dim=-1
    )


def rtz_to_xyz(rtzf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Convert cylindrical coordinates to 3D rectangular coordinates.

    #### Args:
    - rtzf: coordinates in the cylindrical coordinate system. Its shape should
        be `([*,] 3 [+ C])`. A point should be in the form of
        `(rho, theta, z [, features])`.

    #### Returns:
    - Coordinates in the 3D rectangular coordinate system. Its shape is
        `([*,] 3)`. The coordinates of a point is in the form of `(x, y, z)`.
        `x`, `y` and `z` are calculated as follows.
                x = cos(theta) * rho
                y = sin(theta) * rho
                z = z

    '''
    rhos = rtzf[..., 0]
    thetas = rtzf[..., 1]
    return torch.stack(
        (
            torch.cos(thetas) * rhos,
            torch.sin(thetas) * rhos,
            rtzf[..., 2]
        ),
        dim=-1
    )


# SPHERICAL TO X
def dtp_to_rtz(dtpf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Convert spherical coordinates to cylindrical coordinates.

    #### Args:
    - dtpf: coordinates in the spherical coordinate system. Its shape should be
        `([*,] 3 [+ C])`.A point should be in the form of
        `(distance, theta_s, phi [, features])`.

    #### Returns:
    - Coordinates in the cylindrical coordinate system. Its shape is
        `([*,] 3)`. The coordinates of a point is in the form of
        `(rho, theta_c, z)`. `rho`, `theta_c` in radius and `z` are calculated
        as follows.
                rho = sin(phi) * distance
                theta_c = theta_s
                z = cos(phi) * distance

    '''
    distances = dtpf[..., 0]
    phis = dtpf[..., 2]
    return torch.stack(
        (
            torch.sin(phis) * distances,
            dtpf[..., 1],
            torch.cos(phis) * distances
        ),
        dim=-1
    )


def dtp_to_xyz(dtpf: TorchTensor[TorchReal]) -> TorchTensor[TorchFloat]:
    r'''Convert spherical coordinates to 3D rectangular coordinates.

    #### Args:
    - dtpf: coordinates in the spherical coordinate system. Its shape should be
        `([*,] 3 [+ C])`. A point should be in the form of
        `(distance, theta, phi [, features])`.

    #### Returns:
    - Coordinates in the 3D rectangular coordinate system. Its shape is
        `([*,] 3)`. The coordinates of a point is in the form of `(x, y, z)`.
        `x`, `y` and `z` are calculated as follows.
                x = cos(theta) * sin(phi) * distance
                y = sin(theta) * sin(phi) * distance
                z = cos(phi) * distance

    '''
    rhos = dtpf[..., 0]
    thetas = dtpf[..., 1]
    phis = dtpf[..., 2]
    rhos_c = torch.sin(phis) * rhos
    return torch.stack(
        (
            torch.cos(thetas) * rhos_c,
            torch.sin(thetas) * rhos_c,
            torch.cos(phis) * rhos
        ),
        dim=-1
    )
