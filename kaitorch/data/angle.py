from typing import Tuple

import torch

from ..typing import TorchTensor, TorchFloat, TorchReal, Float, Real


PI = torch.pi
QU = PI / 2.
CI = 2 * PI


def correct_radii_(radii: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Correct and limit radii in the range `[-PI, PI)`.

    Warning: This is an inplace method.

    #### Args:
    - radii

    #### Returns:
    - Corrected radii.

    '''
    radii[radii >= PI] -= CI
    radii[radii < -PI] += CI
    return radii


def degree_to_radius(
    degree: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''Convert angle(s) from degree to radius.

    #### Args:
    - degree: angle(s) in degree.

    #### Returns:
    - Angle(s) in radius.

    '''
    return degree * (PI / 180)


def add_radius(a: Real, b: Real) -> Float:
    r'''
    Add radius `a` with `b`, meanwhile make sure the result meet the range
    `[-PI, PI)`.

    ```
    a + b
    ```

    #### Args:
    - a: radii.
    - b: radii.

    #### Returns:
    - Result radii.

    '''
    r = a + b
    if r >= PI:
        r -= CI
    elif r < -PI:
        r += CI
    return r


def minus_radius(a: Real, b: Real) -> Float:
    r'''
    Radius `a` minus `b`, meanwhile make sure the result meet the range
    `[-PI, PI)`.

    ```
    a - b
    ```

    #### Args:
    - a: radii.
    - b: radii.

    #### Returns:
    - Result radii.

    '''
    d = a - b
    if d >= PI:
        d -= CI
    elif d < -PI:
        d += CI
    return d


def add_radii(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''
    Add radii `a` with `b`, meanwhile make sure the result meet the range
    `[-PI, PI)`.

    ```
    a + b
    ```

    #### Args:
    - a: radii.
    - b: radii.

    #### Returns:
    - Result radii.

    '''
    return correct_radii_(a + b)


def minus_radii(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''
    Radii `a` minus `b`, meanwhile make sure the result meet the range
    `[-PI, PI)`.

    ```
    a - b
    ```

    #### Args:
    - a: radii.
    - b: radii.

    #### Returns:
    - Result radii.

    '''
    return correct_radii_(a - b)


# MAX & MIN
def leftmost_and_rightmost(
    radii: TorchTensor[TorchReal]
) -> Tuple[TorchReal, TorchReal]:
    r'''
    Rightmost radius and leftmost radius of objects in a polar-like coordinate
    system.

    The range of `radii` should be less than PI. In another word, the
    difference between the rightmost and the leftmost should be less than PI.

    #### Args:
    - radii: radii of points in radius. Its shape should be `(N,)`.

    #### Returns:
    - Leftmost radius.
    - Rightmost radius.

    '''
    d = minus_radii(
        torch.unsqueeze(radii, dim=-1), torch.unsqueeze(radii, dim=0)
    )  # (N, N)
    d[torch.abs(d) < 1e-6] = 0  # minus self
    return radii[torch.argmax(torch.min(d, dim=-1)[0])], \
        radii[torch.argmin(torch.max(d, dim=-1)[0])]


def leftmosts_and_rightmosts(
    radii: TorchTensor[TorchReal]
) -> Tuple[TorchTensor[TorchReal], TorchTensor[TorchReal]]:
    r'''
    Rightmost radii and leftmost radii of objects in a polar-like coordinate
    system.

    The range of `radii` should be less than PI. In another word, the
    difference between the rightmost and the leftmost should be less than PI.

    #### Args:
    - radii: radii of points in radius. Its shape should be `(B, N)`.

    #### Returns:
    - Leftmost radii. Its shape is `(B,)`.
    - Rightmost radii. Its shape is `(B,)`.

    '''
    d = minus_radii(
        torch.unsqueeze(radii, dim=-1), torch.unsqueeze(radii, dim=1)
    )  # (B, N, N)
    d[torch.abs(d) < 1e-6] = 0  # minus self
    b = torch.arange(len(radii), device=radii.device)
    return radii[b, torch.argmax(torch.min(d, dim=-1)[0], dim=-1)], \
        radii[b, torch.argmin(torch.max(d, dim=-1)[0], dim=-1)]


# ROTATE 2D
def rotation_matrix_2d(radius: TorchReal) -> TorchTensor[TorchFloat]:
    r'''Rotation matrix in the 2D rectangular coordinate system.

    (2, N) = (2, 2) * (2, N)

    #### Args;
    - radius: radius to rotate by.

    #### Returns:
    - Rotation matrix. Its shape is `(2, 2)`.

    '''
    sin = torch.sin(radius)
    cos = torch.cos(radius)
    return torch.tensor(
        [
            [cos, -sin],
            [sin, cos]
        ],
        device=radius.device
    )


def rotation_matrix_2d_t(radius: TorchReal) -> TorchTensor[TorchFloat]:
    r'''Rotation matrix in the 2D rectangular coordinate system.

    (N, 2) = (N, 2) * (2, 2)

    #### Args;
    - radius: radius to rotate by.

    #### Returns:
    - Rotation matrix. Its shape is `(2, 2)`.

    '''
    sin = torch.sin(radius)
    cos = torch.cos(radius)
    return torch.tensor(
        [
            [cos, sin],
            [-sin, cos]
        ],
        device=radius.device
    )


def rotation_matrix_3d_x(radius: Real) -> TorchTensor[TorchFloat]:
    r'''
    Rotation matrix around the X axis in the 3D rectangular coordinate system.

    (3, N) = (3, 3) * (3, N)

    #### Args:
    - radius: radius to rotate by.

    #### Returns:
    - Rotation matrix. Its shape is `(3, 3)`.

    '''
    sin = torch.sin(radius)
    cos = torch.cos(radius)
    return torch.tensor(
        [
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ]
    )


def rotation_matrix_3d_y(radius: Real) -> TorchTensor[TorchFloat]:
    r'''
    Rotation matrix around the Y axis in the 3D rectangular coordinate system.

    (3, N) = (3, 3) * (3, N)

    #### Args:
    - radius: radius to rotate by.

    #### Returns:
    - Rotation matrix. Its shape is `(3, 3)`.

    '''
    sin = torch.sin(radius)
    cos = torch.cos(radius)
    return torch.tensor(
        [
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos]
        ]
    )


def rotation_matrix_3d_z(radius: Real) -> TorchTensor[TorchFloat]:
    r'''
    Rotation matrix around the Z axis in the 3D rectangular coordinate system.

    (3, N) = (3, 3) * (3, N)

    #### Args:
    - radius: radius to rotate by.

    #### Returns:
    - Rotation matrix. Its shape is `(3, 3)`.

    '''
    sin = torch.sin(radius)
    cos = torch.cos(radius)
    return torch.tensor(
        [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ]
    )


def rotate_point_2d(
    xyf: TorchTensor[TorchReal], radius: TorchReal
) -> TorchTensor[TorchFloat]:
    r'''Rotate coordinates in the rectangular coordiante system.

    #### Args:
    - xyf: Coordinates in the rectangular coordinate system. Its shape should
        be `(N, 2 [+ C])`.
    - radius: radius to rotate by.

    #### Returns:
    - Rotated coordinates. Its shape is `(N, 2)`.

    '''
    return torch.matmul(xyf[..., :2], rotation_matrix_2d_t(radius))


def rotate_point_3d_z(
    xyzf: TorchTensor[TorchReal], radius: TorchReal
) -> TorchTensor[TorchFloat]:
    r'''
    Rotate coordinates around Z axis in the 3D rectangular coordiante system.

    #### Args:
    - xyzf: Coordinates in the 3D rectangular coordinate system. Its shape
        should be `(N, 3 [+ C])`.
    - radius: radius to rotate by.

    #### Returns:
    - Rotated coordinates. Its shape is `(N, 3)`.

    '''
    return torch.cat((rotate_point_2d(xyzf, radius), xyzf[..., 2: 3]), -1)


def rotation_matrices_2d(
    radii: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''Rotation matrices for rotating in the rectangular coordinate system.

    (R, 2, N) = (R, 2, 2) * (R, 2, N)

    #### Args;
    - radii: radii to rotate by. Its shape should be `(R,)`.

    #### Returns:
    - Rotation matrices. Its shape is `(R, 2, 2)`.

    '''
    m = torch.zeros((len(radii), 2, 2), device=radii.device)
    m[:, [0, 1], [0, 1]] = torch.cos(radii).unsqueeze(-1)
    sin = torch.sin(radii)
    m[:, 0, 1] = -sin
    m[:, 1, 0] = sin
    return m


def rotation_matrices_2d_t(
    radii: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''Rotation matrices for rotating in the rectangular coordinate system.

    (R, N, 2) = (R, N, 2) * (R, 2, 2)

    #### Args;
    - radii: radii to rotate by. Its shape should be `(R,)`.

    #### Returns:
    - Rotation matrices. Its shape is `(R, 2, 2)`.

    '''
    m = torch.zeros((len(radii), 2, 2), device=radii.device)
    m[:, [0, 1], [0, 1]] = torch.cos(radii).unsqueeze(-1)
    sin = torch.sin(radii)
    m[:, 0, 1] = sin
    m[:, 1, 0] = -sin
    return m


def rotate_points_2d(
    xyf: TorchTensor[TorchReal], radii: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''Rotate coordinates in the rectangular coordiante system.

    #### Args:
    - xyf: Coordinates in the rectangular coordinate system. Its shape should
        be `(R, N, 2 [+ C])`.
    - radii: radii to rotate by. Its shape should be `(R,)`.

    #### Returns:
    - Rotated coordinates. Its shape is `(R, N, 2)`.

    '''
    return torch.matmul(xyf[..., :2], rotation_matrices_2d_t(radii))


def rotate_points_3d_z(
    xyzf: TorchTensor[TorchReal], radii: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''
    Rotate coordinates around Z axis in the 3D rectangular coordiante system.

    #### Args:
    - xyzf: Coordinates in the 3D rectangular coordinate system. Its shape
        should be `(R, N, 3 [+ C])`.
    - radii: radii to rotate by. Its length should be `R`.

    #### Returns:
    - Rotated coordinates. Its shape is `(R, N, 3)`.

    '''
    return torch.cat((rotate_points_2d(xyzf, radii), xyzf[..., 2: 3]), -1)
