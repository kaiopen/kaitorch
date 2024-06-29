from typing import Sequence, Union

from ..typing import TorchTensor, TorchBool, TorchReal, real


def mask_in_range(
    a: TorchTensor[TorchReal], r: Sequence[real]
) -> Union[TorchTensor[TorchBool], bool]:
    r'''Mask points in the range.

    Given a data `(c0, c1, c2)` and a range `(min_0, max_0, min_1, max_1)`. If
    `c0 >= min_0 & c0 < max_0 & c1 >= min_1 & c1 < max_1`, the data is in the
    range.

    If there is a `None` in the `r`, the comparision will be skipped.

    ### Args:
        - a: data. Its shape should be `([*,] C)`.
        - r: a right half-open interval in which the data is masked. It should
            be in the form of
            `(min_0, max_0, min_1, max_1, ..., min_n, max_n)`, where `n` must
            be less than `C`.

    ### Returns:
        - Mask for the preserved data which is in the range. Its shape is
            `([*,])`. If `True`, the data is preserved.

    '''
    mask = True
    for i in range(len(r) // 2):
        c = a[..., i]
        i *= 2
        mi = r[i]
        if mi is not None:
            mask &= c >= mi
        ma = r[i + 1]
        if ma is not None:
            mask &= c < ma
    return mask


def mask_radii_in_range(
    radii: Union[TorchTensor[TorchReal], real], r: Sequence[real]
) -> Union[TorchTensor[TorchBool], bool]:
    r'''Mask radii in the range.

    Given a radius `r` and a range `(minR, maxR)`. If `r >= minR & r < maxR`,
    the radius is in the range.

    ### Args:
        - radii: radii in radius.
        - r: a right half-open interval in which the radii are masked. It
            should be in the form of `(left, right)`.

    ### Returns:
        - Mask for the preserved radii which are in the range. If `True`, the
            radius is preserved.

    '''
    mi = r[0]
    ma = r[1]
    if mi > ma:
        return (radii >= mi) | (radii < ma)
    return (radii >= mi) & (radii < ma)


def mask_in_closed_range(
    a: TorchTensor[TorchReal], r: Sequence[real]
) -> Union[TorchTensor[TorchBool], bool]:
    r'''Mask points in the range.

    Given a data `(c0, c1, c2)` and a range `(min_0, max_0, min_1, max_1)`. If
    `c0 >= min_0 & c0 <= max_0 & c1 >= min_1 & c1 <= max_1`, the data is in the
    range.

    If there is a `None` in the `r`, the comparision will be skipped.

    ### Args:
        - a: data. Its shape should be `([*,] C)`.
        - r: a closed interval in which the data is masked. It should be in the
            form of `(min_0, max_0, min_1, max_1, ..., min_n, max_n)`, where
            `n` must be less than `C`.

    ### Returns:
        - Mask for the preserved data which is in the range. Its shape is
            `([*,])`. If `True`, the data is preserved.

    '''
    mask = True
    for i in range(len(r) // 2):
        c = a[..., i]
        i *= 2
        mi = r[i]
        if mi is not None:
            mask &= c >= mi
        ma = r[i + 1]
        if ma is not None:
            mask &= c <= ma
    return mask


def mask_radii_in_closed_range(
    radii: Union[TorchTensor[TorchReal], real], r: Sequence[real]
) -> Union[TorchTensor[TorchBool], bool]:
    r'''Mask radii in the range.

    Given a radius `r` and a range `(minR, maxR)`. If `r >= minR & r <= maxR`,
    the radius is in the range.

    ### Args:
        - radii: radii in radius.
        - r: a closed interval in which the radii are masked. It should be in
            the form of `(left, right)`.

    ### Returns:
        - Mask for the preserved radii which are in the range. If `True`, the
            radius is preserved.

    '''
    mi = r[0]
    ma = r[1]
    if mi > ma:
        return (radii >= mi) | (radii <= ma)
    return (radii >= mi) & (radii <= ma)
