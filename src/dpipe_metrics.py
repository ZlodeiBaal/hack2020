import numpy as np
from functools import wraps
from typing import Callable, Sequence, Iterable

import numpy as np


# --- checks
from scipy.ndimage import distance_transform_edt, binary_erosion


def extract(sequence: Sequence, indices: Iterable):
    """Extract ``indices`` from ``sequence``."""
    return [sequence[i] for i in indices]


def join(values):
    return ", ".join(map(str, values))


def check_shape_along_axis(*arrays, axis):
    sizes = [tuple(extract(x.shape, np.atleast_1d(axis))) for x in arrays]
    if any(x != sizes[0] for x in sizes):
        raise ValueError(f'Arrays of equal size along axis {axis} are required: {join(sizes)}')


def check_len(*args):
    lengths = list(map(len, args))
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {join(lengths)}')


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def check_shapes(*arrays):
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError(f'Arrays of equal shape are required: {join(shapes)}')


def add_check_function(check_function: Callable):
    """Decorator that checks the function's arguments via ``check_function`` before calling it."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_function(*args, *kwargs.values())
            return func(*args, **kwargs)

        return wrapper

    name = getattr(check_function, '__name__', '`func`')
    decorator.__doc__ = f"Check the function's arguments via `{name}` before calling it."
    return decorator


add_check_bool, add_check_shapes, add_check_len = map(add_check_function, [
    check_bool, check_shapes, check_len
])


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


# --- metrics
# TODO: STANDART METRIC(GT, PRED)
@add_check_bool
@add_check_shapes
def dice_score(y: np.ndarray, x: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y), empty_val=1)


@add_check_bool
@add_check_shapes
def iou(y: np.ndarray, x: np.ndarray) -> float:
    return fraction(np.sum(x & y), np.sum(x | y), empty_val=1)


@add_check_bool
@add_check_shapes
def surface_distances(y_true, y_pred, voxel_shape=None):
    check_bool(y_pred, y_true)
    check_shapes(y_pred, y_true)

    pred_border = np.logical_xor(y_pred, binary_erosion(y_pred))
    true_border = np.logical_xor(y_true, binary_erosion(y_true))

    dt = distance_transform_edt(~true_border, sampling=voxel_shape)
    return dt[pred_border]


@add_check_bool
@add_check_shapes
def assd(y, x, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return np.mean([sd1.mean(), sd2.mean()])


@add_check_bool
@add_check_shapes
def hausdorff_distance(y, x, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return max(sd1.max(), sd2.max())

