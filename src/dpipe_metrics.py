import numpy as np
from functools import wraps, partial
from typing import Callable, Sequence, Iterable

import numpy as np
from scipy.ndimage import center_of_mass

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


# @add_check_bool
# @add_check_shapes
def iou(y: np.ndarray, x: np.ndarray) -> float:
    return fraction(np.sum(x & y), np.sum(x | y), empty_val=1)


# @add_check_bool
# @add_check_shapes
def surface_distances(y_true, y_pred, voxel_shape=None):
    check_bool(y_pred, y_true)
    check_shapes(y_pred, y_true)

    pred_border = np.logical_xor(y_pred, binary_erosion(y_pred))
    true_border = np.logical_xor(y_true, binary_erosion(y_true))

    dt = distance_transform_edt(~true_border, sampling=voxel_shape)
    return dt[pred_border]


# @add_check_bool
# @add_check_shapes
def assd(y, x, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return np.mean([sd1.mean(), sd2.mean()])


# @add_check_bool
# @add_check_shapes
def hausdorff_distance(y, x, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return max(sd1.max(), sd2.max())


def centres_distance(y, x, norm):
    c1, c2 = center_of_mass(y), center_of_mass(x)
    c1, c2 = np.array(c1), np.array(c2)

    return norm(c1, c2)

l1_centres_distance = partial(centres_distance, norm=lambda cc1, cc2: np.abs(cc1-cc2).sum())
l2_centres_distance = partial(centres_distance, norm=lambda cc1, cc2: np.linalg.norm(cc1-cc2))


# --- normas
def volume(x):
    return x.sum()


# --- connected components stuff
def get_metric_matrix(gt, pred, metric=iou):
    """
    Возвращает матрицу количество инстансов GT x PRED с пересечениями
    :param gt:
    :param pred:
    :param metric:
    :return:
    """
    gt, pred = np.array(gt).astype(bool), np.array(pred).astype(bool)
    return np.array([[metric(gt_cc, pred_cc) for pred_cc in pred] for gt_cc in gt])


def get_matched_components(metric_matrix, metric_th):
    """
    Возвращает matching_lists, unmatched_gt, unmatched_pred.

    matching_lists - лист из номеров замэтчившихся связных компонент размера кол-во интсансов в GT:
                    то есть если 0ой инстанс GT мэтчится с 1,2,3 PRED, а 1ый инстанс GT не мэтчится ни с чем,
                    то matching list выглядит как: [ [1,2,3], [], ...]

    unmatched_gt, unmatched_pred - list of indexes

    :param metric_matrix: то, что возвращает get_metric_matrix
    :param metric_th:
    :return:
    """

    matched_gt = set()
    matched_pred = set()
    matching_lists = []

    gt_indexes = np.array(range(metric_matrix.shape[0]))
    pred_indexes = np.array(range(metric_matrix.shape[1]))

    for i, gt_cc_metrics in zip(gt_indexes, metric_matrix):
        res = pred_indexes[gt_cc_metrics > metric_th]
        if len(res):
            matched_gt.add(i)
            for j in res:
                matched_pred.add(j)

        matching_lists.append(res)

    return matching_lists, list(sorted(set(gt_indexes) - matched_gt)), list(sorted(set(pred_indexes) - matched_pred))


def collect_instance_from_matched_indexes(gt, pred, matching_lists, unmatched_gt, unmatched_pred):

    gt, pred = np.array(gt).astype(bool), np.array(pred).astype(bool)
    matched = []

    for i, matching_list in enumerate(matching_lists):
        if matching_list:
            matched.append([gt[i], [pred[j] for j in matching_list]])

    return matched, [gt[i] for i in unmatched_gt], [pred[i] for i in unmatched_pred]


def collect_instance_from_matched_indexes_gt_and_pred(gt, pred, matching_lists, pred_matching_lists, unmatched_gt, unmatched_pred):
    gt, pred = np.array(gt).astype(bool), np.array(pred).astype(bool)
    matched = []
    pred_matched = []

    for i, matching_list in enumerate(matching_lists):
        if len(matching_list) > 0:
            matched.append([gt[i], [pred[j] for j in matching_list]])

    for i, matching_list in enumerate(pred_matching_lists):
        if len(matching_list) > 0:
            pred_matched.append([pred[i], [gt[j] for j in matching_list]])

    return matched, pred_matched, [gt[i] for i in unmatched_gt], [pred[i] for i in unmatched_pred]


def get_matching(gt, pred, metric, metric_ths):
    """
    Возвращает list (длины len(metric_ths)) из мэтчингов [GT, [PRED]], [PRED, [GT]], unmatched GT, unmatched PRED

    :param gt:
    :param pred:
    :param metric:
    :param metric_ths:
    :return:
    """
    res = []
    gt, pred = np.array(gt).astype(bool), np.array(pred).astype(bool)

    if (len(gt) == 0) or (len(pred) == 0):
        for metric_th in metric_ths:
            res.append([[], [], gt, pred])

        return res

    metric_matrix = get_metric_matrix(gt, pred, metric)

    for metric_th in metric_ths:
        matching_lists, unmatched_gt, unmatched_pred = get_matched_components(metric_matrix, metric_th)
        pred_matching_lists, pred_unmatched_pred, pred_unmatched_gt = get_matched_components(metric_matrix.T, metric_th)

        # TODO: remove it if generalize
        assert len(set(unmatched_gt) ^ set(pred_unmatched_gt)) == 0
        assert len(set(unmatched_pred) ^ set(pred_unmatched_pred)) == 0

        res.append(collect_instance_from_matched_indexes_gt_and_pred(gt, pred, matching_lists, pred_matching_lists, unmatched_gt, unmatched_pred))

    return res
