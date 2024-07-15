"""Useful numerical and array operations."""

import numpy as np


def round_nearest(x, a):
    """Round a number to the nearest decimal.

    Parameters
    ----------
    x : float
        Number to round.
    a : float
        Decimal to which to round.

    Returns
    -------
    float
        Rounded number.
    """
    if a is not None:
        if np.size(a) > 1:
            return np.array([round_nearest(x, a) for x, a in zip(x, a)])
        return np.round(
            np.round(np.divide(x, a)) * a, -(np.floor(np.log10(a))).astype(int)
        )
    return x


def get_nearest_indices(array, values):
    """Get index indices in array nearest values."""

    array = np.array(array)
    idxs = np.searchsorted(array, values, side="left")

    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return idxs


def stack_padding(it):
    """Stack arrays and pad arrays that are too short."""

    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    # find longest row length
    row_length = max(it, key=len).__len__()
    mat = np.array([resize(row, row_length) for row in it])

    return mat


def last_nonzero(arr, axis, invalid_val=-1):
    """Get indices of last non-zero in array along given axis."""
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def first_nonzero(arr, axis, invalid_val=-1):
    """Get indices of first non-zero in array along given axis."""
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def cartesian_product(*arrays):
    return np.dstack(np.meshgrid(*arrays, indexing="ij")).reshape(-1, len(arrays))
