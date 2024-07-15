"""
Metric functions.
"""

import numpy as np


def weighted_difference(target, predicted, smooth=1):
    """Custom loss metric that takes into consideration total proportion of
    correctly activated axons and total proportion of incorrectly activated
    axons.

    Parameters
    ----------
    target : numpy.ndarray
        1D boolean numpy array of target activation values.
    predicted : numpy.ndarray
        1D boolean unmpy array of calculated activation values.
    smooth : int, optional
        Smoothing factor to prevent loss going to 0 in sub-optimal
        situations, by default 1

    Returns
    -------
    float
        loss
    """

    intersection = np.sum(np.abs(target * predicted))
    intersection_ = intersection / np.sum(target)
    sum_ = np.sum(np.abs(target) + np.abs(predicted))
    scale_sum = (sum_ - intersection - np.sum(np.abs(target))) / (
        target.shape[0] - np.sum(np.abs(target))
    )

    loss = (((1 - intersection_) + smooth) * (scale_sum + smooth)) - smooth**2

    return loss


def hamming(target, predicted):
    """Hamming Distance.

    Parameters
    ----------
    target : numpy.ndarray
        1D boolean numpy array of target activation values.
    predicted : numpy.ndarray
        1D boolean numpy array of calculated acivation values.

    Returns
    -------
    float
        Hamming Distance
    """
    return np.count_nonzero(target != predicted)


def smc(target, predicted):
    """Simple Matching Coefficient.

    Parameters
    ----------
    target : numpy.ndarray
        1D boolean numpy array of target activation values.
    predicted : numpy.ndarray
        1D boolean numpy array of calculated acivation values.

    Returns
    -------
    float
        Simple Matching Coefficient.
    """
    assert target.shape[0] == predicted.shape[0]

    return 1 - (hamming(target, predicted) / target.shape[0])


def jaccard(target, predicted):
    """Jaccard Similarity Index.

    Parameters
    ----------
    target : numpy.ndarray
        1D boolean numpy array of target activation values.
    predicted : numpy.ndarray
        1D boolean numpy array of calculated activation values.

    Returns
    -------
    float
        Jaccard Difference Index.
    """
    x = np.asarray(target, np.bool)
    y = np.asarray(predicted, np.bool)

    num = np.double(np.bitwise_and(x, y).sum(axis=-1))
    den = np.double(np.bitwise_or(x, y).sum(axis=-1))

    return 1 - (num / den)


def activation_metrics(target, predicted):
    """Calculates what proportion of modelled axons are correctly / incorrectly
    activated by stimulus.

    Parameters
    ----------
    target : numpy.ndarray
        1D boolean numpy array of target activation values.
    predicted : numpy.ndarray
        1D boolean numpy array of calculated activation values.

    Returns
    -------
    float, float
        Correcly activated fraction of axons, incorrectly activated
        fraction of axons.
    """
    intersection = np.sum(np.abs(target * predicted))
    correct = intersection / np.sum(np.abs(target))
    sum_ = np.sum(np.abs(target) + np.abs(predicted))
    incorrect = (sum_ - intersection - np.sum(np.abs(target))) / (
        target.shape[0] - np.sum(np.abs(target))
    )

    return correct, incorrect
