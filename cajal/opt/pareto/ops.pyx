# distutils: language = c++
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
import warnings

from cajal.mpi.random import RNG

ctypedef fused generic:
    int
    float
    double
    long long

def crowding(np.ndarray[generic, ndim=2] scores):
    cdef Py_ssize_t popsize = scores.shape[0]
    cdef Py_ssize_t n_scores = scores.shape[1]

    # create crowding matrix of population (row) and score (column)
    cdef np.ndarray[double, ndim=2] crowding_matrix = np.empty((popsize, n_scores))
    cdef np.ndarray[double, ndim=1] crowd, sorted_scores, sorted_crowding
    cdef np.ndarray[np.int64_t, ndim=1] sorted_sores_index, re_sort_order

    # normalise scores (ptp is max-min)
    cdef np.ndarray[double, ndim=2] normed_scores

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            normed_scores = ((scores - np.min(scores, axis=0)) /
                             np.ptp(scores, axis=0)).astype('d')
        except Warning as e:
            raise StopIteration('Population converged.') from e

    # calculate crowding distance for each score in turn
    for col in range(n_scores):
        crowd = np.zeros(popsize)

        # end points have maximum crowding
        crowd[0] = 1
        crowd[popsize - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])
        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowd[1:popsize - 1] = \
            (sorted_scores[2:popsize] -
             sorted_scores[0:popsize - 2])

        # resort to original order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowd[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1).astype('d')
    return crowding_distances


def tournament_crowding(np.ndarray[generic, ndim=2] scores, Py_ssize_t number_to_select):

    cdef Py_ssize_t population_size = scores.shape[0]
    cdef vector[int] population_ids = range(population_size)
    cdef vector[double] crowding_distances = crowding(scores)
    cdef np.ndarray[np.int64_t, ndim=1] picked_population_ids = np.zeros(number_to_select, dtype=np.int)

    cdef Py_ssize_t i, f1id, f2id

    for i in range(number_to_select):
        population_size = population_ids.size()

        f1id = RNG.randint(0, population_size)
        f2id = RNG.randint(0, population_size)

        # If fighter # 1 is better
        if crowding_distances[f1id] >= crowding_distances[f2id]:
            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[f1id]
            # remove selected solution from available solutions
            population_ids.erase(population_ids.begin() + f1id)
            crowding_distances.erase(crowding_distances.begin() + f1id)

        else:
            picked_population_ids[i] = population_ids[f2id]
            # remove selected solution from available solutions
            population_ids.erase(population_ids.begin() + f2id)
            crowding_distances.erase(crowding_distances.begin() + f2id)

    return picked_population_ids
