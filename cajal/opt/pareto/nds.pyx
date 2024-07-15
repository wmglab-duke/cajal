# distutils: language = c++
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

ctypedef np.uint8_t uint8

ctypedef fused generic:
    int
    float
    double
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint c_dom1d(generic[:] arr1, generic[:] arr2):
    cdef bint is_l_eq = False
    cdef int i
    for i in range(arr1.shape[0]):
        if arr2[i] < arr1[i]:
            return False
        is_l_eq |= (arr1[i] < arr2[i])
    return is_l_eq

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[uint8, ndim=1] c_dominates_2d(generic[:, :] arr1, generic[:, :] arr2):
    cdef Py_ssize_t i = arr1.shape[0]
    cdef Py_ssize_t j = arr1.shape[1]

    cdef bint is_one_value_less = False
    cdef bint is_all_values_less_or_eq = True

    cdef np.ndarray[uint8, ndim=1] out = np.zeros(i, dtype=bool)

    cdef Py_ssize_t x, y
    for x in range(i):
        for y in range(j):
            if arr2[x, y] < arr1[x, y]:
                is_all_values_less_or_eq = False
                break
            is_one_value_less |= (arr1[x, y] < arr2[x, y])

        out[x] = is_one_value_less and is_all_values_less_or_eq
        is_one_value_less = False
        is_all_values_less_or_eq = True

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[vector[Py_ssize_t]] nds_(generic[:, :] arr):
    cdef Py_ssize_t n_pop = arr.shape[0]
    cdef Py_ssize_t n_obj = arr.shape[1]

    cdef bint is_dominated, i_is_l_eq, j_is_l_eq, j_dominates_i, i_dominates_j
    cdef vector[vector[Py_ssize_t]] fronts

    cdef np.ndarray[np.int64_t, ndim=1] ranks = np.zeros(n_pop, dtype=np.int64)

    cdef Py_ssize_t i, j, k, rank, n_sorted
    cdef vector[Py_ssize_t] cur_front

    rank = 1
    n_sorted = 0

    while n_sorted < n_pop:
        cur_front.clear()
        for i in range(n_pop):
            if ranks[i] > 0:
                continue
            is_dominated = False
            j = 0
            while j < cur_front.size():
                # check if j dominates i
                j_dominates_i = False
                j_is_l_eq = False
                for k in range(n_obj):
                    if arr[i, k] < arr[cur_front[j], k]:
                        j_dominates_i = False
                        break
                    j_is_l_eq |= (arr[cur_front[j], k] < arr[i, k])
                else:
                    j_dominates_i = j_is_l_eq
                    if j_dominates_i:
                        is_dominated = True
                        break
                if not j_dominates_i:
                    # check if i dominates j
                    i_dominates_j = False
                    i_is_l_eq = False
                    for k in range(n_obj):
                        if arr[cur_front[j], k] < arr[i, k]:
                            i_dominates_j = False
                            break
                        i_is_l_eq |= (arr[i, k] < arr[cur_front[j], k])
                    else:
                        i_dominates_j = i_is_l_eq
                        if i_dominates_j:
                            cur_front.erase(cur_front.begin() + j)
                            j -= 1
                j += 1
            if not is_dominated:
                cur_front.push_back(i)

        for k in range(cur_front.size()):
            ranks[cur_front[k]] = rank

        fronts.push_back(cur_front)
        n_sorted += cur_front.size()

        rank += 1

    return fronts

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uint8 not_dominated_1d(generic[:, :] x, Py_ssize_t i, Py_ssize_t n, Py_ssize_t width):
    cdef Py_ssize_t j
    for j in range(n):
        if c_dom1d(x[j], x[i]):
            return False
    return True

def dominates_2d(generic[:, :] array_1, generic[:, :] array_2):
    return c_dominates_2d(array_1, array_2)

def dominates_1d(generic[:] array_1, generic[:] array_2):
    return c_dom1d(array_1, array_2)

def dominates(arr1, arr2):
    assert arr1.shape == arr2.shape
    if arr1.ndim == 2:
        return dominates_2d(arr1, arr2)
    return dominates_1d(arr1, arr2)

def nds(generic[:, :] arr):
    return nds_(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[uint8, ndim=1] not_dominated(generic[:, :] arr):
    cdef Py_ssize_t n_pop = arr.shape[0]
    cdef Py_ssize_t n_obj = arr.shape[1]
    cdef Py_ssize_t i, j, k
    cdef vector[int] cur_front
    cdef bint is_dominated, i_is_l_eq, j_is_l_eq, j_dominates_i, i_dominates_j

    cdef np.ndarray[uint8, ndim=1] ranks = np.zeros(n_pop, dtype=bool)

    for i in range(n_pop):
        if ranks[i] != 0:
            continue
        is_dominated = False
        j = 0
        while j < cur_front.size():
            # check if j dominates i
            j_dominates_i = False
            j_is_l_eq = False
            for k in range(n_obj):
                if arr[i, k] < arr[cur_front[j], k]:
                    j_dominates_i = False
                    break
                j_is_l_eq |= (arr[cur_front[j], k] < arr[i, k])
            else:
                j_dominates_i = j_is_l_eq
                if j_dominates_i:
                    is_dominated = True
                    break
            if not j_dominates_i:
                # check if i dominates j
                i_dominates_j = False
                i_is_l_eq = False
                for k in range(n_obj):
                    if arr[cur_front[j], k] < arr[i, k]:
                        i_dominates_j = False
                        break
                    i_is_l_eq |= (arr[i, k] < arr[cur_front[j], k])
                else:
                    i_dominates_j = i_is_l_eq
                    if i_dominates_j:
                        cur_front.erase(cur_front.begin() + j)
                        j -= 1
            j += 1
        if not is_dominated:
            cur_front.push_back(i)

    for i in range(cur_front.size()):
        ranks[cur_front[i]] = 1

    return ranks