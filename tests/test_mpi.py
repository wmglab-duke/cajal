import pytest

import numpy as np

from cajal.mpi import RNG, Backend as MPI


@pytest.mark.mpi
def test_size():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size > 0


@pytest.mark.mpi
def test_rng_parity():
    rand = RNG.random(3)
    all_rand = MPI.COMM.allgather(rand)
    assert np.array_equal(*all_rand)


@pytest.mark.mpi
def test_rng_synch():
    errors = []

    rand = None
    for _ in range(MPI.RANK + 1):
        rand = RNG.random(3)
    all_rand = MPI.COMM.allgather(rand)
    if np.array_equal(*all_rand):
        errors.append("Random numbers aren't different after variable # of calls.")

    RNG.synchronize()
    rand = RNG.random(3)
    all_rand = MPI.COMM.allgather(rand)
    if not np.array_equal(*all_rand):
        errors.append("RNG unsynchronized.")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
