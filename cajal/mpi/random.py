from mpi4py import MPI
import numpy as np

from cajal.mpi.__backend import Backend as MPISRC


class MPIRNG(np.random.Generator):
    def __init__(self, seed=None):
        if MPISRC.ENABLED:
            if seed is None:
                seed = np.empty(1, dtype=np.uint32)
                if MPISRC.MASTER():
                    seed = np.random.SeedSequence().generate_state(1)
                MPISRC.COMM.Bcast([seed, MPI.UINT32_T], root=MPISRC.MASTER_RANK)
            super().__init__(np.random.PCG64(seed))
        else:
            super().__init__(np.random.PCG64(seed))

    def synchronize(self):
        seed = np.empty(1, dtype=np.uint32)
        if MPISRC.MASTER():
            seed = np.random.SeedSequence().generate_state(1)
        MPISRC.COMM.Bcast([seed, MPI.UINT32_T], root=MPISRC.MASTER_RANK)
        self.__init__(seed)

    def seed(self, seed):
        self.__init__(seed)


RNG = MPIRNG()
