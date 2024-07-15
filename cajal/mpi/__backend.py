from mpi4py import MPI


class ContextSwitcher:
    def __enter__(self):
        Backend.ENABLED = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Backend.ENABLED = Backend.SIZE > 1


class Uninstantiable:
    """Prevent instantiation of Class."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("Class {} may not be instantiated.".format(cls))


# pylint: disable=no-self-argument
class Backend(Uninstantiable):

    """MPI Backend (Uninstantiable)

    Attributes
    -------
    COMM
        MPI.COMM_WORLD
    RANK
        MPI rank (int)
    SIZE
        MPI world size (int)
    """

    COMM = MPI.COMM_WORLD
    RANK = COMM.rank
    SIZE = COMM.size
    MASTER_RANK = 0
    ENABLED = SIZE > 1
    RANK_INDEPENDENCE = ContextSwitcher()

    @classmethod
    def MASTER_BARRIER(cls):
        cls.COMM.barrier()
        return cls.MASTER()

    @classmethod
    def barrier(cls):
        return cls.COMM.barrier()

    @classmethod
    def set_master_rank(cls, rank):
        cls.MASTER_RANK = rank
        cls.COMM.barrier()

    @classmethod
    def MASTER(cls):
        cls.COMM.barrier()
        return cls.RANK == cls.MASTER_RANK
