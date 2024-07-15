import sys

from mpi4py import MPI
from decorator import decorator

from cajal.mpi import Backend as MPISRC


@decorator
def master_only(f, *args, **kwargs):
    if MPISRC.MASTER():
        return f(*args, **kwargs)


def pprint(str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == MPISRC.MASTER_RANK:
        print(str, end=end)


def uprint(*objects, sep=" ", end="\n", file=sys.stdout):
    enc = file.encoding
    if enc == "UTF-8":
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors="backslashreplace").decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)
