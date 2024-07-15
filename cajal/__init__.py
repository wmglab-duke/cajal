"""`cajal` is everything."""

__author__ = "Minhaj Hussain"
__email__ = "mah148@duke.edu"

__all__ = ["mpi", "nrn", "opt", "common"]


import cajal.mpi as mpi
import cajal.nrn as nrn
import cajal.opt as opt
import cajal.common as common

import pathlib
import shutil
import subprocess
from mpi4py import MPI

import neuron

COMM = MPI.COMM_WORLD

if not neuron.load_mechanisms(
    str(pathlib.Path(__file__).parent.joinpath("mod").resolve()),
    warn_if_already_loaded=False,
):
    COMM.barrier()
    if COMM.rank == 0:
        print("Compiling NEURON mechanisms (Only on first load)")
        n = subprocess.Popen(
            [shutil.which("nrnivmodl")],
            cwd=pathlib.Path(__file__).parent.joinpath("mod"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        n.wait()
    COMM.barrier()
    neuron.load_mechanisms(str(pathlib.Path(__file__).parent.joinpath("mod").resolve()))
    COMM.barrier()
    if COMM.rank == 0:
        print("NEURON mechanisms loaded")
