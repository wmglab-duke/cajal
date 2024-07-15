import matplotlib.pyplot as plt
import numpy as np

# -- NOTE: extra imports --
from cajal.mpi import Backend as MPI, Thresholder
from cajal.nrn.specs import Mutable as Mut

from cajal.nrn import Backend as N
from cajal.nrn.cells import MRG
from cajal.nrn.stimuli import MonophasicPulse
from cajal.nrn.sources import IsotropicPoint
from cajal.units import mm, um, ohm, cm, mA, ms

N.tstop = 5 * ms
N.dt = 0.005 * ms

# axon
mrg = MRG.SPEC(diameter=5.7 * um, axonnodes=71)

# extracellular current source
pointsource = IsotropicPoint.SPEC(x=0, y=0, z=0.5 * mm, rhoe=500 * ohm * cm)
stim = MonophasicPulse.SPEC(amp=-1 * mA, pw=Mut(), delay=0.5 * ms)
electrode = pointsource << stim

# use Thresholder for parallel thresholds
env = Thresholder([mrg], [electrode])

# range of pws
pws = np.linspace(0.1, 1.0, 21)

# calculate thresholds
env.run(pws)

if MPI.MASTER():
    plt.plot(pws, env.thresholds)
    plt.xlabel("PW (ms)")
    plt.ylabel("activation threshold (mA)")
    plt.show()
