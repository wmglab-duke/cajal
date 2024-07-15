import matplotlib.pyplot as plt
import numpy as np

from cajal.mpi import NeuronModel, Backend as MPI
from cajal.nrn.cells import MRG
from cajal.nrn.stimuli import MonophasicPulse
from cajal.nrn.sources import IsotropicPoint
from cajal.nrn.specs import Mutable
from cajal.units import mm, um, ohm, cm, mA, ms


# construct model specification
mrg = MRG.SPEC(diameter=7 * um, length=75 * mm)

pointsource = IsotropicPoint.SPEC(x=0, y=0, z=Mutable(), rhoe=500 * ohm * cm)
stim = MonophasicPulse.SPEC(amp=-0.1 * mA, pw=0.1 * ms, delay=0.5 * ms)
electrode = pointsource << stim


# subclass NeuronModel to specify what we want to record
class ParallelVRec(NeuronModel):
    def recording(self, axons):
        self.record("v", "v", [a.middle(1, "node") for a in axons], 1)


# lets analyze how distance of the current source affects the axonal
#        response
# look at 11 distances from 100 to 2000 um
n_dist = 11
distances = np.linspace(100, 2000, n_dist)

# instantiate the NeuronModel
env = ParallelVRec([mrg], [electrode])

# now run
env.run(distances)

# only visualize on one process
if MPI.MASTER_BARRIER():
    for i in range(n_dist):
        plt.plot(env.v[i, 0, 0, :], label=f"{distances[i]} um")
    plt.legend()
    plt.show()
