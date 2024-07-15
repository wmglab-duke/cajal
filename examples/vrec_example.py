import matplotlib.pyplot as plt

from cajal.nrn import SimulationEnvironment
from cajal.nrn.cells import MRG
from cajal.nrn.stimuli import MonophasicPulse
from cajal.nrn.sources import IsotropicPoint
from cajal.nrn.monitors import StateMonitor
from cajal.units import mm, um, ohm, cm, mA, ms

# axon
mrg = MRG(diameter=7 * um, length=75 * mm)

# extracellular current source
pointsource = IsotropicPoint(x=0, y=0, z=500 * um, rhoe=500 * ohm * cm)
stim = MonophasicPulse(amp=-0.1 * mA, pw=0.1 * ms, delay=0.5 * ms)
electrode = pointsource << stim

# recording
v_rec = StateMonitor(mrg.node, "v")

# simulation environment
env = SimulationEnvironment(axons=[mrg], extra_stim=[electrode], monitors=[v_rec])

# run
env.run(10 * ms, early_stopping=False)

# visualize Vm at different nodes along the axon
for i in [51, 40, 30, 20, 10]:
    plt.plot(v_rec.t, v_rec.v[i, :], label=f"node {i}")
plt.legend()
plt.show()
