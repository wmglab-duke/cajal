import matplotlib.pyplot as plt

from cajal.nrn import SimulationEnvironment
from cajal.nrn.cells import MRG
from cajal.nrn.stimuli import MonophasicPulse, SineWave
from cajal.nrn.sources import IsotropicPoint
from cajal.nrn.monitors import StateMonitor
from cajal.units import mm, um, ohm, cm, mA, ms, Hz, kHz

# axon
mrg = MRG(diameter=7 * um, length=75 * mm)

# extracellular current source
pointsource = IsotropicPoint(x=0, y=0, z=500 * um, rhoe=500 * ohm * cm)
stim = SineWave(amp=1*mA, freq=2*kHz, delay=0.5 * ms)
electrode = pointsource << stim

# intracellular stim
intra = mrg.node[5] << MonophasicPulse(amp=1, pw=0.1 * ms, delay=0.5 * ms).repeat(100*Hz)

# recording with intra only
v_rec_intra_only = StateMonitor(mrg.node[-10], "v")

# simulation environment
env = SimulationEnvironment(
    axons=[mrg], 
    intra_stim=[intra], 
    monitors=[v_rec_intra_only]
)

# run
env.longrun(100 * ms, chunksize=1*ms, early_stopping=False)

# recording with intra & extra
v_rec_intra_and_extra = StateMonitor(mrg.node[-10], "v")

# simulation environment
env = SimulationEnvironment(
    axons=[mrg], 
    intra_stim=[intra], 
    extra_stim=[electrode], 
    monitors=[v_rec_intra_and_extra]
)

# run
env.longrun(100 * ms, chunksize=1*ms, early_stopping=False)

# visualize Vm at recorded node
plt.plot(v_rec_intra_only.cached['t'], v_rec_intra_only.cached['v'][0, :], label="intra only", color='blue')
plt.plot(v_rec_intra_and_extra.cached['t'], v_rec_intra_and_extra.cached['v'][0, :], label="intra + extra", color='red', linestyle='--')
plt.legend()
plt.show()