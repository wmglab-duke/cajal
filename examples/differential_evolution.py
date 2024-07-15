import argparse
import datetime

import numpy as np
from tqdm import tqdm

from cajal.mpi import NeuronModel, Backend as MPI
from cajal.nrn import MRG, Backend as N
from cajal.nrn.sources import PreComputedInterpolate1D
from cajal.nrn.specs import OptNumerical as Opt
from cajal.nrn.stimuli import SymmetricBiphasic
from cajal.opt.differentialevolution import DENEURON
from cajal.opt.differentialevolution.callbacks import Logger, EarlyStopping, Timer
from cajal.opt.loss import PredictionLoss


parser = argparse.ArgumentParser()
parser.add_argument("--sample")
args = parser.parse_args()

sample = args.sample

N.tstop = 5
nc = 6

nodes = 101

fields = [np.load(f"./fields/{sample}/{i}.npy") for i in range(nc)]
fiber_zs = np.load(f"./fields/{sample}/fiber_z.npy")
target = np.load(f"./fields/{sample}/target.npy")
weights = np.load(f"./fields/{sample}/weights.npy")

midpoint = fiber_zs.min() + (np.ptp(fiber_zs) / 2)


class MyMRG(MRG):
    def init_AP_monitors(self):
        self.set_AP_monitors(axonnodes=[10, 90])


class WeightedBinaryCrossEntropy(PredictionLoss):
    def __init__(self, target, weights):
        super().__init__(target)
        self.weights = weights

    def loss(self, target, predicted):
        target = np.asarray(target)
        predicted = np.asarray(predicted).flatten()
        term_0 = (1 - target) * np.log(1 - predicted + 1e-7)
        term_1 = target * np.log(predicted + 1e-7)
        return -np.average((term_0 + term_1), axis=0, weights=self.weights)


def run():
    # Define model components:
    time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S")

    # axons
    axons = []
    for i in range(fields[0].shape[0]):
        axons.append(
            MyMRG.SPEC(
                y=midpoint,
                gid=i,
                enforce_odd_axonnodes=True,
                axonnodes=nodes,
                diameter=5.7,
                try_exact=False,
                piecewise_geom_interp=False,
                interpolation_method=1,
                passive_end_nodes=False,
            )
        )

    # stim
    extra = [
        PreComputedInterpolate1D.SPEC(
            field,
            fiber_zs,
            in_memory=True,
            method="linear",
            fill_value="point_source",
            truncate=0.2,
        )
        << SymmetricBiphasic.SPEC(amp=Opt([(-0.3, 0.3)]), pw=0.4, delay=0.5)
        for field in fields
    ]

    # build model
    nrn_model = NeuronModel(axons, extra, load_balancing="dynamic")

    # define loss, callbacks, build optimizer and run
    savedir = f"./results/sample_{sample}_{time}/"
    timer = Timer()
    callbacks = [Logger(savedir), EarlyStopping(min_loss=1e-7), timer]
    loss = WeightedBinaryCrossEntropy(target, weights)
    de = DENEURON(
        nrn_model,
        loss,
        updating="deferred",
        maxiter=200,
        popsize=100,
        strategy="best1bin",
        parameter_adaptation=0.1,
        mutation=0.8,
        recombination=1.0,
        boundary_constraint="resample",
        callbacks=callbacks,
    )
    hist = de.solve()

    if MPI.MASTER():
        tqdm.write(f"\n{hist.message}")
        hist.save(savedir + "history.hist")
        with open(savedir + "time.txt", "w") as f:
            f.write(timer.total_time_str)


if __name__ == "__main__":
    run()
