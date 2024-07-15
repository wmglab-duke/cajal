import numpy as np

from cajal.mpi import MPIRunner, MPIModelSpec, NeuronModel
from cajal.opt.differentialevolution._differentialevolution import DEBASE
from cajal.opt.loss import OutputLoss, MultiObjective


class DEPNS(DEBASE):
    """
    Differential Evolution class that integrates modelling neural
    tissue with NEURON / a custom approximation. This class subclasses
    DEBASE (for DE logic) and requires an instance of either MPIRunner,
    MPIModelSpec or NeuronModel as its MPI handler.
    """

    _dispatch = {
        "output_only": "_output_only_dispatch",
        "with_vector": "_with_vector_dispatch",
        "all": "_full_dispatch",
    }

    def __init__(
        self,
        mpi,
        loss,
        bounds=None,
        maxiter=50,
        callbacks=None,
        updating="deferred",
        hybrid_batch_size=None,
        strategy="best1bin",
        mutation=(0.5, 1),
        recombination=0.7,
        init="rlhs",
        popsize=100,
        objectives=1,
        boundary_constraint="reinitialise",
        resample_limit=10,
        p_best=0.15,
        archive_size=0,
        parameter_adaptation=None,
        seed=None,
    ):
        if not isinstance(mpi, (MPIRunner, MPIModelSpec, NeuronModel)):
            raise TypeError(
                "The model passed to DEPNS must be "
                "of type MPIModelSpec or NeuronModel"
            )
        self.loss = loss

        # -- establish loss function form --
        if isinstance(loss, OutputLoss):
            self.loss_func = getattr(self, self._dispatch["output_only"])
        elif "parameter_args" in loss.required:
            self.loss_func = getattr(self, self._dispatch["all"])
        else:
            self.loss_func = getattr(self, self._dispatch["with_vector"])

        # -- establish whether multiobjective problem --
        if isinstance(loss, MultiObjective):
            objectives = loss.ndim

        # -- establish bounds --
        if bounds is None:
            bounds = mpi.bounds()

        DEBASE.__init__(
            self,
            mpi,
            bounds,
            maxiter,
            callbacks,
            updating,
            hybrid_batch_size,
            strategy,
            mutation,
            recombination,
            init,
            popsize,
            objectives,
            boundary_constraint,
            resample_limit,
            p_best,
            archive_size,
            parameter_adaptation,
            seed,
        )

    def _output_only_dispatch(self, *args):
        return self.loss(args[0])

    def _with_vector_dispatch(self, output, pop_vec):
        return self.loss(output=output, pop_vec=pop_vec)

    def _full_dispatch(self, output, pop_vec):
        parameter_args = self.mpi.parameters_dict(pop_vec)
        return self.loss(output=output, pop_vec=pop_vec, parameter_args=parameter_args)

    def logs(self):
        x = self.x()
        logs = {
            "population": self._scale_parameters(self.population),
            "x": np.copy(x[0]),
            "energies": np.copy(self.population_energies),
            "model_outputs": {k: np.copy(v) for k, v in self.__outputs__.items()},
            "loss": np.copy(x[1]),
        }
        return logs


class DENEURON(DEPNS):
    def _set_verbosity(self, verbose):
        if verbose:
            self.mpi._persistent_pbar = False
            self.mpi.init_report_persistent(
                self.num_population_members * self.mpi.size * (self.maxiter + 1)
            )
        else:
            self.mpi._persistent_pbar = True
