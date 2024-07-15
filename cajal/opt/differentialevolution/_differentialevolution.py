# pylint: disable=too-many-lines
"""
Core Differential Evolution classes, based on the SciPy (Copyright (c) 2001-2002
Enthought, Inc.  2003-2019, SciPy Developers.) Differential Evolution implementation.
The Base DE implementation supports large-scale parallelisation on HPC clusters using
MPI via the MPIRunner interface (and its subclasses) in the mpi module.
"""
import math
from typing import Union

import numpy as np
from scipy.stats import cauchy

from cajal.mpi import Backend as MPISRC, MPIRunner, NeuronModel
from cajal.mpi.random import MPIRNG
from cajal.opt.differentialevolution.callbacks import configure_callbacks
from cajal.opt.pareto import nds, dominates, tournament_crowding


COMM = MPISRC.COMM

_status_message = {
    "success": "Optimization terminated successfully.",
    "maxfev": "Maximum number of function evaluations has " "been exceeded.",
    "maxiter": "Maximum number of iterations has been " "exceeded.",
    "pr_loss": "Desired error not necessarily achieved due " "to precision loss.",
    "nan": "NaN result encountered.",
    "out_of_bounds": "The result is outside of the provided " "bounds.",
}


class Population:  # pylint: disable=too-many-instance-attributes
    """Class that instantiates the population of candidate solutions
    that are evolved each generation.

    Parameters
    ----------
    bounds : np.ndarray
        Bounds for each parameter value.
    init : str, np.ndarray
        'latinhypercube', 'random', or an array of shape (M, N)
        where N is the number of parameters and M > 5.
    popsize : int
        The resulting population will have popsize * N (number of
        parameters) if using 'random' or 'latinhypercube' initialisation.
    outputs_shape : Iterable
        Shape of outputs from MPI model function.
    objectives : int
        Number of objectives.
    seed : int
        Seed for random number generator.
    """

    __init_error_msg = (
        "The population initialization method must be one of "
        "'latinhypercube' or 'random', or an array of shape "
        "(M, N) where N is the number of parameters and M>5"
    )

    def __init__(
        self,
        bounds,
        init,
        popsize,
        outputs_shape,
        outputs_dtype,
        outputs_aliases,
        objectives: int,
        seed=None,
    ):
        self.limits = np.array(bounds, dtype="f").T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
                "bounds should be a sequence containing "
                "real valued (min, max) pairs for each value "
                "in x"
            )

        self.rng = MPIRNG(seed)
        self.parameter_count = np.size(self.limits, 1)
        self.num_population_members = max(5, popsize)

        self.outputs_shape = outputs_shape
        self.outputs_dtype = outputs_dtype
        self.outputs_aliases = outputs_aliases
        self.__outputs__ = {}

        self.objectives = objectives
        self.multiobjective = objectives > 1

        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.population_shape = (self.num_population_members, self.parameter_count)
        self.energies_shape = (self.num_population_members, self.objectives)

        self.population = None
        self.population_energies = None
        self._nfev = None

        if isinstance(init, str):
            if init == "rlhs":
                self.init_population_rlhs()
            elif init == "mlhs":
                self.init_population_mlhs()
            elif init == "random":
                self.init_population_random()
            elif init == "slhs":
                self.init_population_slhs()
            elif init.startswith("chaos"):
                self.init_population_chaos(init)
            else:
                raise ValueError(self.__init_error_msg)
        elif isinstance(init, np.ndarray):
            self.init_population_array(init)
        else:
            self.init_population_ml(init)

    def init_population_rlhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.rng

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the
        # following size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (
            segsize * rng.random(self.population_shape)
            + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[
                :, np.newaxis
            ]
        )

        # Create an array for population of candidate solutions.
        self.population = np.empty(samples.shape, dtype="f")

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies and function output records
        self._init_energies()
        self._init_outputs()

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_mlhs(self):
        rng = self.rng
        segsize = 1.0 / self.num_population_members
        samples = (segsize * 0.5 * np.ones(self.population_shape)) + np.linspace(
            0.0, 1.0, self.num_population_members, endpoint=False
        )[:, np.newaxis]

        self.population = np.empty(samples.shape, dtype="f")

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies and function output records
        self._init_energies()
        self._init_outputs()

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_slhs(self):
        pass

    def init_population_chaos(self, init):
        rng = self.rng
        pop = rng.random(self.population_shape)
        for i in range(int(init.split(":")[-1])):
            pop[:] = np.sin(np.pi * pop)
        self.population = pop

        self._init_energies()
        self._init_outputs()

        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.rng
        self.population = rng.random(self.population_shape)

        # reset population energies and function output records
        self._init_energies()
        self._init_outputs()

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_ml(self, init):
        """
        Initialises the population using a machine learning model
        approximation.
        """

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The
            array should have shape (M, len(x)), where len(x) is the
            number of parameters. The population is clipped to the
            lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (
            np.size(popn, 0) < 5
            or popn.shape[1] != self.parameter_count
            or len(popn.shape) != 2
        ):
            raise ValueError(
                "The population supplied needs to have shape "
                "(M, len(x)), where M > 4 and x is the vector "
                "input to your function."
            )

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members, self.parameter_count)
        self.energies_shape = (self.num_population_members, self.objectives)

        # reset population energies
        self._init_energies()
        self._init_outputs()

        # reset number of function evaluations counter
        self._nfev = 0

    def _init_energies(self):
        self.population_energies = np.full(self.energies_shape, np.inf)

    def _init_outputs(self):
        aliases = self.outputs_aliases
        popsize = self.num_population_members
        for i, shape in enumerate(self.outputs_shape):
            try:
                dtype = np.dtype(self.outputs_dtype[i])
            except TypeError:
                dtype = np.dtype(self.outputs_dtype)
            var = f"out_{aliases[i]}" if aliases else f"out_{i}"
            setattr(self, var, np.empty((popsize, *shape), dtype=dtype))
            self.__outputs__[aliases[i] if aliases else var] = getattr(self, var)

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5
            d[np.isnan(d)] = 0
        return d

    def x(self):
        """
        The best solution from the solver
        """
        if self.multiobjective:
            par_front_inds = nds(self.population_energies)[0]
            pareto_best = self._scale_parameters(self.population[par_front_inds])
            best_losses = self.population_energies[par_front_inds]
            return pareto_best, best_losses

        pop_best = self._scale_parameters(self.population[0])
        best_loss = self.population_energies[0]
        return pop_best, best_loss

    def fronts(self):
        """
        Return population organised into pareto fronts. If the problem is not
        multiobjective, returns ranked population.
        """
        if self.multiobjective:
            nd_sort = nds(self.population_energies)
            pop = self._scale_parameters(self.population)
            eng = self.population_energies
            return {i: (pop[inds], eng[inds]) for i, inds in enumerate(nd_sort)}

        sort = np.argsort(self.population_energies)
        params = self._scale_parameters(self.population[sort])
        return params, self.population_energies[sort]


# noinspection PyArgumentList
class Mutator(Population):  # pylint: disable=too-many-instance-attributes
    """
    Population of candidates that can mutate itself. Subclasses
    Population.

    Parameters
    ----------
    strategy : str
        How mutants are generated. Valid options are:
            * 'best1bin'
            * 'randtobest1bin'
            * 'currenttobest1bin'
            * 'currenttopbestbin'
            * 'best2bin',
            * 'rand2bin'
            * 'rand1bin'
            * 'best1exp'
            * 'rand1exp'
            * 'randtobest1exp'
            * 'currenttobest1exp'
            * 'currenttopbestexp'
            * 'best2exp'
            * 'rand2exp'
    mutation : float, tuple, list, np.ndarray
        Mutation rate. If specified as a sequence, dithering
        is performed.
    recombination : float
        Value in [0, 1] - specifies likelihood of trial vector
        inheriting from parent.
    bounds : np.ndarray
        Bounds for each parameter value.
    init : str, np.ndarray
        'latinhypercube', 'random', or an array of shape (M, N)
        where N is the number of parameters and M > 5.
    popsize : int
        The resulting population will have popsize * N (number
        of parameters) if using 'random' or 'latinhypercube'
        initialisation.
    objectives : int
        Number of objectives.
    p_best : float
        Percentage of top solutions to select from in
        'currenttopbest' mutation strategy.
    archive_size : int (optional)
        Size of the optional archive.
    parameter_adaptation : float (optional)
        Value of 'c' in Jingqiao Zhang, Sanderson, A.C., 2009. JADE:
        Adaptive Differential Evolution With Optional External Archive.
        IEEE Transactions on Evolutionary Computation 13, 945–958.
        https://doi.org/10.1109/TEVC.2009.2014613. Used when performing
        parameter adaptation. Zhang et al. use 0.1.
    seed : int (optional)
        Seed for random number generator. By default None.
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {
        "best1bin": "_best1",
        "pbest1bin": "_pbest1",
        "randtobest1bin": "_randtobest1",
        "currenttobest1bin": "_currenttobest1",
        "currenttopbestbin": "_currenttopbest",
        "currentrandtobest1bin": "_currentrandtobest1",
        "best2bin": "_best2",
        "rand2bin": "_rand2",
        "rand1bin": "_rand1",
    }
    _exponential = {
        "best1exp": "_best1",
        "pbest1exp": "_pbest1exp",
        "rand1exp": "_rand1",
        "randtobest1exp": "_randtobest1",
        "currenttobest1exp": "_currenttobest1",
        "currenttopbestexp": "_currenttopbest",
        "currentrandtobest1exp": "_currentrandtobest1",
        "best2exp": "_best2",
        "rand2exp": "_rand2",
    }

    # Dispatch of boundary violation method
    _repair = {
        "reinitialise": "_reinitialise",
        "projection": "_projection",
        "reflection": "_reflection",
        "wrapping": "_wrapping",
    }
    _feasibility_preserving = {
        "resample": "_resample",
        "midpoint_target": "_midpoint_target",
        "rand_target": "_rand_target",
    }

    def __init__(
        self,
        strategy,
        mutation,
        recombination,
        bounds,
        boundary_constraint,
        resample_limit,
        init,
        popsize,
        outputs_shape,
        outputs_dtype,
        outputs_aliases,
        objectives,
        p_best,
        archive_size=0,
        parameter_adaptation=None,
        seed=None,
    ):
        super(Mutator, self).__init__(
            bounds,
            init,
            popsize,
            outputs_shape,
            outputs_dtype,
            outputs_aliases,
            objectives,
            seed,
        )

        # mutation functionality properties
        self.mutation_func = None
        self.mutation_strategy = None
        self.scale = None
        self.dither = None
        self.cross_over_probability = None
        self.p_best = p_best
        self.archive_active = archive_size > 0
        self.archive_size = archive_size
        self.archive = None

        # crossover functionality properties
        self.constraint_func = None
        self.boundary_constraint = boundary_constraint
        self.resample_limit = resample_limit

        # parameter adaptation properties
        self.parameter_adaptation = parameter_adaptation
        if self.parameter_adaptation:
            if hasattr(mutation, "__iter__"):
                self.mu_f = mutation[0]
            else:
                self.mu_f = mutation
            self.mu_cr = recombination
            self.mutation_vec = None
            self.cross_over_vec = None
            self.successful_f = []
            self.successful_cr = []

        # full initialisation
        self._init_mutation_strategy(strategy)
        self._init_mutation_rate(mutation)
        self._init_boundary_constraint_handling(boundary_constraint)
        self._init_recombination(recombination)
        self._init_archive()

    # ----- initialisation functions -----

    def _init_mutation_strategy(self, strategy):
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.mutation_strategy = strategy

    def _init_mutation_rate(self, mutation):
        if self.parameter_adaptation:
            self._regenerate_mutation_vec()
        else:
            # Mutation constant should be in [0, 2). If specified as a sequence
            # then dithering is performed.
            self.scale = mutation
            if (
                not np.all(np.isfinite(mutation))
                or np.any(np.array(mutation) >= 2)
                or np.any(np.array(mutation) < 0)
            ):
                raise ValueError(
                    "The mutation constant must be a float in "
                    "U[0, 2), or specified as a tuple(min, max)"
                    " where min < max and min, max are in "
                    "U[0, 2)."
                )

            if hasattr(mutation, "__iter__") and len(mutation) > 1:
                self.dither = [mutation[0], mutation[1]]
                self.dither.sort()

    def _init_recombination(self, recombination):
        if self.parameter_adaptation:
            self._regenerate_cross_over_vec()
        else:
            self.cross_over_probability = recombination

    def _init_boundary_constraint_handling(self, boundary_constraint):
        if boundary_constraint in self._repair:
            self.constraint_func = getattr(self, self._repair[boundary_constraint])
        elif boundary_constraint in self._feasibility_preserving:
            self.constraint_func = getattr(
                self, self._feasibility_preserving[boundary_constraint]
            )
        else:
            raise ValueError(
                "Please select a valid boundary violation handling strategy."
            )

    def _init_archive(self):
        if self.archive_active:
            self.archive = np.empty((0, self.parameter_count))

    # ----- parameter adaptation -----

    def _regenerate_mutation_vec(self):
        self.mutation_vec = np.clip(self._cauchy_mut, None, 1)
        while np.any(self.mutation_vec <= 0):
            resample = np.clip(self._cauchy_mut, None, 1)
            self.mutation_vec = np.where(
                self.mutation_vec <= 0, resample, self.mutation_vec
            )

    def _regenerate_cross_over_vec(self):
        self.cross_over_vec = np.clip(self._normal_cr, 0, 1)

    @property
    def _cauchy_mut(self):
        return cauchy.rvs(
            loc=self.mu_f,
            scale=0.1,
            size=self.num_population_members,
            random_state=self.rng,
        )

    @property
    def _normal_cr(self):
        return self.rng.normal(
            loc=self.mu_cr, scale=0.1, size=self.num_population_members
        )

    def _reset_successful(self):
        self.successful_f = []
        self.successful_cr = []

    def _update_mus(self):
        suc_f, suc_cr = np.asarray(self.successful_f), np.asarray(self.successful_cr)
        if suc_f.size > 0:
            c = self.parameter_adaptation
            lehmer = np.sum(np.square(suc_f) / np.sum(suc_f))
            self.mu_f = ((1 - c) * self.mu_f) + (c * lehmer)
            self.mu_cr = ((1 - c) * self.mu_cr) + (c * np.mean(suc_cr))

    # ----- boundary constraint violation -----

    def _ensure_constraint(self, trial, indices):
        if self.boundary_constraint in self._repair:
            self.constraint_func(trial)
        else:
            self.constraint_func(trial, np.array(indices, ndmin=1))

    def _reinitialise(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.rng.random(mask[0].size)

    @staticmethod
    def _projection(trial):
        greater = np.where(trial > 1)
        less = np.where(trial < 0)
        trial[greater] = 1
        trial[less] = 0

    @staticmethod
    def _reflection(trial):
        while ((trial > 1) | (trial < 0)).any():
            greater = np.where(trial > 1)
            less = np.where(trial < 0)
            trial[greater] = 2 - trial[greater]
            trial[less] *= -1

    @staticmethod
    def _wrapping(trial):
        while ((trial > 1) | (trial < 0)).any():
            greater = np.where(trial > 1)
            less = np.where(trial < 0)
            trial[greater] = trial[greater] - 1
            trial[less] = 1 + trial[less]

    def _resample(self, trial, inds):
        trial_copy = np.reshape(trial, (-1, self.parameter_count))
        for t, ind in zip(trial_copy, inds):
            resample_count = 0
            while ((t > 1) | (t < 0)).any():
                t[:] = self._mutate(ind)
                resample_count += 1
                if resample_count > self.resample_limit:
                    self._reinitialise(t)
                    break
        trial[:] = trial_copy

    def _rand_target(self, trial, inds):
        greater = np.where(trial > 1)
        less = np.where(trial < 0)
        trial[greater] = (
            self.rng.random(greater.size) * (1 - self.population[inds][greater])
            + self.population[inds][greater]
        )
        trial[less] = self.rng.random(less.size) * self.population[inds][less]

    def _midpoint_target(self, trial, inds):
        greater = np.where(trial > 1)
        less = np.where(trial < 0)
        trial[greater] = (1 + self.population[inds][greater]) / 2
        trial[less] = self.population[inds][less] / 2

    # ----- mutation functions -----

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""

        trial = np.copy(self.population[candidate])
        rng = self.rng
        fill_point = rng.integers(0, self.parameter_count)

        if self.parameter_adaptation:
            self.scale = self.mutation_vec[candidate]
            self.cross_over_probability = self.cross_over_vec[candidate]

        if self.mutation_strategy in [
            "currenttobest1exp",
            "currenttobest1bin",
            "currenttopbestexp",
            "currenttopbestbin",
            "currentrandtobest1bin",
            "currentrandtobest1exp",
        ]:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        # binomial crossover:
        if self.mutation_strategy in self._binomial:
            crossovers = rng.random(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        # else exponential:
        i = 0
        while i < self.parameter_count and rng.random() < self.cross_over_probability:
            trial[fill_point] = bprime[fill_point]
            fill_point = (fill_point + 1) % self.parameter_count
            i += 1

        return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = self.population[samples[:2]]
        return self.population[0] + self.scale * (r0 - r1)

    def _pbest1(self, samples):
        """pbest1bin, pbest1exp"""
        r0, r1 = self.population[samples[:2]]
        return self.population[self.p_best_ind()] + self.scale * (r0 - r1)

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = self.population[samples[:3]]
        return r0 + self.scale * (r1 - r2)

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = self.population[samples[:3]]
        bprime = np.copy(r0)
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (r1 - r2)
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = self.population[samples[:2]]
        if self.archive_active:
            r1 = self._samples_with_archive(1, [candidate, samples[0]])
        bprime = self.population[candidate] + self.scale * (
            self.population[0] - self.population[candidate] + r0 - r1
        )
        return bprime

    def _currenttopbest(self, candidate, samples):
        """currenttopbestbin, currenttopbestexp"""
        r0, r1 = self.population[samples[:2]]
        p = self.p_best_ind()
        if self.archive_active:
            r1 = self._samples_with_archive(1, [candidate, samples[0]])
        bprime = self.population[candidate] + self.scale * (
            self.population[p] - self.population[candidate] + r0 - r1
        )
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = self.population[samples[:4]]
        return self.population[0] + self.scale * (r0 + r1 - r2 - r3)

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = self.population[samples]
        return r0 + self.scale * (r1 + r2 - r3 - r4)

    def _currentrandtobest1(self, candidate, samples):
        r0, r1, r2 = self.population[samples[:3]]
        if self.archive_active:
            r2 = self._samples_with_archive(1, [candidate, samples[1]])
        bprime = self.population[candidate] + self.scale * (
            self.population[0] - r0 + r1 - r2
        )
        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(np.size(self.population, 0)))
        idxs.remove(candidate)
        self.rng.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def _samples_with_archive(self, number_samples, exclude):
        total = np.vstack([self.population, self.archive])
        idxs = list(range(np.size(total, 0)))
        return total[self.rng.choice(np.setdiff1d(idxs, exclude), number_samples)]

    # ----- population bests and promotion -----

    def best_ind(self):
        """Get index of 'best' member of population."""
        if self.multiobjective:
            l_par = nds(self.population_energies)[0]
            return self.rng.choice(l_par, 1)[0]
        return np.argmin(self.population_energies)

    def p_best_ind(self):
        """Get ind from best p% members of population."""
        if self.multiobjective:
            return self.best_ind()
        return self.rng.choice(
            np.argsort(self.population_energies.flatten())[
                : int(self.p_best * self.num_population_members)
            ]
        )

    def _promote_lowest_energy(self):
        # put the lowest energy into the best solution position.
        low = self.best_ind()
        self.population_energies[[0, low], :] = self.population_energies[[low, 0], :]
        self.population[[0, low], :] = self.population[[low, 0], :]
        for output in self.__outputs__.values():
            output[[0, low], :] = output[[low, 0], :]


class DEBASE(Mutator):
    """
    Base Differential Evolution class.
    """

    _adv_methods = {
        "immediate": "_immediate",
        "deferred": "_deferred",
        "hybrid": "_hybrid",
    }

    def __init__(
        self,
        mpi: Union[MPIRunner, NeuronModel],
        bounds,
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
        self.history = None
        self.callbacks = configure_callbacks(callbacks, self)

        self.maxiter = maxiter
        self._nit = 0

        self.mpi = mpi

        # -- get Mutator and Population ready to receive data --
        return_arguments = self.mpi.return_attrs()
        outputs_shape = []
        outputs_dtype = []
        for var in return_arguments:
            try:
                outputs_shape.append(self.mpi.results_shapes[var])
                outputs_dtype.append(self.mpi.results_dtypes[var])
            except KeyError:
                try:
                    outputs_shape.append(
                        (*self.mpi.record_dims[var], self.mpi.record_length)
                    )
                    outputs_dtype.append(self.mpi.record_dtypes[var])
                except KeyError as e:
                    raise AttributeError(
                        f"{var} is not a valid return attribute of the " f"MPI model."
                    ) from e

        Mutator.__init__(
            self,
            strategy,
            mutation,
            recombination,
            bounds,
            boundary_constraint,
            resample_limit,
            init,
            popsize,
            outputs_shape,
            outputs_dtype,
            return_arguments,
            objectives,
            p_best,
            archive_size,
            parameter_adaptation,
            seed,
        )

        # set the updating / parallelisation options
        self._update = getattr(self, self._adv_methods[updating])

        if updating == "hybrid":
            self.hybrid_batch_size = hybrid_batch_size
            self.num_hybrid_fragments = math.ceil(
                self.num_population_members / self.hybrid_batch_size
            )
            self.hybrid_indices = np.arange(self.num_population_members)

        if self.multiobjective:
            self.selection = getattr(self, "_select_{}_mo".format(updating))
        else:
            self.selection = getattr(self, "_select_{}_so".format(updating))

    @property
    def initialised(self):
        return not np.all(np.isinf(self.population_energies))

    def solve(self, verbose=1):
        """
        Runs the Differential Evolution Solver.
        Returns
        -------
        hist : History
            History callback object.
        """

        # do the optimisation.
        self.callbacks.on_solve_begin(self.logs())
        self._set_verbosity(verbose)
        warning_flag = True

        for nit in range(self._nit + 1, self._nit + self.maxiter + 1):
            self._nit = nit
            self._update_pbar()
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration as e:
                status_message = str(e)
                break
            except Exception as e:
                raise e

        else:
            status_message = _status_message["maxiter"]
            warning_flag = False

        info = {"message": status_message, "success": warning_flag is not True}
        self.callbacks.on_solve_end(dict(self.logs(), **info))

        return self.history

    def search(self, iters, verbose=1):
        if not self.initialised:
            self.callbacks.on_solve_begin(self.logs())
            self._set_verbosity(verbose)

        for nit in range(self._nit + 1, self._nit + iters + 1):
            self._nit = nit
            self._update_pbar()
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration as e:
                status_message = str(e)
                break
            except Exception as e:
                raise e

        else:
            status_message = _status_message["maxiter"]

        return self.history, status_message

    def end(self):
        info = {"message": "DE called end."}
        self.callbacks.on_solve_end(dict(self.logs(), **info))
        return self.history

    # noinspection PyArgumentList
    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies

        self._init()
        self.callbacks.on_generation_begin(self._nit, self.logs())

        if self.parameter_adaptation:
            self._regenerate_mutation_vec()
            self._regenerate_cross_over_vec()
        else:
            if self.dither is not None:
                self.scale = (
                    self.rng.random() * (self.dither[1] - self.dither[0])
                    + self.dither[0]
                )

        self._reset_successful()
        self._update()
        self._update_mus()
        self._truncate()
        self.callbacks.on_generation_end(self._nit, self.logs())

        return self.x()

    next = __next__

    # ----- immediate updating -----

    def _immediate(self):
        candidate_range = self.rng.permutation(np.size(self.population, 0))
        for candidate in candidate_range:
            trial = self._mutate(candidate)
            self._ensure_constraint(trial, candidate)
            scaled_trial = self._scale_parameters(trial)
            output = self.evaluate(scaled_trial)
            energy = self.loss_func(output, scaled_trial)
            self._nfev += 1

            self.selection(candidate, energy, trial, output)

    def _select_immediate_so(self, candidate, energy, trial, output):
        if energy < self.population_energies[candidate]:
            self.population[candidate] = trial
            self.population_energies[candidate] = energy
            for var, out in zip(self.__outputs__.values(), output):
                var[candidate, :] = out
            self._update_archive(candidate)

            if energy < self.population_energies[0]:
                self._promote_lowest_energy()

            if self.parameter_adaptation:
                self.successful_f.append(self.mutation_vec[candidate])
                self.successful_cr.append(self.cross_over_vec[candidate])

    def _select_immediate_mo(self, candidate, energy, trial, output):
        energy = energy.flatten()
        if dominates(energy, self.population_energies[candidate]):
            self.population[candidate] = trial
            self.population_energies[candidate] = energy
            for var, out in zip(self.__outputs__.values(), output):
                var[candidate, :] = out
            self._update_archive(candidate)

            if self.parameter_adaptation:
                self.successful_f.append(self.mutation_vec[candidate])
                self.successful_cr.append(self.cross_over_vec[candidate])

        elif not dominates(self.population_energies[candidate], energy):
            self.population = np.vstack([self.population, trial])
            self.population_energies = np.vstack([self.population_energies, energy])
            for (var, arr), out in zip(self.__outputs__.values(), output):
                setattr(self, f"out_{var}", np.vstack([arr, out]))
                self.__outputs__[var] = getattr(self, f"out_{var}")

        self._promote_lowest_energy()

    # ----- deferred updating -----

    def _deferred(self):
        trial_pop = np.vstack(
            [self._mutate(i) for i in range(self.num_population_members)]
        )
        self._ensure_constraint(trial_pop, np.arange(self.num_population_members))
        scaled_trial_pop = self._scale_parameters(trial_pop)
        trial_predictions = self.evaluate(scaled_trial_pop)
        trial_energies = np.vstack(
            [
                self.loss_func(out, p)
                for out, p in zip(zip(*trial_predictions), scaled_trial_pop)
            ]
        )

        self.selection(trial_energies, trial_pop, trial_predictions)
        self._promote_lowest_energy()

    def _deferred_update(self, loc, pop, energies, predictions):
        self.population = np.where(loc[:, np.newaxis], pop, self.population)
        self.population_energies = np.where(
            loc[:, np.newaxis], energies, self.population_energies
        )
        for arr, out in zip(self.__outputs__.values(), predictions):
            arr[:] = np.where(loc[:, np.newaxis], out, arr)
        self._update_archive(loc)

        if self.parameter_adaptation:
            self.successful_f = self.mutation_vec[loc]
            self.successful_cr = self.cross_over_vec[loc]

    def _select_deferred_so(self, energies: np.ndarray, pop, predictions):
        loc = (energies < self.population_energies).flatten()
        self._deferred_update(loc, pop, energies, predictions)

    def _select_deferred_mo(self, energies, pop, predictions):
        dominant = dominates(energies, self.population_energies)
        dominated = dominates(self.population_energies, energies)
        not_dom_not_dominated = np.logical_not(np.logical_or(dominant, dominated))
        self._deferred_update(dominant, pop, energies, predictions)

        self.population = np.vstack([self.population, pop[not_dom_not_dominated]])
        self.population_energies = np.vstack(
            [self.population_energies, energies[not_dom_not_dominated]]
        )
        for (var, arr), out in zip(self.__outputs__.items(), predictions):
            setattr(self, f"out_{var}", np.vstack([arr, out[not_dom_not_dominated]]))
            self.__outputs__[var] = getattr(self, f"out_{var}")

    # ----- hybrid updating -----

    def _hybrid(self):
        for i in range(self.num_hybrid_fragments):
            inds = self.hybrid_indices[
                i * self.hybrid_batch_size : (i + 1) * self.hybrid_batch_size
            ]

            trial_pop = np.vstack([self._mutate(i) for i in inds])
            self._ensure_constraint(trial_pop, inds)
            scaled_trial_pop = self._scale_parameters(trial_pop)
            trial_predictions = self.evaluate(scaled_trial_pop)
            trial_energies = np.vstack(
                [
                    self.loss_func(out, p)
                    for out, p in zip(zip(*trial_predictions), scaled_trial_pop)
                ]
            )
            self.selection(trial_energies, trial_pop, trial_predictions, inds)
            self._promote_lowest_energy()

        if self.parameter_adaptation:
            if len(self.successful_f) > 1:
                self.successful_f = np.concatenate(self.successful_f)
                self.successful_cr = np.concatenate(self.successful_cr)
            else:
                self.successful_f = np.array(self.successful_f).flatten()
                self.successful_cr = np.array(self.successful_cr).flatten()

    def _hybrid_update(self, loc, inds, pop, energies, predictions):
        self.population[inds] = np.where(loc[:, np.newaxis], pop, self.population[inds])
        self.population_energies[inds] = np.where(
            loc[:, np.newaxis], energies, self.population_energies[inds]
        )
        for arr, out in zip(self.__outputs__.values(), predictions):
            arr[inds] = np.where(loc[:, np.newaxis], predictions, arr[inds])
        self._update_archive(inds[loc])

        if self.parameter_adaptation:
            self.successful_f.append(np.atleast_1d(self.mutation_vec[inds][loc]))
            self.successful_cr.append(np.atleast_1d(self.cross_over_vec[inds][loc]))

    def _select_hybrid_so(self, energies: np.ndarray, pop, predictions, inds):
        loc = (energies < self.population_energies[inds]).flatten()
        self._hybrid_update(loc, inds, pop, energies, predictions)

    def _select_hybrid_mo(self, energies, pop, predictions, inds):
        dominant = dominates(energies, self.population_energies[inds])
        dominated = dominates(self.population_energies[inds], energies)
        not_dom_not_dominated = np.logical_not(np.logical_or(dominant, dominated))
        self._hybrid_update(dominant, inds, pop, energies, predictions)

        self.population = np.vstack([self.population, pop[not_dom_not_dominated]])
        self.population_energies = np.vstack(
            [self.population_energies, energies[not_dom_not_dominated]]
        )
        for (var, arr), out in zip(self.__outputs__.items(), predictions):
            setattr(self, f"out_{var}", np.vstack([arr, out[not_dom_not_dominated]]))
            self.__outputs__[var] = getattr(self, f"out_{var}")

    # -- initialisation --

    def _init(self):
        """Calculate all energies for initial population."""
        if np.all(np.isinf(self.population_energies)):
            self.callbacks.on_init_begin(self.logs())

            pop = self._scale_parameters(self.population)
            outputs = self.evaluate(pop)
            energies = np.vstack(
                [self.loss_func(out, p) for out, p in zip(zip(*outputs), pop)]
            )

            self.population_energies[:] = energies
            for rec, out in zip(self.__outputs__.values(), outputs):
                rec[:] = out
            self._promote_lowest_energy()
            self._update_pbar()

            self.callbacks.on_init_end(self.logs())

    def _truncate(self):
        """
        Truncate the population to the original popsize when performing
        multi-objective optimisation using non-domination criteria and
        crowding distance.
        """
        if np.size(self.population, 0) > self.num_population_members:
            to_save = self.num_population_members
            nd_sort = nds(self.population_energies)
            bin_sizes = np.cumsum([len(i) for i in nd_sort])
            first_larger = np.where(bin_sizes > to_save)[0][0]

            already_saved = bin_sizes[first_larger - 1] if first_larger else 0
            to_select = to_save - already_saved

            if to_select:
                tc = tournament_crowding(
                    self.population_energies[nd_sort[first_larger]], to_select
                )
                selected = np.array(nd_sort[first_larger], dtype="i")[tc]
            else:
                selected = []

            if first_larger:
                saved = np.concatenate(
                    [np.concatenate(nd_sort[:first_larger]), selected]
                ).astype("i")
            else:
                saved = selected

            self.population = self.population[saved]
            self.population_energies = self.population_energies[saved]
            for var, arr in self.__outputs__.items():
                setattr(self, f"out_{var}", arr[saved])
                self.__outputs__[var] = getattr(self, f"out_{var}")

    def _update_archive(self, loc):
        if self.archive_active:
            self.archive = np.vstack((self.archive, self.population[loc]))
            if np.size(self.archive, 0) > self.archive_size:
                self.archive = self.archive[
                    self.rng.choice(
                        self.archive.shape[0], self.archive_size, replace=False
                    )
                ]

    def _set_verbosity(self, verbose):
        if verbose:
            self.mpi._persistent_pbar = False
            self.mpi.init_report_persistent(
                self.num_population_members * (self.maxiter + 1)
            )
        else:
            self.mpi._persistent_pbar = True

    def _update_pbar(self):
        if self.mpi.pbar:
            self.mpi.pbar.set_description(f"GEN {self._nit}")
            self.mpi.pbar.set_postfix(loss=self.population_energies[0])

    def evaluate(self, parameters, **kwargs):
        """
        Evaluate the outputs from running parameters through your model.

        Parameters
        ----------
        parameters : ndarray
            An array of parameter vectors.
        Returns
        -------
        prediction : ndarray
            An array of predictions corresponding to each population member.
        """
        population = np.reshape(parameters, (-1, self.parameter_count))
        outputs = self.mpi.run(population, **kwargs)

        if not self.mpi.using_mpio:
            for ret, var in zip(outputs, self.mpi.return_attrs()):
                COMM.Bcast([ret, self.mpi.mpi_dtypes[var]], root=MPISRC.MASTER_RANK)

        return outputs

    def logs(self):
        x = self.x()
        logs = {
            "population": self._scale_parameters(self.population),
            "x": np.copy(x[0]),
            "energies": np.copy(self.population_energies),
            "loss": np.copy(x[1]),
        }
        return logs

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        """Generate optimizer from checkpoint."""

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    @staticmethod
    def loss_func(*args):
        """Output is a tuple of values, so by default want to compose into a
        single output vector."""
        return np.concatenate([np.atleast_1d(v) for v in args[0]]).astype("d")
