"""
Callback functions for Differential Evolution optimisation.
"""
from datetime import datetime, timedelta
import pickle
import os
import shutil
import sys

import numpy as np

from cajal.common import Dict
from cajal.common.logging import logger
from cajal.mpi import Backend as MPI


_MACHEPS = np.finfo(np.float64).eps


def configure_callbacks(callbacks, de):
    """Configure callbacks for use in Differential Evolution
    optimisation routine.

    Parameters
    ----------
    callbacks : list
        list of Callbacks
    de : DEBASE subclass
        subclass of DEBASE (i.e. Differential Evolution solver.)
    """
    if isinstance(callbacks, CallbackList):
        return callbacks

    if not callbacks:
        callbacks = []

    de.history = History()
    callbacks = [de.history, *callbacks]
    callback_list = CallbackList(callbacks)
    callback_list.set_de(de)

    return callback_list


class StateKeys:
    """State aliases."""

    SOLVE = "solve"
    INIT = "init"


class Callback:
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        self.de = None
        self.params = None

    def set_params(self, params):
        """Set callback parameters."""
        self.params = params

    def set_de(self, de):
        """Set the DE instance."""
        self.de = de

    def on_solve_begin(self, logs: dict = None):
        """Called when initially running DE solver.

        Parameters
        ----------
        logs : dict, optional
            Data, by default None
        """

    def on_solve_end(self, logs: dict = None):
        """Called when solver exits.

        Parameters
        ----------
        logs : dict, optional
            Data, by default None
        """

    def on_init_begin(self, logs: dict = None):
        """Called before first calculating all losses and predictions.
        Subclasses should override for any actions to run.
        """

    def on_init_end(self, logs: dict = None):
        """Called after first calculating all losses and predictions.
        Subclasses should override for any actions to run.
        """

    def on_generation_begin(self, generation, logs: dict = None):
        """Called at the start of running a complete generation.
        Subclasses should override for any actions to run.
        """

    def on_generation_end(self, generation, logs: dict = None):
        """Called at the end of running a complete generation.
        Subclasses should override for any actions to run.
        """


class CallbackList:
    """Gather callbacks into a list that dispatches commands
    to all internal callback functions."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self.params = {}
        self.de = None

    def append(self, callback):
        """Append a callback to the list."""
        self.callbacks.append(callback)

    def set_params(self, params):
        """Set callback parameters."""
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_de(self, de):
        """Set DE instance."""
        self.de = de
        for callback in self.callbacks:
            callback.set_de(de)

    def __iter__(self):
        return iter(self.callbacks)

    def _call_begin_hook(self, state, logs=None):
        """Helper function for on_{solve|init|generation}_begin methods."""
        if state == StateKeys.SOLVE:
            self.on_solve_begin(logs)
        elif state == StateKeys.INIT:
            self.on_init_begin(logs)
        else:
            self.on_generation_begin(logs)

    def _call_end_hook(self, state, logs=None):
        """Helper function for on_{solve|init|generation}_end methods."""
        if state == StateKeys.SOLVE:
            self.on_solve_end(logs)
        elif state == StateKeys.INIT:
            self.on_init_end(logs)
        else:
            self.on_generation_end(logs)

    def on_solve_begin(self, logs=None):
        """Called when initially running the solver."""
        for c in self.callbacks:
            c.on_solve_begin(logs)

    def on_solve_end(self, logs=None):
        """Called when solver returns."""
        for c in self.callbacks:
            c.on_solve_end(logs)

    def on_init_begin(self, logs=None):
        """Called when calculating losses for initial population."""
        for c in self.callbacks:
            c.on_init_begin(logs)

    def on_init_end(self, logs=None):
        """Called after initial population losses have been calculated."""
        for c in self.callbacks:
            c.on_init_end(logs)

    def on_generation_begin(self, generation, logs=None):
        """Called before evolving a generation."""
        for c in self.callbacks:
            c.on_generation_begin(generation, logs)

    def on_generation_end(self, generation, logs=None):
        """Called after evolving a generation."""
        for c in self.callbacks:
            c.on_generation_end(generation, logs)


class History(Callback):
    """Callback that records events into a `History` object."""

    def __init__(self):
        super(History, self).__init__()
        self.generation = []
        self.history = Dict()
        self.x = None
        self.fun = None
        self.nfev = None
        self.nit = None
        self.message = None
        self.success = None

    def on_generation_end(self, generation, logs=None):
        logs = logs or {}
        self.generation.append(generation)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.de.history = self

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)

    def on_solve_end(self, logs=None):
        self.x = self.de.x()
        self.fun = self.de.population_energies[0]
        self.nfev = getattr(self.de, "_nfev")
        self.nit = getattr(self.de, "_nit")
        self.message = logs.get("message", None)
        self.success = logs.get("success", None)

    def meta(self):
        keys = ["x", "fun", "nfev", "nit", "message", "success"]
        meta = Dict({k: getattr(self, k) for k in keys})
        return meta

    def save_history(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def save_metadata(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.meta(), f)

    def save(self, path):
        to_save = Dict(history=self.history, metadata=self.meta())
        with open(path, "wb") as f:
            pickle.dump(to_save, f)


class Timer(Callback):
    """Time running the DE algorithm."""

    def __init__(self):
        super(Timer, self).__init__()
        self.t_start = None
        self.generation = []
        self.generation_times = []
        self.total_time = None

    def on_solve_begin(self, logs=None):
        self.t_start = datetime.now()

    def on_generation_end(self, generation, logs=None):
        t_now = datetime.now()
        t = t_now - self.t_start
        self.generation.append(generation)
        self.generation_times.append(t - sum(self.generation_times, timedelta(0)))

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)

    def on_solve_end(self, logs=None):
        t_now = datetime.now()
        t = t_now - self.t_start
        self.total_time = t

    @property
    def t_start_str(self):
        """Return solver start time as string"""
        return str(self.t_start)

    @property
    def generation_times_str(self):
        """Return generation run times as strings"""
        return [str(i) for i in self.generation_times]

    @property
    def total_time_str(self):
        """Return total solver run time as string"""
        return str(self.total_time)


class CheckConvergence(Callback):
    """
    Monitor population energies. Causes the solver to exit if::
        std(energies) <= atol + tol * |mean(energies)|
    """

    def __init__(self, atol=0, tol=0.1, min_gens=None):
        super(CheckConvergence, self).__init__()
        self.atol = atol
        self.tol = tol
        self.min_gens = min_gens or 0

    def on_generation_end(self, generation, logs=None):
        energies = logs["energies"]
        if generation >= self.min_gens:
            if np.any(np.isinf(energies)):
                return
            std = np.std(energies, axis=0)
            tol = self.atol + self.tol * np.abs(np.mean(energies, axis=0))
            converged = (std <= tol).all()
            if converged:
                message = (
                    f"DE solver exited in generation {generation} as "
                    f"the standard deviation of energies ({std}) is "
                    f"below the tolerance for population convergence : "
                    f"({tol})"
                )
                raise StopIteration(message)

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)


class EarlyStopping(Callback):
    """Monitor population energies. Causes the solver to exit
    if the best population energy has not improved by a prescribed
    amount over a prescribed number of generations."""

    def __init__(self, min_delta=0, patience=0, baseline=None, min_loss=None):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.baseline = baseline
        self.min_delta = -1 * abs(min_delta)
        self.min_loss = min_loss
        self.wait = 0
        self.best = None
        self.stopped_generation = 0
        self.emitted_warning = False

    def on_solve_begin(self, logs=None):
        self.wait = 0
        self.stopped_generation = 0
        self.emitted_warning = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf

    def on_generation_end(self, generation, logs=None):
        current = self._get_loss(logs)
        if current is None:
            return
        if self.min_loss is not None:
            if current <= self.min_loss:
                message = (
                    f"DE solver exited in generation {generation} as "
                    f"a solution with a loss ({current}) less than or "
                    f"equal to the minimum loss bound {self.min_loss} "
                    f"was found."
                )
                raise StopIteration(message)
        if np.less_equal(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_generation = generation
                message = (
                    f"Early Stopping generation " f"{self.stopped_generation:05d}."
                )
                raise StopIteration(message)

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)

    def _get_loss(self, logs=None):
        logs = logs or {}
        energies = logs.get("energies", None)
        if energies is None:
            return None
        loss = energies[0]
        if np.size(loss) > 1:
            if MPI.MASTER():
                if not self.emitted_warning:
                    logger.info(
                        "EarlyStopping is only configured for use with"
                        "single-objective losses. It will be skipped for "
                        "the duration of this optimisation."
                    )
                    self.emitted_warning = True
            return None
        return loss


class Logger(Callback):
    def __init__(self, save_dir, save_script=True):
        super(Logger, self).__init__()
        if MPI.MASTER():
            self.save_dir = save_dir
            if os.path.exists(self.save_dir):
                txt = (
                    f"{self.__class__.__name__} save directory {self.save_dir} "
                    f"exists. Data may be overwritten"
                )
                logger.warning(txt)
            os.makedirs(self.save_dir, exist_ok=True)
            if save_script:
                script_save = os.path.join(self.save_dir, "script.py")
                file = os.path.abspath(sys.argv[0])
                shutil.copyfile(file, script_save)

    def on_generation_end(self, generation, logs=None):
        if MPI.MASTER():
            generation_dir = "gen_%04d.log" % generation
            savedir = os.path.join(self.save_dir, generation_dir)
            self.save(logs, savedir)

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)

    @staticmethod
    def save(logs, path):
        with open(path, "wb") as f:
            pickle.dump(logs, f)


class CheckPointing(Callback):
    """Save optimiser state to continue optimisation later."""

    def __init__(self, path, save_freq=1, save_latest_only=True):
        super(CheckPointing, self).__init__()
        self.path = path
        self.save_freq = save_freq
        self.save_latest_only = save_latest_only


class SlackBot(Callback):
    """Post messages to a slack channel."""

    def __init__(self, message, channel=None):
        super(SlackBot, self).__init__()
        self.channel = channel
        self.message = message

    def on_generation_end(self, generation, logs=None):
        from cajal.common.slack import post_message_to_slack

        loss = logs.get("loss", None)
        try:
            if loss.ndim > 1:
                loss = (
                    "[\n"
                    + "".join(
                        ["\t" + str([round(x, 3) for x in s]) + ",\n" for s in loss]
                    )
                    + "]"
                )
                logs.update({"loss": loss})
        except AttributeError:
            pass
        msg = self.message.format(generation=generation, **logs)
        post_message_to_slack(msg, channel=self.channel)

    def on_init_end(self, logs=None):
        self.on_generation_end(0, logs)

    def on_solve_end(self, logs=None):
        from cajal.common.slack import post_message_to_slack

        post_message_to_slack("Solver exiting. Final loss: {loss:.03f}".format(**logs))
