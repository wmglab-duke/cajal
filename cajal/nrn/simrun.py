"""
Classes to construct simulation environments to run models.
"""
import gc
import math

from neuron import h
import numpy as np
from tqdm import tqdm

from cajal.common.logging import logger
from cajal.common.math import round_nearest
from cajal.exceptions import (
    BlockThresholdTopInitError,
    BlockThresholdBottomInitError,
    ThresholdTopInitError,
    ThresholdBottomInitError,
)
from cajal.mpi import Backend as MPI
from cajal.nrn.__backend import Backend as N
from cajal.nrn.electrodes import ExtraStimList, StimList
from cajal.units import ms, unitdispatch


class SimulationEnvironment:
    """Environment within which to run NEURON simulations. If parameters
    are not specified, the Environment inherits v_init, dt, tstop, temp
    and rhoe from the backend.

    Parameters
    ----------
    axons : list
        List of axons in model. (default=None)
    extra_stim : list
        List of extracellular stimulating electrodes. (default=None)
    intra_stim : list
        List of intracellular stimulating electrodes. (default=None)
    monitors : list
        List of StateMonitors. (default=None)

    """

    def __init__(
        self, axons=None, extra_stim=None, intra_stim=None, LFP=None, monitors=None
    ):
        self.__axons = None
        self.__extra_stim = ExtraStimList()
        self.__intra_stim = StimList()
        self.__LFP_electrodes = None
        self.__state_monitors = None

        self._initialised = False
        self._tstart = 0 * ms

        self.axons = axons
        self.extra_stim = extra_stim
        self.intra_stim = intra_stim
        self.LFP_electrodes = LFP
        self.monitors = monitors

        self.stimulus_waveform = None
        self.state = h.SaveState()

    @property
    def axons(self):
        """Return axons in environment."""
        return self.__axons

    @axons.setter
    def axons(self, axons):
        if self.axons is axons:
            pass
        else:
            try:
                self.__axons = axons
                self._ensure_geometric_relations()

            except TypeError:
                raise TypeError(
                    "Axons must be supplied in the form of a list "
                    "of objects of type cajal.nrn.cells.Axon."
                ) from None

    @property
    def extra_stim(self):
        """Return extracellular sources in environment."""
        return self.__extra_stim

    @extra_stim.setter
    def extra_stim(self, extra_stim):
        if isinstance(extra_stim, ExtraStimList):
            self.__extra_stim = extra_stim
        elif self.extra_stim._stims is extra_stim:
            pass
        elif not extra_stim:
            self.__extra_stim = ExtraStimList()
        else:
            self.__extra_stim = ExtraStimList(*extra_stim)
            self._ensure_geometric_relations()

    @property
    def intra_stim(self):
        """Return extracellular sources in environment."""
        return self.__intra_stim

    @intra_stim.setter
    def intra_stim(self, intra_stim):
        if self.intra_stim._stims is intra_stim:
            pass
        else:
            try:
                self.__intra_stim = StimList(intra_stim)

            except TypeError:
                raise TypeError(
                    "Electrodes must be supplied in the form of a list of "
                    "objects of type cajal.nrn.electrodes._Extra."
                ) from None

    @property
    def LFP_electrodes(self):
        """LFP electrodes."""
        return self.__LFP_electrodes

    @LFP_electrodes.setter
    def LFP_electrodes(self, LFP_electrodes):
        if self.LFP_electrodes is LFP_electrodes:
            pass
        else:
            try:
                self.__LFP_electrodes = LFP_electrodes
                self._ensure_geometric_relations()

            except TypeError:
                raise TypeError(
                    "Electrodes must be supplied in the form of a list of "
                    "objects of type cajal.nrn.electrodes._Extra."
                ) from None

    @unitdispatch
    def smartrun(
        self,
        runtime: "ms" = None,
        steady_state=False,
        progressbar=True,
        early_stopping=None,
        reinit=False,
    ):
        early_stopping = (
            early_stopping if early_stopping is not None else N.EARLY_STOPPING
        )
        if not self.axons:
            logger.info("No axons in SimulationEnvironment.")
            return None
        runtime = runtime or N.tstop
        if runtime > N.chunksize:
            return self.longrun(
                runtime,
                steady_state=steady_state,
                progressbar=progressbar,
                early_stopping=early_stopping,
                reinit=reinit,
            )
        return self.run(
            runtime,
            steady_state=steady_state,
            progressbar=progressbar,
            early_stopping=early_stopping,
            reinit=reinit,
        )

    @unitdispatch
    def run(
        self,
        runtime: "ms" = None,
        dt: "ms" = None,
        steady_state=None,
        progressbar=True,
        early_stopping=None,
        reinit=False,
        longrunning=False,
    ):
        """
        Run a simulation with extracellular stimulating electrodes.

        Parameters
        ----------
        runtime : float, optional
            Simulation duration in ms, by default global N.tstop.
        dt: float, optional
            Simulation timestep in ms, by default global N.dt.
        steady_state : List[float], optional
            Whether or not to run initially with large time-step to
            achieve steady state (run from t=-steady_state[0]
            to t=0 with dt=steady_state[1]), by default None
        progressbar : bool, optional
            Choose whether to show a progress bar of simulation.
        early_stopping : bool, optional
            Will terminate simulation if an AP is detected in any
            axon within the environment. Use with care: if more than one
            axon is in your environment, the entire simulation will be
            terminated if an AP is detected in a single axon.
            By default False.
        reinit : bool, optional
            Select whether to reinitialise environment when running.
            By default False.
        """
        if not self.axons:
            logger.info("No axons in SimulationEnvironment.")
            return None

        exception = None
        runtime = runtime or N.tstop
        if dt is not None:
            N.dt = float(dt.to("ms"))
        reinit = True if steady_state else reinit

        if reinit:
            self._initialised = False

        early_stopping = N.EARLY_STOPPING if early_stopping is None else early_stopping

        # define time vectors
        if steady_state or not self._initialised:
            self._tstart = 0 * ms
            if self.monitors:
                for monitor in self.monitors:
                    monitor.clear()
            for axon in self.axons:
                for apm in axon.apm:
                    apm.clear()

        t = np.arange(self._tstart, self._tstart + runtime, N.dt)

        if steady_state:
            tss = np.arange(-1 * abs(steady_state[0]), 0, steady_state[1])
            t = np.concatenate([tss, t])

        tvec = h.Vector(t)

        # vectorised extracellular stimulation
        self._ensure_geometric_relations()
        self.extra_stim.init(t)
        self.intra_stim.init(t)
        for axon in self.axons:
            axon.ppinit(t)

        play_vecs = []
        if self.extra_stim:
            play_vecs = self._set_play_vectors(tvec)

        # initialise
        if steady_state:
            self.achieve_steady_state(steady_state)
        else:
            self._init()

        if self._initialised:
            self.state.restore(1)
            for axon in self.axons:
                for apm in axon.apm:
                    apm._init_t()
            h.frecord_init()

        # run
        n_tsteps = int(runtime / h.dt)
        if steady_state:
            t = t[len(tss) :]
        r = tqdm(t) if progressbar else range(n_tsteps)

        try:
            for i, tt in enumerate(r):
                for axon in self.axons:
                    axon.advance()
                    axon.ppadvance(i)
                if early_stopping:
                    self._early_stopping()
                h.fadvance()
                if progressbar:
                    if math.fmod(round(tt, 3), 0.5) == 0:
                        r.set_description(f"{tt:.1f} ms")
        except StopIteration as e:
            exception = e
            if not MPI.ENABLED and N.log_early_stopping:
                logger.info(str(e))

        for axon in self.axons:
            for apm in axon.apm:
                apm.cache()
            axon.count_APs()

        # for continuing run
        self._tstart = round_nearest(h.t, N.dt) * ms
        # save state for continuing run
        self.state.save()
        # clear play vectors
        play_vecs.clear()
        # cache monitors
        if self.monitors:
            for monitor in self.monitors:
                monitor.cache()

        self._initialised = True

        if longrunning:
            if exception is not None:
                raise exception

        return self

    def _set_play_vectors(self, t):
        play_vecs = []
        for i, axon in enumerate(self.axons):
            for j, sec in enumerate(axon):
                vec = h.Vector(self.extra_stim.Ve[i][:, j])
                vec.play(sec(0.5)._ref_e_extracellular, t, 1)
                play_vecs.append(vec)
        return play_vecs

    simulate_ext = run

    @unitdispatch
    def longrun(
        self,
        runtime: "ms",
        chunksize: "ms" = None,
        steady_state=False,
        progressbar=True,
        early_stopping=False,
        reinit=False,
    ):
        chunksize = chunksize or N.chunksize
        n_chunks, rem = divmod(float(runtime), float(chunksize))
        n_chunks = int(n_chunks)
        n_steps = n_chunks + 1 if rem > 0 else n_chunks
        pbar = tqdm(range(n_steps)) if progressbar else range(n_steps)
        try:
            for chunk in pbar:
                if chunk == 0:
                    self.run(
                        chunksize,
                        steady_state=steady_state,
                        progressbar=False,
                        early_stopping=early_stopping,
                        reinit=reinit,
                        longrunning=True,
                    )
                elif chunk == n_steps - 1:
                    self.run(
                        rem or chunksize,
                        steady_state=False,
                        progressbar=False,
                        early_stopping=early_stopping,
                        longrunning=True,
                    )
                else:
                    self.run(
                        chunksize,
                        steady_state=False,
                        progressbar=False,
                        early_stopping=early_stopping,
                        longrunning=True,
                    )
                gc.collect()
                if progressbar:
                    pbar.set_description(f"{chunk*chunksize:.1f} ms")
        except StopIteration:
            pass

        return self

    def _init(self):
        if not self._initialised:
            for axon in self.axons:
                axon.init_voltages()
        h.t = 0
        h.finitialize()
        h.fcurrent()
        for axon in self.axons:
            for apm in axon.apm:
                apm._init_t()
        if not self._initialised:
            for axon in self.axons:
                axon.initialize()

    def _advance_e(self, t_ind):
        for i, axon in enumerate(self.axons):
            for j, sec in enumerate(axon):
                sec.e_extracellular = self.extra_stim.Ve[i][t_ind, j]

    def _early_stopping(self):
        for axon in self.axons:
            axon.check_early_stop()

    def achieve_steady_state(self, steady_state):
        """Runs simulation with long timestep (10ms) to achieve
        steady state.
        """
        for axon in self.axons:
            axon.init_voltages()

        h.finitialize()
        h.fcurrent()
        for axon in self.axons:
            axon.initialize()

        h.t = -1 * abs(steady_state[0])
        h.dt = steady_state[1]

        while h.t <= -h.dt:
            for axon in self.axons:
                axon.advance()
            h.fadvance()

        h.dt = N.dt
        h.t = 0
        for axon in self.axons:
            for apm in axon.apm:
                apm._init_t()
        h.frecord_init()

    def _ensure_geometric_relations(self):
        if self.extra_stim and self.axons:
            self.extra_stim.ensure_geometric_relations(self.axons)

    def find_thresh(
        self,
        upper_bound=0.3,
        lower_bound=0.0,
        resolution=0.01,
        increment=0.1,
        waveform=None,
        progressbar=False,
        verbose=False,
        max_attempts=20,
        max_upper=10,
        n_attempts=0,
        exception=None,
    ):
        if lower_bound < 0:
            raise ValueError(
                "Lower bound for binary search cannot be lower " "than 0mA."
            )
        max_attempts = max_attempts or np.inf
        if n_attempts <= max_attempts and upper_bound <= max_upper:
            try:
                return self.find_thresh_comp(
                    upper_bound, lower_bound, resolution, waveform, progressbar, verbose
                )
            except ThresholdTopInitError as e:
                return self.find_thresh(
                    upper_bound * (1 + increment),
                    upper_bound,
                    resolution,
                    increment,
                    waveform,
                    progressbar,
                    verbose,
                    max_attempts,
                    max_upper,
                    n_attempts + 1,
                    e,
                )
            except ThresholdBottomInitError as e:
                return self.find_thresh(
                    lower_bound,
                    lower_bound * (1 - increment),
                    resolution,
                    increment,
                    waveform,
                    progressbar,
                    verbose,
                    max_attempts,
                    max_upper,
                    n_attempts + 1,
                    e,
                )
        else:
            if n_attempts > max_attempts:
                msg = (
                    "upper" if isinstance(exception, ThresholdTopInitError) else "lower"
                )
                logger.error(
                    f"Failed to establish appropriate {msg} bound for binary threshold "
                    f"search within permitted # of attempts ({max_attempts}), for "
                    f"{self.axons[0]}, stimulus : {self.stimulus_waveform}; returning NaN."
                )
            else:
                logger.error(
                    f"Failed to find initial upper boundary estimate that generated an "
                    f"action potential less than the specified maximum upper amplitude "
                    f"{max_upper}mA for {self.axons[0]}, stimulus : {self.stimulus_waveform}; "
                    f"returning NaN."
                )
            return np.nan

    def find_block_thresh(
        self,
        upper_bound=1.0,
        lower_bound=0.0,
        resolution=0.01,
        increment=0.1,
        waveform=None,
        progressbar=False,
        verbose=False,
        max_attempts=20,
        n_attempts=0,
        exception=None,
    ):
        if n_attempts <= max_attempts:
            try:
                return self.find_block_thresh_comp(
                    upper_bound, lower_bound, resolution, waveform, progressbar, verbose
                )
            except BlockThresholdTopInitError as e:
                return self.find_block_thresh(
                    upper_bound * (1 + increment),
                    upper_bound,
                    resolution,
                    increment,
                    waveform,
                    progressbar,
                    verbose,
                    max_attempts,
                    n_attempts + 1,
                    e,
                )
            except BlockThresholdBottomInitError as e:
                return self.find_block_thresh(
                    lower_bound,
                    lower_bound * (1 - increment),
                    resolution,
                    increment,
                    waveform,
                    progressbar,
                    verbose,
                    max_attempts,
                    n_attempts + 1,
                    e,
                )
        else:
            msg = (
                "upper"
                if isinstance(exception, BlockThresholdTopInitError)
                else "lower"
            )
            logger.error(
                f"Failed to establish appropriate {msg} bound for binary block threshold "
                f"search within permitted # of attempts ({max_attempts}), for "
                f"{self.axons[0]}, stimulus : {self.stimulus_waveform}; returning NaN."
            )
            return np.nan

    def find_thresh_comp(
        self,
        upper_bound=1.0,
        lower_bound=0.0,
        resolution=0.01,
        stimulus_waveform=None,
        progressbar=False,
        verbose=False,
    ):
        """
        Determine threshold current amplitude to generate propagating action
        potential for given stimulus waveform.

        Parameters
        ----------
        upper_bound: float, optional
            Initial upper bound for binary search. By default 1.0.
        lower_bound: float, optional
            Initial lower bound for binary search. By default 0.0.
        threshold: float, optional
            Vm at which AP is said to have occurred. By default 0.
        resolution: float, optional
            Size of window for found threshold as fraction of upper bound.
            By default 0.01.
        test_node: float, optional
            Distance of node from axon origin (in mm) at which to check
            for an action potential. By default None.
        stimulus_waveform: Stimulus, optional)
            Waveform to use to stimulate axon. If None, uses rectangular pulse.
            By default None.
        progressbar: bool, optional
            Whether or not to display progressbar. By default False.
        verbose: bool, optional
            Log status. By default False.

        Returns
        -------
        float: Threshold current amplitude (upper bound of binary search window).
        """

        if not self.stimulus_waveform:
            if stimulus_waveform:
                self.stimulus_waveform = [
                    (e_stim << stim).stimulus
                    for e_stim, stim in zip(self.extra_stim, stimulus_waveform)
                ]
            else:
                self.stimulus_waveform = [e_stim.stimulus for e_stim in self.extra_stim]

        # Check upper_bound elicits an AP and
        if not self.check_ap_threshold(upper_bound, progressbar, verbose):
            raise ThresholdTopInitError(
                "The initial top bound for the binary search does not "
                "generate an action potential."
            )
        if self.check_ap_threshold(lower_bound, progressbar, verbose):
            raise ThresholdBottomInitError(
                "The initial bottom bound for the binary search generates "
                "an action potential."
            )

        stimamp_top = upper_bound
        stimamp_bottom = lower_bound
        thresh_window = (stimamp_bottom - stimamp_top) / stimamp_top
        while abs(thresh_window) >= resolution:
            stimamp = (stimamp_bottom + stimamp_top) / 2
            if self.check_ap_threshold(stimamp, progressbar, verbose):
                stimamp_top = stimamp
            else:
                stimamp_bottom = stimamp
            thresh_window = (stimamp_bottom - stimamp_top) / stimamp_top
        stimamp = (stimamp_bottom + stimamp_top) / 2
        if self.check_ap_threshold(stimamp, progressbar, verbose):
            stimamp_top = stimamp
        if verbose:
            logger.info("Threshold found: %g mA", stimamp_top)
        return stimamp_top

    def find_block_thresh_comp(
        self,
        upper_bound=1.0,
        lower_bound=0.0,
        resolution=0.01,
        stimulus_waveform=None,
        progressbar=False,
        verbose=False,
    ):
        """
        Determine threshold current amplitude to block propagating action
        potential for given stimulus waveform.

        Parameters
        ----------
        upper_bound: float, optional
            Initial upper bound for binary search. By default 1.0.
        lower_bound: float, optional
            Initial lower bound for binary search. By default 0.0.
        threshold: float, optional
            Vm at which AP is said to have occurred. By default 0.
        resolution: float, optional
            Size of window for found threshold as fraction of upper bound.
            By default 0.01.
        test_node: float, optional
            Distance of node from axon origin (in mm) at which to check
            for an action potential. By default None.
        stimulus_waveform: Stimulus, optional
            Waveform to use to stimulate axon. If None, uses waveform supplied
            upon envronment construction (in extra_stim). By default None.
        progressbar: bool, optional
            Display progressbar of simulation. By default False.
        verbose:

        Returns
        -------
        float: Threshold current amplitude (upper bound of binary search window).
        """

        if not self.stimulus_waveform:
            self.stimulus_waveform = (
                (self.extra_stim[0] << stimulus_waveform).stimulus
                if stimulus_waveform
                else self.extra_stim[0].stimulus
            )

        # Check upper_bound elicits an AP and
        if self.check_ap_threshold(upper_bound, progressbar, verbose):
            raise BlockThresholdTopInitError(
                "The initial top bound for binary search generates an "
                "action potential."
            )
        if not self.check_ap_threshold(lower_bound, progressbar, verbose):
            raise BlockThresholdBottomInitError(
                "The initial bottom bound for binary search does not "
                "generate an action potential."
            )

        stimamp_top = upper_bound
        stimamp_bottom = lower_bound
        thresh_window = (stimamp_bottom - stimamp_top) / stimamp_top
        while abs(thresh_window) >= resolution:
            stimamp = (stimamp_bottom + stimamp_top) / 2
            if not self.check_ap_threshold(stimamp, progressbar, verbose):
                stimamp_top = stimamp
            else:
                stimamp_bottom = stimamp
            thresh_window = (stimamp_bottom - stimamp_top) / stimamp_top
        stimamp = (stimamp_bottom + stimamp_top) / 2
        if not self.check_ap_threshold(stimamp, progressbar, verbose):
            stimamp_top = stimamp

        if verbose:
            logger.info("Block threshold found: %g mA", stimamp_top)
        return stimamp_top

    def check_ap_threshold(self, stimamp, progressbar=False, verbose=False):
        """
        Check whether an AP is detected given a pattern of extracellular
        stimulation. Used when calculating thresholds.

        Parameters
        ----------
        stimamp: float
            Stimulus amplitude to check.
        progressbar: bool, optional
            Whether or not to display progressbar. By default False.
        verbose: bool, optional
            Log activity to stdout. By default False.

        Returns
        -------
        bool: Whether AP is detected at the given stimulus amplitude.
        """
        if verbose:
            logger.info(
                f"Checking whether stimulus amplitude "
                f"{stimamp:.3f} mA yields a propagating "
                f"action potential."
            )
        for e_stim, stim in zip(self.extra_stim, self.stimulus_waveform):
            e_stim << stim * stimamp
        self.smartrun(early_stopping=True, progressbar=progressbar, reinit=True)
        n_aps = self.axons[0].count_APs()
        propagated = n_aps > 0

        if verbose:
            if propagated:
                logger.info(
                    f"Activation achieved! Total number of "
                    f"propagated APs detected by {self.axons[0].apm}: "
                    f"{n_aps}"
                )
            else:
                logger.info("No APs detected.")

        return propagated

    @property
    def n_extra_stim(self):
        return len(self.extra_stim) if self.extra_stim else 0

    @property
    def n_intra_stim(self):
        return len(self.intra_stim) if self.intra_stim else 0

    @property
    def n_axons(self):
        """Total number of axons in simulation environment.

        Returns
        -------
        int
            Number of axons.
        """
        if self.axons:
            return len(self.axons)
        return 0

    def __repr__(self):
        string = (
            "\nSimulation Environment:"
            "\n  # axons: {}"
            "\n  # extracellular stimulating electrodes: {}"
            "\n  dt: {}"
        ).format(self.n_axons, self.n_extra_stim, N.dt)
        return string
