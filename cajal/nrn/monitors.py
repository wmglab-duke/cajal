"""
Classes to monitor state variables in your multi-section models
and monitor changes in transmembrane voltage for propagating action
potentials.
"""
from typing import List

import numpy as np
from neuron import h
from scipy.signal import find_peaks
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MPL_INSTALLED = True
except ImportError:
    MPL_INSTALLED = False

from cajal.common import Dict
from cajal.common.logging import logger
from cajal.common.math import round_nearest
from cajal.nrn import cells, Section, Backend as N
from cajal.units import unitdispatch, ms, mV, strip_units


class StateMonitorSpec:
    def __init__(self, compartments, state_vars, dtype):
        self.compartments = compartments
        if isinstance(state_vars, (list, tuple)):
            self.state_vars = state_vars
        else:
            self.state_vars = [state_vars]
        self.dtype = dtype


class StateMonitor:

    """Monitor state variable in your models."""

    def __init__(self, compartments, state_vars, dtype="f"):
        """Construct StateMonitor

        Parameters
        ----------
        compartments : h.Section, list, Axon
            Model compartments (h.Section objects) from which to record
            state variables. May be a list, single section, or an Axon
            model (model that subclasses cajal.nrn.cells.Axon)
        state_vars : str, iterable
            State variable names to record. All state variables must be
            present in all compartments passed to the constructor.
        """

        if isinstance(compartments, list):
            self.compartments: List[Section] = compartments
        elif isinstance(compartments, cells.Axon):
            self.compartments: List[Section] = compartments.py_all
        else:
            self.compartments: List[Section] = [compartments]

        if isinstance(state_vars, (list, tuple)):
            self.state_vars = state_vars
        else:
            self.state_vars = [state_vars]

        self._construct_records()
        self._t = h.Vector().record(getattr(h, "_ref_t"))
        self.__x = None
        self.__y = None
        self.__z = None
        self.__cache__ = Dict()
        self.dtype = dtype

    @property
    def cached(self):
        return self.__cache__

    def _construct_records(self):
        for var in self.state_vars:
            recs = [
                h.Vector().record(getattr(sec(0.5), "_ref_{}".format(var)))
                for sec in self.compartments
            ]
            setattr(self, "_rec_{}".format(var), recs)

    def __getattr__(self, name):
        if name in self.state_vars:
            recs = getattr(self, "_rec_{}".format(name))
            return np.array(
                np.vstack([np.array(rec) for rec in recs]), dtype=self.dtype
            )

        raise AttributeError(
            f"{self.__class__.__name__} object has no " f"attribute named {name}"
        )

    @property
    def t(self):  # pylint: disable=invalid-name
        """Time-steps in simulation.

        Returns
        -------
        np.ndarray
            1D vector of timesteps of simulation.
        """
        return np.array(self._t)

    @property
    def x(self):  # pylint: disable=invalid-name
        """x-coordinates of all compartments being monitored.

        Returns
        -------
        np.ndarray
            x-coordinates
        """
        if self.__x is None:
            self.__x = np.array(
                [
                    np.mean([sec.x3d(i) for i in range(sec.n3d())])
                    for sec in self.compartments
                ]
            )
        return self.__x

    @property
    def y(self):  # pylint: disable=invalid-name
        """y-coordinates of all compartments being monitored.

        Returns
        -------
        np.ndarray
            y-coordinates
        """
        if self.__y is None:
            self.__y = np.array(
                [
                    np.mean([sec.y3d(i) for i in range(sec.n3d())])
                    for sec in self.compartments
                ]
            )
        return self.__y

    @property
    def z(self):  # pylint: disable=invalid-name
        """z-coordinates of all compartments being monitored.

        Returns
        -------
        np.ndarray
            z-coordinates
        """
        if self.__z is None:
            self.__z = np.array(
                [
                    np.mean([sec.z3d(i) for i in range(sec.n3d())])
                    for sec in self.compartments
                ]
            )
        return self.__z

    def cache(self):
        for var in self.state_vars:
            try:
                self.__cache__[var] = np.concatenate(
                    [self.__cache__[var], getattr(self, var)], axis=1
                )
            except (KeyError, ValueError):
                self.__cache__[var] = getattr(self, var)
        try:
            self.__cache__["t"] = np.concatenate(
                [self.__cache__["t"], getattr(self, "t")]
            )
        except (KeyError, ValueError):
            self.__cache__["t"] = getattr(self, "t")

    def clear(self):
        self.__cache__.clear()

    def video(self, var, path, axis="y", lim=None, show_timer=False, progressbar=True):
        if not MPL_INSTALLED:
            raise RuntimeError("Need matplotlib to generate video.")

        data = getattr(self, var)

        if lim is None:
            minimum = data.min()
            if minimum < 0:
                minimum = 1.1 * minimum
            elif minimum == 0:
                minimum = -0.1
            else:
                minimum = 0.9 * minimum

            maximum = data.max()
            if maximum > 0:
                maximum = 1.1 * maximum
            elif maximum == 0:
                maximum = 0.1
            else:
                maximum = 0.9 * maximum

            lim = (minimum, maximum)

        fig, ax = plt.subplots()

        if show_timer:
            textvar = plt.text(
                0.1,
                0.95,
                f"{0*N.dt:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        plt.ylim(*lim)

        x = getattr(self, axis)
        (line,) = ax.plot(x, data[:, 0])

        if progressbar:
            pbar = tqdm(total=data.shape[1])

        def animate(i):
            line.set_ydata(data[:, i])  # update the data.
            if progressbar:
                pbar.update()
            if show_timer:
                textvar.set(text=f"{i*N.dt:.3f}")
            return (line,)

        ani = animation.FuncAnimation(
            fig, animate, interval=2, blit=True, frames=data.shape[1]
        )

        writer = animation.FFMpegWriter(
            fps=30 * (0.005 / float(N.dt)), metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(path, writer=writer)


class APMonitor:
    """Monitor compartments for passing action potentials. Wraps
    the NEURON APCount class.

    This class is responsible for calling early stopping during parallel
    execution, which can save a lot of time during simulation if you
    are only interested in the events leading up to an action potential /
    whether an AP was generated and propagated.

    An action potential is defined as a rising edge in transmembrane
    potential that crosses 'threshold' (by default -20mV).
    """

    @unitdispatch
    def __init__(
        self, compartment, threshold: "mV" = -20, t: "ms" = None, record_voltages=False
    ):
        is_sec = isinstance(compartment, Section)
        self.compartment = compartment(0.5) if is_sec else compartment
        self.record_voltages = record_voltages
        if record_voltages:
            self.rec = StateMonitor(compartment, "v")
        self.apc = h.APCount(self.compartment)

        # -- set parameters --
        self._threshold = threshold
        self.apc.thresh = float(self._threshold)
        self._time = t if t is not None else 0 * ms
        self.time_f = float(self._time)
        self.tstart = None
        self.__cache__ = None
        self._init_t()

        # -- add reference to axon for easy access --
        sec = compartment if is_sec else compartment.sec
        sec.cell().apm.append(self)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    @unitdispatch
    def threshold(self, value: "mV"):
        self._threshold = value
        self.apc.thresh = float(self._threshold)

    @property
    def time(self):
        return self._time

    @time.setter
    @unitdispatch
    def time(self, value: "ms"):
        self._time = value
        self.time_f = float(value)

    def _init_t(self):
        self._spikes = h.Vector()
        self.apc.record(self._spikes)
        self.tstart = round_nearest(h.t, N.dt_)

    def check_early_stop(self):
        if self._n > 0:
            self._early_stopping()

    def _early_stopping(self):
        txt = (
            f"called early_stopping() at time {h.t:.3f} when "
            f"{self.compartment}.v = {self.compartment.v}"
        )
        raise StopIteration(txt)

    def __repr__(self):
        return f"APMonitor in {self.compartment}"

    @unitdispatch
    def n(self, t: "ms" = None, use_peaks=False):  # pylint: disable=invalid-name
        """Number of action potentials that occurred. If a t parameter
        was specified, only action potentials that occurred after that
        time are registered. This can be useful if you want to ignore
        early stimulus effects.

        Returns
        -------
        int
            Number of recorded action potentials.
        """
        if use_peaks:
            if self.record_voltages:
                return len(self.peaks(t))
            logger.warning(
                "APMonitor was not instructed to record voltages."
                "Returning n from h.APCount instead."
            )
            return self.n(t, use_peaks=False)
        t = strip_units(t) if t is not None else self.time_f
        if self.__cache__ is not None:
            return np.count_nonzero(self.__cache__ > t)
        if t > 0:
            ap_valid = np.array(self._spikes) > t
            return np.count_nonzero(ap_valid)
        return int(self.apc.n)

    @unitdispatch
    def peaks(self, t: "ms" = None, threshold=None):
        if self.record_voltages:
            t = strip_units(t) if t is not None else self.time_f
            tvec = self.rec.cached.t
            v = self.rec.cached.v[0, :]
            peaks = tvec[find_peaks(v, height=threshold)[0]]
            return peaks[peaks > t]
        raise AttributeError(
            "APMonitor must be initialized with record_voltages=True"
            "to use peaks() function."
        )

    @property
    def _n(self):
        spikes = self._spikes.as_numpy()
        if self.__cache__ is not None and self.__cache__.size > 0:
            last = self.tstart
            return np.count_nonzero(
                np.logical_and(spikes > self.time_f, spikes > last + h.dt)
            )
        if self.time_f > 0:
            return np.count_nonzero(spikes > self.time_f)
        return int(self.apc.n)

    def cache(self):
        if self.record_voltages:
            self.rec.cache()
        to_cache = np.array(self._spikes.to_python())
        to_cache = to_cache[round_nearest(to_cache, N.dt_) > self.tstart + N.dt_]
        if self.__cache__ is not None and self.__cache__.size > 0:
            self.__cache__ = np.unique(np.concatenate([self.__cache__, to_cache]))
        else:
            self.__cache__ = to_cache

    def spikes(self):
        """Array of spike times > checking time."""
        if self.__cache__ is None:
            spikes = np.array(self._spikes.to_python())
            return spikes[spikes > self.time_f]
        else:
            return self.__cache__[self.__cache__ > self.time_f]

    def all_spikes(self):
        """Array of spike times."""
        return (
            np.array(self._spikes.to_python())
            if self.__cache__ is None
            else self.__cache__
        )

    @unitdispatch
    def set_time(self, t: "ms"):
        """Set time after which to check for APs."""
        self.time = t

    def clear(self):
        self.__cache__ = None
