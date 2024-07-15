import abc
import copy
import inspect
import enum


import numpy as np
import scipy

from cajal.common.logging import logger
from cajal.exceptions import ArgumentError
from cajal.nrn.__backend import Backend as N
from cajal.nrn.specs import Specable, Spec, ContainerSpec, SpecList
from cajal.units import ms, kHz, radian
from cajal.units.utils import apply_units, unitdispatch, strip_units


class ModeKeys(enum.Enum):
    EXTRA = "extra"
    INTRA = "intra"


class SequenceComponent:
    @unitdispatch
    def __init__(self, stim, duration: "ms" = None):
        self.stim = stim
        self.duration = duration or N.tstop


class StimulusSpec(Spec):
    def __init__(self, *args, **kwargs):
        from cajal.nrn.electrodes import IClamp

        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_scale", 1)
        object.__setattr__(self, "_control", IClamp)
        object.__setattr__(self, "_ignore", None)

    def __rshift__(self, other):
        return other << self

    def repeat(self, freq, delay=0):
        return RepeatStimulusSpec(self, freq, delay)

    def iclamp(self):
        from cajal.nrn.electrodes import IClamp

        self._control = IClamp
        return self

    def vclamp(self):
        from cajal.nrn.electrodes import VClamp

        self._control = VClamp
        return self

    def vset(self):
        from cajal.nrn.electrodes import VSet

        self._control = VSet
        return self

    def ignore(self, val):
        self._ignore = val
        return self

    def build(self):
        built = super().build()
        built._scale = self._scale
        built._control = self._control
        built._ignore = self._ignore
        return built


class RepeatStimulusSpec(SpecList):
    def __init__(self, stim, freq, delay):
        repeat = ContainerSpec(freq=freq, delay=delay)
        super().__init__(stim, repeat)
        self._stim = stim
        self._repeat = repeat

    def __rshift__(self, other):
        return other << self

    def build(self):
        return Repeat(self._stim.build(), **self._repeat.build())

    def iclamp(self):
        self._stim.iclamp()
        return self

    def vclamp(self):
        self._stim.vclamp()
        return self

    def vset(self):
        self._stim.vset()
        return self

    def ignore(self, val):
        self._stim.ignore(val)
        return self


class SequenceComponentSpec(SpecList):
    def __init__(self, stim, duration):
        duration = ContainerSpec(duration=duration)
        super().__init__(stim, duration)
        self._stim = stim
        self._dur = duration

    def build(self):
        return SequenceComponent(self._stim.build(), self._dur.build())


class SequentialStimulusSpec(SpecList):
    def __init__(self, stims):
        super().__init__(*[SequenceComponentSpec(stim, dur) for stim, dur in stims])

    def build(self):
        return Sequential(super().build())


class InputType(enum.Enum):
    SUPERPOSITION = "Superposition"
    SEQUENTIAL = "Sequential"


superposition = InputType.SUPERPOSITION
sequential = InputType.SEQUENTIAL


class SuperPositioner:
    __slots__ = "source"

    def __init__(self, source):
        self.source = source

    def superposition(self, stims):
        return self.source.superposition(stims)

    def __lshift__(self, other):
        return self.superposition(other)


class Stimulus(abc.ABC, Specable):

    """Abstract class for custom stimulus API.

    In order to subclass Stimulus::

        class MyStimulus(Stimulus):
            def __init__(self, param1, param2 ...):
                super(MyStimulus, self).__init__()
                self.param1 = param1
                self.param2 = param2
                ...

            def timecourse(self, t):
                Define logic to construct timecourse from t,
                the vector of timesteps over which to calculate
                the stimulus timecourse and parameters defined
                in __init__ and return stimulus vector.

    The timecourse method may be defined with additional default keyword
    arguments, however the first and only positional arguments (apart from
    self) must be the vector of timesteps.

    The new stimulus can then be used with the core Model and
    optimisation interfaces.

    NOTE: If you want to use your new stimulus with the implemented
    threshold-finding apparatus, you should implement the dunder Python
    method __mul__ for your class, with instructions on how to scale your
    stimulus::

        class MyStimulus(Stimulus):
            def __init__(self, amp, ...):
                super(MyStimulus, self).__init__()
                self.amp = amp
                ...

            def timecourse(self, t):
                ...

            def __mul__(self, scale):
                return MyStimulus(self.amp*scale, ...)
    """

    name = None
    _spec = StimulusSpec

    def __init__(self, **kwargs):
        from cajal.nrn.electrodes import IClamp

        object.__setattr__(self, "_scale", 1)
        object.__setattr__(self, "_units", "mA")
        object.__setattr__(self, "_control", IClamp)
        object.__setattr__(self, "_ignore", None)
        self._check_timecourse_args()

    def _check_timecourse_args(self):
        fullargspec = inspect.getfullargspec(self.timecourse)
        if fullargspec.defaults:
            positional_args = fullargspec.args[1 : -len(fullargspec.defaults)]
        else:
            positional_args = fullargspec.args[1:]

        # self, t only can be positional.
        if len(positional_args) != 1:
            raise ArgumentError(
                "Timecourse method must only have a single positional "
                + "argument corresponding to the time over which to calculate "
                + "the stimulus course; found: "
                + str(positional_args)
                + "."
            )

    def __call__(self, t):
        units = self._units
        return apply_units(
            self._scale * strip_units(self.timecourse(apply_units(t, "ms"))), units
        )

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        self._units = units

    @property
    def control(self):
        return self._control

    @abc.abstractmethod
    def timecourse(self, t):
        """Generate the stimulus timecourse. Must be implemented by all
        subclasses."""

    @property
    def standard_timecourse(self):
        return self(np.arange(0, N.tstop, N.dt))

    @classmethod
    def parameter_units(cls):
        """Get dictionary of default units."""
        return cls.__init__.__annotations__

    def __repr__(self):
        return (
            f"Scale:{self._scale} :: "
            f"{self.__class__.__name__} "
            f"{self.parameters_dict_str()}"
        )

    def __rshift__(self, other):
        return other.__lshift__(self)

    def repeat(self, freq, delay=0):
        return Repeat(self, freq, delay)

    def plot(self, t=None, dpi=150):
        """Plot stimulus timecourse over tstop with timestep dt."""
        import matplotlib.pyplot as plt

        fig = plt.figure(dpi=dpi)
        if t is None:
            t = np.arange(0, N.tstop, N.dt)
        timecourse = self(t)
        plt.plot(t, timecourse)
        plt.xlabel("Time (ms)")
        plt.ylabel(f"Amplitude ({self._units})")
        plt.show()
        return fig

    def __imul__(self, other):
        self._scale *= other
        return self

    def __mul__(self, other):
        new = copy.deepcopy(self)
        new._scale *= other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Superposition):
            return Superposition(self, *other)
        if isinstance(other, Stimulus):
            return Superposition(self, other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Superposition):
            return Superposition(self, *[-1.0 * o for o in other])
        if isinstance(other, Stimulus):
            return Superposition(self, -1.0 * other)
        return NotImplemented

    def __neg__(self):
        return -1.0 * self

    def __setattr__(self, name: str, value) -> None:
        units = self.__dict__.get("_units")
        if units is None:
            raise AttributeError(
                "cannot assign parameter values to stimulus before "
                "Stimulus.__init__() call"
            )
        object.__setattr__(self, name, value)

    def iclamp(self):
        from cajal.nrn.electrodes import IClamp

        self._control = IClamp
        return self

    def vclamp(self):
        from cajal.nrn.electrodes import VClamp

        self._control = VClamp
        return self

    def vset(self):
        from cajal.nrn.electrodes import VSet

        self._control = VSet
        return self

    def ignore(self, val):
        self._ignore = val
        return self


class MultiStimulus:

    """Generic container for multiple Stimuli"""

    def __init__(self, *args):
        if not args or (len(args) == 1 and args[0] is None):
            self._stims = []
        elif len(args) == 1 and hasattr(args[0], "__iter__"):
            self._stims = list(args[0])
        else:
            self._stims = list(args)

    def __iter__(self):
        return iter(self._stims)

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return MultiStimulus([self._stims[i] for i in idx])
        elif isinstance(idx, int):
            return self._stims[idx]
        else:
            return MultiStimulus(self._stims[idx])

    def __setitem__(self, idx, item):
        self._stims[idx] = item

    def __len__(self):
        return len(self._stims)

    def __contains__(self, item):
        return item in self._stims

    def __repr__(self):
        return "".join([f"\n{repr(e)}" for e in self])


class Superposition(MultiStimulus, Stimulus):

    """Container for adding multiple stimuli together."""

    def __init__(self, *args):
        Stimulus.__init__(self)
        MultiStimulus.__init__(self, *args)

    def timecourse(self, t):
        return np.sum([e(t) for e in self], axis=0)

    def __add__(self, other):
        if isinstance(other, Superposition):
            return Superposition(*self, *other)
        if isinstance(other, Stimulus):
            return Superposition(*self, other)
        return NotImplemented


class Repeat(Stimulus):

    """
    Turn any stimulus into a periodic waveform.
    Access via stimulus.repeat(freq, delay).
    """

    @unitdispatch
    def __init__(self, stim, freq: "kHz", delay: "ms" = 0):
        super(Repeat, self).__init__()
        self.stim = stim
        self.freq = freq
        self.delay = delay
        self._ignore = stim._ignore

    @property
    def control(self):
        return self.stim._control

    def timecourse(self, t):
        tshift = t - self.delay
        rem = tshift % (1 / self.freq)
        return np.where(tshift < 0, 0, self.stim(rem))

    def __mul__(self, scale):
        return Repeat(self.stim * scale, self.freq, self.delay)

    def iclamp(self):
        self.stim.iclamp()
        return self

    def vclamp(self):
        self.stim.vclamp()
        return self

    def vset(self):
        self.stim.vset()
        return self

    def ignore(self, val):
        self.stim.ignore(val)
        self._ignore = val
        return self


class Sequential(Stimulus):

    """
    Stitch stimuli end-to-end.

    Parameters
    ----------
    stims: list
        Sequence of tuples of (Stimulus, duration). Duration will be converted to ms.
    """

    _spec = SequentialStimulusSpec

    def __init__(self, stims):
        super(Sequential, self).__init__()
        if not all(x._control == stims[0][0]._control for (x, dur) in stims):
            raise AssertionError(
                "All stimuli in Sequential must be of "
                "same type i.e. cannot IClamp then VClamp in same "
                "sequence."
            )
        self.stims = [SequenceComponent(stim, dur) for (stim, dur) in stims]
        self.rds = np.cumsum([c.duration for c in self.stims])
        self.chunks = apply_units([0, *self.rds[:-1]], "ms")

    def timecourse(self, t):
        def shift(t, chunk, dur):
            t = t - chunk
            t = t[np.where(np.logical_and(t >= 0, t < dur))]
            return t

        def overlaps(a, b):
            over = min(a[1], b[1]) - max(a[0], b[0])
            return over

        tcourses = []
        tstart, tend = t[0], t[-1]
        for component, chunk, rd in zip(self.stims, self.chunks, self.rds):
            if (
                overlaps((float(tstart), float(tend)), (float(chunk), float(rd)))
                >= N.dt
            ):
                tcourses.append(component.stim(shift(t, chunk, component.duration)))
        tcourses.append([])
        total = np.concatenate(tcourses)

        if tstart < 0:
            total = np.pad(total, (np.count_nonzero(t < 0), 0), constant_values=0)

        try:
            total = np.pad(total, (0, np.size(t) - np.size(total)), constant_values=0)
        except ValueError:
            total = total[: np.size(t)]

        assert np.size(total) == np.size(t)
        return strip_units(total)


class Timed(Stimulus):
    @unitdispatch
    def __init__(self, stim, times: "ms"):
        super(Timed, self).__init__()
        self.stim = stim
        self.times = times

    def _make_singular(self, time, t):
        chunk = t[t >= time]
        t_course = self.stim(chunk - time)
        out = np.zeros(len(t))
        out[t >= time] = t_course
        return out

    def timecourse(self, t):
        return sum([self._make_singular(time, t) for time in self.times])


class MonophasicPulse(Stimulus):
    """Monophasic Pulse.

    Parameters
    ----------
    amp : float
        Current Amplitude

    pw : float
        Pulse Width (ms). Be conscious of time discretisation.

    delay : float
        Delay (ms). Be conscious of time discretisation.
    """

    name = "monophasic_pulse"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms", offset=0):
        super(MonophasicPulse, self).__init__()
        self.amp = amp
        self.pw = pw
        self.delay = delay
        self.offset = offset

    def timecourse(self, t):
        tstop = self.pw + self.delay
        if tstop >= t[-1]:
            logger.info("The pulse ends after the simulation end time.")
        return (
            self.amp * np.logical_and(t > self.delay, t <= tstop).astype(float)
            + self.offset
        )

    def __mul__(self, scale):
        return MonophasicPulse(self.amp * scale, self.pw, self.delay, self.offset)


class Sinusoid(Stimulus):
    """Sinusoid.

    Parameters
    ----------
    amp : float
        Current Amplitude

    freq : float
        Frequency (kHz). Be conscious of time discretisation.

    delay : float
        Delay (ms). Be conscious of time discretisation.

    phase : float
        Phase shift.

    Notes
    -----
    Sinusoid follows the formula:

    amp * sin(2 * pi * freq * t + phase)
    """

    name = "sinusoid"

    @unitdispatch
    def __init__(self, amp, freq: "kHz", delay: "ms", phase: "radian", offset=0):
        super(Sinusoid, self).__init__()
        self.amp = amp
        self.freq = freq
        self.delay = delay
        self.phase = phase
        self.offset = offset

    def timecourse(self, t):
        sig = self.amp * np.sin(2 * np.pi * self.freq * (t - self.delay) + self.phase_)
        sig[t < self.delay] = 0
        return sig + self.offset

    def __mul__(self, scale):
        return Sinusoid(
            self.amp * scale, self.freq, self.delay, self.phase_, self.offset
        )


class SineWave(Sinusoid):
    """Sine Wave.

    Parameters
    ----------
    amp : float
        Current Amplitude

    freq : float
        Frequency (kHz). Be conscious of time discretisation.

    delay : float
        Delay (ms). Be conscious of time discretisation.
    """

    name = "sine_wave"

    @unitdispatch
    def __init__(self, amp, freq: "kHz", delay: "ms", offset=0):
        super(SineWave, self).__init__(amp, freq, delay, 0, offset)

    def __mul__(self, scale):
        return SineWave(self.amp * scale, self.freq, self.delay, self.offset)


class CosineWave(Sinusoid):
    """Cosine Wave.

    Parameters
    ----------
    amp : float
        Current Amplitude

    freq : float
        Frequency (kHz). Be conscious of time discretisation.

    delay : float
        Delay (ms). Be conscious of time discretisation.
    """

    name = "cosine_wave"

    @unitdispatch
    def __init__(self, amp, freq: "kHz", delay: "ms", offset=0):
        super(CosineWave, self).__init__(amp, freq, delay, np.pi / 2, offset)

    def __mul__(self, scale):
        return CosineWave(self.amp * scale, self.freq, self.delay, self.offset)


class Step(Stimulus):
    """Step function."""

    name = "step"

    @unitdispatch
    def __init__(self, amp, delay: "ms"):
        super(Step, self).__init__()
        self.amp = amp
        self.delay = delay

    def timecourse(self, t):
        sig = np.zeros_like(t)
        sig[t > self.delay] += self.amp
        return sig

    def __mul__(self, scale):
        return Step(self.amp * scale, self.delay)


class SinusoidPulse(Stimulus):
    """Arbitrary truncated sinusoid."""

    name = "sinusoid_pulse"

    @unitdispatch
    def __init__(
        self, amp, freq: "kHz", phase: "radian", pw: "ms", delay: "ms", offset
    ):
        super(SinusoidPulse, self).__init__()
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.pw = pw
        self.delay = delay
        self.offset = offset

    def timecourse(self, t):
        x = t - self.delay
        sig = self.amp * np.sin(2 * np.pi * self.freq * x + self.phase_) + self.offset
        sig[np.logical_or(t < self.delay, t > self.delay + self.pw)] = 0
        return sig

    def __mul__(self, scale):
        return SinusoidPulse(
            self.amp * scale, self.freq, self.phase, self.pw, self.delay, self.offset
        )


class SinePulse(SinusoidPulse):
    """Truncated sine (between delay and delay + pw)"""

    name = "sine_pulse"

    @unitdispatch
    def __init__(self, amp, freq: "kHz", pw: "ms", delay: "ms", offset):
        super(SinePulse, self).__init__(amp, freq, 0, pw, delay, offset)

    def __mul__(self, scale):
        return SinePulse(self.amp * scale, self.freq, self.pw, self.delay, self.offset)


class CosinePulse(SinusoidPulse):
    """Truncated cosine (between delay and delay + pw)"""

    name = "cosine_pulse"

    @unitdispatch
    def __init__(self, amp, freq: "kHz", pw: "ms", delay: "ms", offset):
        super(CosinePulse, self).__init__(amp, freq, np.pi / 2, pw, delay, offset)

    def __mul__(self, scale):
        return CosinePulse(
            self.amp * scale, self.freq, self.pw, self.delay, self.offset
        )


class Arbitrary(Stimulus):
    """Wrapper for arbitrary array. Interpolates values over total simulation
    timecourse based on tpoints. By default, assumes that the input array is
    sorted over tpoints for speed."""

    name = "arbitrary"

    @unitdispatch
    def __init__(
        self, course, tpoints: "ms", method="linear", fill_value=0, assume_sorted=True
    ):
        super(Arbitrary, self).__init__()
        self.course = course
        self.tpoints = tpoints
        self.method = method
        self.fill_value = fill_value
        self.assume_sorted = assume_sorted
        self.f = scipy.interpolate.interp1d(
            tpoints,
            course,
            kind=method,
            fill_value=fill_value,
            assume_sorted=assume_sorted,
            bounds_error=False,
        )

    def timecourse(self, t):
        return self.f(t)


class ArbitraryCB(Arbitrary):
    name = "arbitrarycb"

    @unitdispatch
    def __init__(
        self, course, tpoints: "ms", method="linear", fill_value=0, assume_sorted=True
    ):
        course = np.asarray(course) - np.mean(course)
        super(ArbitraryCB, self).__init__(
            course, tpoints, method, fill_value, assume_sorted
        )

    def timecourse(self, t):
        return self.f(t)


class Biphasic(Stimulus):
    name = "biphasic"

    @unitdispatch
    def __init__(self, amp1, pw1: "ms", amp2, pw2: "ms", delay: "ms", offset=0):
        super(Biphasic, self).__init__()
        self.amp1 = amp1
        self.amp2 = amp2
        self.pw1 = pw1
        self.pw2 = pw2
        self.delay = delay
        self.offset = offset

    def timecourse(self, t):
        p1 = self.amp1 * np.logical_and(
            t > self.delay, t <= self.delay + self.pw1
        ).astype(float)
        p2 = self.amp2 * np.logical_and(
            t > self.delay + self.pw1, t <= self.delay + self.pw1 + self.pw2
        ).astype(float)
        return p1 + p2 + self.offset

    def __mul__(self, scale):
        return Biphasic(
            self.amp1 * scale,
            self.pw1,
            self.amp2 * scale,
            self.pw2,
            self.delay,
            self.offset,
        )


class SymmetricBiphasic(Biphasic):
    name = "symmetric_biphasic"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms"):
        super(SymmetricBiphasic, self).__init__(amp, pw, -amp, pw, delay)
        self.amp = amp
        self.pw = pw

    def __mul__(self, scale):
        return SymmetricBiphasic(self.amp * scale, self.pw, self.delay)


class ChargeBalancedBiphasic(Biphasic):
    name = "charge_balanced_biphasic"

    @unitdispatch
    def __init__(self, amp, pw1: "ms", pw2: "ms", delay: "ms"):
        amp2 = -1 * (amp * float(pw1)) / float(pw2)
        super(ChargeBalancedBiphasic, self).__init__(amp, pw1, amp2, pw2, delay)
        self.amp = amp

    def __mul__(self, scale):
        return ChargeBalancedBiphasic(self.amp * scale, self.pw1, self.pw2, self.delay)


class IncreasingTriangular(Stimulus):
    name = "increasing_triangular"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms"):
        super(IncreasingTriangular, self).__init__()
        self.amp = amp
        self.pw = pw
        self.delay = delay

    def timecourse(self, t):
        out = np.zeros(len(t))
        window = np.logical_and(t > self.delay, t <= self.delay + self.pw)
        out[window] = self.amp
        out[window] *= (t[window] - self.delay) / self.pw
        return out

    def __mul__(self, scale):
        return IncreasingTriangular(self.amp * scale, self.pw, self.delay)


class DecreasingTriangular(Stimulus):
    name = "decreasing_triangular"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms"):
        super(DecreasingTriangular, self).__init__()
        self.amp = amp
        self.pw = pw
        self.delay = delay

    def timecourse(self, t):
        out = np.zeros(len(t))
        window = np.logical_and(t > self.delay, t <= self.delay + self.pw)
        out[window] = self.amp_
        out[window] *= ((t[window] - self.delay) / self.pw)[::-1]
        return out

    def __mul__(self, scale):
        return DecreasingTriangular(self.amp * scale, self.pw, self.delay)


class IncreasingExponential(Stimulus):
    name = "increasing_exponential"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms", tau: "ms"):
        super(IncreasingExponential, self).__init__()
        self.amp = amp
        self.pw = pw
        self.delay = delay
        self.tau = tau

    def timecourse(self, t):
        n = self.amp * (
            (np.exp((t - self.delay) / self.tau) - 1) / (np.exp(self.pw / self.tau) - 1)
        )
        mask = np.logical_and(t >= self.delay, t <= self.delay + self.pw)
        n[~mask] = 0
        return n

    def __mul__(self, scale):
        return IncreasingExponential(self.amp * scale, self.pw, self.delay, self.tau)


class DecreasingExponential(Stimulus):
    name = "increasing_exponential"

    @unitdispatch
    def __init__(self, amp, pw: "ms", delay: "ms", tau: "ms"):
        super(DecreasingExponential, self).__init__()
        self.amp = amp
        self.pw = pw
        self.delay = delay
        self.tau = tau

    def timecourse(self, t):
        n = self.amp * ((np.exp(t / self.tau) - 1) / (np.exp(self.pw / self.tau) - 1))
        out = np.zeros(len(t))
        out[np.logical_and(t >= self.delay, t <= self.delay + self.pw)] = n[
            t <= self.pw
        ][::-1]
        return out

    def __mul__(self, scale):
        return DecreasingExponential(self.amp * scale, self.pw, self.delay, self.tau)


class Gaussian(Stimulus):
    name = "gaussian"

    @unitdispatch
    def __init__(self, amp, mu: "ms", sigma: "ms"):
        super(Gaussian, self).__init__()
        self.amp = amp
        self.mu = mu
        self.sigma = sigma

    def timecourse(self, t):
        return self.amp * np.exp(-((t - self.mu) ** 2) / (2 * self.sigma) ** 2)

    def __mul__(self, scale):
        return Gaussian(self.amp * scale, self.mu, self.sigma)
