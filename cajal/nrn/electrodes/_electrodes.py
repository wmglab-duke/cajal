"""The `electrodes` module provides the methods and classes to
construct current sources in your models.
"""

import numpy as np

from cajal.common.logging import logger
from cajal.nrn import Backend as N
from cajal.nrn.stimuli import Stimulus, Superposition
from cajal.nrn.sources import get_source_class
from cajal.nrn.specs import SpecList


class Stimulator:
    """Houses the stimulus waveform component of a source object."""

    def __init__(self, stimulus=None):
        self._stimulus = None
        self._ve_time = None
        self.stimulus = stimulus
        self.t = None

        self._t_rec = None
        self._ve_time_rec = None

    @property
    def stimulus(self):
        """Stimulus waveform object."""
        return self._stimulus

    @stimulus.setter
    def stimulus(self, stimulus):
        if stimulus is None:
            self._stimulus = None
        elif isinstance(stimulus, Stimulus):
            self._stimulus = stimulus
            self._stimulus.units = "mA"
        else:
            raise TypeError("Supply the stimulus in the form of a Stimulus object.")

    @property
    def initialised(self):
        return self._stimulus is not None

    def init(self, t):
        """Initialise the stimulus over the given timecourse."""
        if self.stimulus is not None:
            self._ve_time = self.stimulus(t)  # pylint: disable=not-callable
        else:
            self._ve_time = np.zeros_like(t)
        self.t = t
        self._record()

    def _record(self):
        if self.t[0] <= 0:
            self._t_rec = self.t.copy()
            self._ve_time_rec = self._ve_time.copy()
        else:
            inds = self._t_rec < self.t[0]
            self._t_rec = np.concatenate((self._t_rec[inds], self.t))
            self._ve_time_rec = np.concatenate((self._ve_time_rec[inds], self._ve_time))

    def set_stimulus(self, stimulus):
        """Set stimulus waveform."""
        self.stimulus = stimulus
        return self

    def __lshift__(self, other):
        if isinstance(other, Stimulus):
            self.stimulus = other
            return self
        return NotImplemented

    @property
    def ve_time(self):
        """Retrieve stimulus timecourse vector (mA)."""
        return self._ve_time

    def plot_ve_time(self, dpi=100, figsize=(7, 5), ylim=None):
        import matplotlib.pyplot as plt

        """Plot stimulus timecourse."""
        if self._t_rec is None:
            logger.info("Simulation has not been run.")
            return None

        fig = plt.figure(dpi=dpi, figsize=figsize)
        inds = self._t_rec <= N.t
        plt.plot(self._t_rec[inds], self._ve_time_rec[inds])
        plt.xlabel("t (ms)")
        plt.ylabel("I (mA)")
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()

        return fig

    def __repr__(self):
        return repr(self.stimulus)


def ExtraStim(source, *args, **kwargs):
    """Construct extracellular current source."""

    if source is None or isinstance(source, ExtraStimList):
        return source

    Source = get_source_class(source)

    if Source.__name__ == "_Extra":
        return source

    class _Extra(Stimulator, Source):
        def __init__(self, *a, **kw):
            Stimulator.__init__(self)
            Source.__init__(self, *a, **kw)

        def __repr__(self):
            return f"{Source.__name__} << {Stimulator.__repr__(self)}"

        def ensure_geometric_relations(self, axons):
            self.load_axons(axons)

        @classmethod
        def from_source(cls, source):
            return cls(**source.parameters_dict())

    if isinstance(source, Source):
        return _Extra.from_source(source)

    return _Extra(*args, **kwargs)


class StimList:
    """Generic container for electrodes."""

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
            return StimList([self._stims[i] for i in idx])
        elif isinstance(idx, int):
            return self._stims[idx]
        else:
            return StimList(self._stims[idx])

    def __setitem__(self, idx, item):
        self._stims[idx] = item

    def __len__(self):
        return len(self._stims)

    def __contains__(self, item):
        return item in self._stims

    def __repr__(self):
        return self._stims.__repr__()

    def __lshift__(self, other):
        """
        Sequences of stimuli can be piped into a StimList (in which case
        len(other) == len(self), otherwise if a single stimulus is supplied,
        every source in the StimList container receives that stimulus.
        """
        if hasattr(other, "__iter__"):
            if len(other) == len(self):
                for elec, stim in zip(self, other):
                    elec << stim
            else:
                raise ValueError(
                    f"Inappropriate # of stimuli: {len(other)} supplied, "
                    f"{len(self)} expected."
                )
        else:
            for elec in self:
                elec << other
        return self

    def append(self, item):
        """Append to list."""
        self._stims.append(item)

    def extend(self, extras):
        """Extend list."""
        self._stims.extend(extras)

    def __add__(self, other):
        return StimList(self._stims + other._stims)

    @property
    def initialised(self):
        return all([e.initialised for e in self])

    def init(self, t):
        if not self._stims:
            return
        if not self.initialised:
            logger.warning(
                "Not all extracellular electrodes have been assigned " "a stimulus."
            )
        for e in self:
            e.init(t)


class ExtraStimList(StimList):
    """Wrap extracellular sources in a single list. This class handles
    vectorised calculation of the full set of voltages applied to every axon
    model at every timestep of the simulation, granting performance
    advantages over iterating over every section in a for loop."""

    def __init__(self, *args):
        args = [ExtraStim(a) for a in args] if args else [None]
        super(ExtraStimList, self).__init__(*args)
        self.Ve = None
        self.n_axons = None

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return ExtraStimList([self._stims[i] for i in idx])
        elif isinstance(idx, int):
            return self._stims[idx]
        else:
            return ExtraStimList(self._stims[idx])

    @property
    def ve_space(self):
        return [np.vstack([e.ve_space[i] for e in self]) for i in range(self.n_axons)]

    @property
    def ve_time(self):
        return np.vstack([e.ve_time for e in self])

    def ensure_geometric_relations(self, axons):
        self.n_axons = len(axons)
        for e in self:
            e.ensure_geometric_relations(axons)

    def init(self, t):
        if not self._stims:
            return
        super().init(t)
        self.Ve = [np.dot(self.ve_time.T, i) for i in self.ve_space]


class ElectrodeArray(ExtraStimList):
    def __init__(self, *args, weights=None):
        super(ElectrodeArray, self).__init__(*args)
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is not None:
            if not len(weights) == len(self):
                raise ValueError(
                    "Number of weights and number of electrodes in array "
                    + "do not match."
                )
            for e, w in zip(self, weights):
                e.scale = w
            self._weights = weights

    def __lshift__(self, other):
        for e in self:
            e << other
        return self

    def combined_ve_space(self, i):
        return np.sum(self.ve_space[i], axis=0)

    @classmethod
    def SPEC(cls, source_specs, weights=None):
        return ElectrodeArraySpec(source_specs, weights=weights)


# -- Specs --


class ExtraStimSpec(SpecList):
    __slots__ = "source_spec", "stim_spec", "superposition"

    def __init__(self, source_spec, stim_spec, superposition=False, repeat=None):
        if isinstance(stim_spec, list):
            super(ExtraStimSpec, self).__init__(source_spec, *stim_spec)
        else:
            super(ExtraStimSpec, self).__init__(source_spec, stim_spec)
        self.source_spec = source_spec
        self.stim_spec = stim_spec
        self.superposition = superposition

    def build(self):
        source, stims = self.source_spec.build(), None
        if isinstance(self.stim_spec, list):
            stims = [s.build() for s in self.stim_spec]
            if self.superposition:
                stims = Superposition(stims)
        else:
            stims = self.stim_spec.build()
        return source << stims

    def __repr__(self):
        if isinstance(self.stim_spec, list):
            return f"{self.source_spec}" + "".join(
                [f"\n\t|--- << {sp}" for sp in self.stim_spec]
            )
        return f"{self.source_spec}\n\t|--- << {self.stim_spec}"


class ElectrodeArraySpec(SpecList):
    def __init__(self, source_specs, weights=None):
        if weights:
            assert len(weights) == len(source_specs)
        super(ElectrodeArraySpec, self).__init__(*source_specs)
        self.source_specs = source_specs
        self.weights = weights
        self.stim_spec = None

    def build(self):
        try:
            array = ElectrodeArray(
                *[spec.build() for spec in self.source_specs], weights=self.weights
            )
            return array << self.stim_spec.build()
        except AttributeError:
            raise AttributeError(
                "ElectrodeArray has not been specified a stimulus."
            ) from None

    def set_stimulus(self, stimspec):
        if self.stim_spec is not None:
            self._specs.pop()
            self._mutable_specs.pop()
        self.append(stimspec)
        self.stim_spec = stimspec
        return self

    def __lshift__(self, other):
        from cajal.nrn.stimuli import StimulusSpec

        if isinstance(other, StimulusSpec):
            return self.set_stimulus(other)
        super().__lshift__(other)
