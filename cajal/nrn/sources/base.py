import numpy as np

from cajal.nrn.specs import Specable, Spec
from cajal.nrn.stimuli import superposition, SuperPositioner, Superposition


# -- specs --
class SourceSpec(Spec):
    def __init__(self, *args, **kwargs):
        super(SourceSpec, self).__init__(*args, **kwargs)
        self._superposition = False

    def __lshift__(self, other):
        from cajal.nrn.electrodes import ExtraStimSpec
        from cajal.nrn.stimuli import StimulusSpec

        if other is superposition:
            return self.superposition()
        if isinstance(other, StimulusSpec) or all(
            [isinstance(item, StimulusSpec) for item in other]
        ):
            return ExtraStimSpec(self, other, self._superposition)
        super().__lshift__(other)

    def superposition(self):
        self._superposition = True
        return self


# -- core implementation --
class Source(Specable):

    """
    Abstract base class for field potential calculation.
    """

    name = None
    _spec = SourceSpec

    def __init__(self):
        self.axons = []
        self.n_axons = None
        self._scale = None
        self.ve_space = []

    def load_axons(self, axons):
        """Load axons into engine."""
        if (self.axons is axons) or (axons is None):
            pass
        elif isinstance(axons, list):
            self.axons = axons
            self.n_axons = len(axons)
            self._ve_space_wrapper()
        else:
            raise TypeError("Axons must be supplied as a list.")
        return self

    def _ve_space_wrapper(self):
        if not self.axons:
            return None
        if self._scale is not None:
            out = [ve * self._scale for ve in self.init_ve_space()]
        else:
            out = self.init_ve_space()
        if self.n_axons != len(out):
            raise ValueError(
                "The number of ve_space arrays generated does not agree with"
                + " the number of axons."
            )
        if out and self.axons:
            for ve, ax in zip(out, self.axons):
                if np.size(ve) != len(ax):
                    raise ValueError(
                        "There is not a transfer impedance for every "
                        + "section in {}".format(ax)
                    )
        self.ve_space = out
        return None

    def __lshift__(self, other):
        from cajal.nrn.electrodes import ExtraStim
        from cajal.nrn.stimuli import Stimulus

        if other is superposition:
            return SuperPositioner(self)
        if isinstance(other, Stimulus):
            return ExtraStim(self) << other
        return NotImplemented

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def superposition(self, stims):
        return self << Superposition(stims)

    def init_ve_space(self):
        """Populate and return a list of voltage distributions produced by a
        a 1mA current source at this electrode location at every section
        in every axon in self.axons.
        """
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__} : {self.parameters_dict()}"
