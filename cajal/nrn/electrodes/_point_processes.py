from neuron import h, nrn

from cajal.nrn.electrodes._electrodes import Stimulator
from cajal.nrn.stimuli import Stimulus


class PointProcess:
    """
    Parameters
    ----------
    section : nrn.Section, nrn.Segment
        Section into which the current will be injected.
    segment : float (optional)
        Specify which compartment in the section the current will
        be inserted. By default 0.5.
    """

    def __init__(self, section, segment=0.5):
        if isinstance(section, nrn.Section):
            section = section(segment)
        self.compartment = section
        self.play_vec = self.t_vec = None


class PlayablePointProcess(PointProcess, Stimulator):
    """
    Generic wrapper for Neuron PointProcesses that support
    vector play.

    Parameters
    ----------
    section : nrn.Section, nrn.Segment
        Section into which the current will be injected.
    segment : float (optional)
        Specify which compartment in the section the current will
        be inserted. By default 0.5.
    stimulus : src.nrn.stimuli.Stimulus
        Parameterised stimulus to be delivered. By default None.
    """

    __units__ = None

    def __init__(self, section, segment=0.5, stimulus=None):
        PointProcess.__init__(self, section, segment)
        Stimulator.__init__(self, stimulus)

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
            self._stimulus.units = self.__units__
        else:
            raise TypeError("Supply the stimulus in the form of a Stimulus object.")

    def init(self, t):
        super().init(t)
        self.play_vec = h.Vector(self._ve_time)
        self.t_vec = h.Vector(t)


class IClamp(PlayablePointProcess):
    """
    Class for delivering intracellular current stimulation.
    Wraps the NEURON IClamp point-process.
    """

    __units__ = "nA"

    def __init__(self, section, segment=0.5, stimulus=None):
        PlayablePointProcess.__init__(self, section, segment, stimulus)
        self.pp = h.IClamp(self.compartment)
        self.pp.dur = 1e9
        self.pp.delay = 0

    def init(self, t):
        super().init(t)
        self.play_vec.play(getattr(self.pp, "_ref_amp"), self.t_vec, True)

    def __repr__(self):
        return repr(self.compartment) + " << IClamp << " + Stimulator.__repr__(self)


class VClamp(PlayablePointProcess):
    """
    Voltage Clamp class. Wraps the NEURON VClamp point-process.
    """

    __units__ = "mV"

    def __init__(self, section, segment=0.5, stimulus=None):
        PlayablePointProcess.__init__(self, section, segment, stimulus)
        self.pp = h.VClamp(self.compartment)
        self.pp.dur[0] = 1e9

    def init(self, t):
        super().init(t)
        self.play_vec.play(getattr(self.pp, "_ref_amp")[0], self.t_vec, True)

    def __repr__(self):
        return repr(self.compartment) + " << VClamp << " + Stimulator.__repr__(self)


class PyPlayable(PlayablePointProcess):
    def __init__(self, section, segment=0.5, stimulus=None):
        super(PyPlayable, self).__init__(section, segment, stimulus)
        self.compartment.sec.cell().ppappend(self)

    def advance(self, i):
        """Custom code to execute"""


class VSet(PyPlayable):
    __units__ = "mV"

    def __init__(self, section, segment=0.5, stimulus=None):
        super(VSet, self).__init__(section, segment, stimulus)
        self._ignore = stimulus._ignore if stimulus is not None else None

    def init(self, t):
        super().init(t)
        self._ignore = self.stimulus._ignore if self.stimulus is not None else None

    def advance(self, i):
        v = self.play_vec[i]
        if v == self._ignore:
            pass
        else:
            self.compartment.sec.v = v

    def __repr__(self):
        return repr(self.compartment) + " << VSet << " + Stimulator.__repr__(self)
