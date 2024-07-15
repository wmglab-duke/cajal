from neuron import nrn


from cajal.nrn.stimuli import Stimulus


class Section(nrn.Section):
    def __call__(self, value):
        return Segment(self, value)

    def __lshift__(self, other):
        if isinstance(other, Stimulus):
            return other.control(self) << other
        return NotImplemented

    @property
    def mechanisms(self):
        return self(0.5).mechanisms


class Segment(nrn.Segment):
    def __lshift__(self, other):
        if isinstance(other, Stimulus):
            return other.control(self) << other
        return NotImplemented

    @property
    def mechanisms(self):
        return [mech for mech in self]
