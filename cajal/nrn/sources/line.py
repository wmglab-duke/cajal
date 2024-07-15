"""
Line Sources. Not Point Sources.
"""

from abc import abstractmethod

import numpy as np

from cajal.nrn import Backend as N
from cajal.nrn.sources.base import Source
from cajal.units import unitdispatch, ohm, cm


class Line(Source):
    """
    Holt, G.R., 1998. A critical reexamination of some assumptions and
    implications of cable theory in neurobiology (phd). California Institute
    of Technology. https://doi.org/Holt, Gary R.  (1998)  A critical
    reexamination of some assumptions and implications of cable theory in
    neurobiology.  Dissertation (Ph.D.), California Institute of Technology.
    doi:10.7907/HPPC-S237.
    https://resolver.caltech.edu/CaltechETD:etd-09122006-135415
    """

    _valid_medium = {"infinite": 4, "semi_infinite": 2}

    @unitdispatch
    def __init__(self, xyz=None, rhoe: "ohm*cm" = None, medium="infinite"):
        super(Line, self).__init__()

        if medium not in self._valid_medium:
            raise ValueError(
                "The given medium is not recognised. Choose from: {}".format(
                    list(self._valid_medium.keys())
                )
            )

        self.den = self._valid_medium[medium]
        self.xyz = xyz or self.generate_trajectory()
        self.rhoe = rhoe or N.rhoe
        self.start = self.xyz[:-1, :]
        self.end = self.xyz[1:, :]
        self.diff = np.diff(self.xyz, axis=0)
        self.ds = np.sqrt(np.einsum("ij, ij -> i", self.diff, self.diff))
        self.frac = self.ds / np.sum(self.ds)

    def init_ve_space(self):
        return [self.ve_space_array(a.sec_loc) for a in self.axons]

    def ve_space_array(self, loc):
        """Generate potentials at locations in loc."""
        loc = np.expand_dims(loc, 1)
        h = np.einsum("ijk, jk -> ij", loc - self.end, self.diff) / self.ds
        r2 = np.abs(np.sum(np.square(loc - self.start), axis=-1) - h**2)
        ll = h + self.ds

        cond1 = np.where(np.logical_and(h < 0, ll < 0))
        cond2 = np.where(np.logical_and(h < 0, ll >= 0))
        cond3 = np.where(np.logical_and(h >= 0, ll >= 0))

        phi = np.zeros((loc.shape[0], self.frac.shape[0]))
        phi[cond1] = self._condition_1(h[cond1], ll[cond1], r2[cond1])
        phi[cond2] = self._condition_2(h[cond2], ll[cond2], r2[cond2])
        phi[cond3] = self._condition_2(h[cond3], ll[cond3], r2[cond3])

        phi = (
            10**4
            * np.sum(phi * self.rhoe_ * self.frac / self.ds, axis=-1)
            / (self.den * np.pi)
        )

        return phi

    @staticmethod
    def _condition_1(h, ll, r2):
        """h < 0, l < 0"""
        return np.log((np.sqrt(h**2 + r2) - h) / (np.sqrt(ll**2 + r2) - ll))

    @staticmethod
    def _condition_2(h, ll, r2):
        """h < 0, l > 0"""
        return np.log((np.sqrt(h**2 + r2) - h) * (ll + np.sqrt(ll**2 + r2)) / r2)

    @staticmethod
    def _condition_3(h, ll, r2):
        """h > 0, l > 0"""
        return np.log((ll + np.sqrt(ll**2 + r2)) / (np.sqrt(h**2 + r2) + h))

    @abstractmethod
    def generate_trajectory(self):
        """Generate the set of (x,y,z) coordinates that defines the line path."""
        return self.xyz


class Arbitrary(Line):
    name = "arbitrary"

    def __init__(self, xyz, **kwargs):
        super().__init__(xyz, **kwargs)

    def generate_trajectory(self):
        super().generate_trajectory()


class Helix(Line):
    """Construct a helical line source."""

    name = "helix"

    def __init__(
        self,
        y_start,
        y_end,
        radius,
        orbits,
        phase,
        samples,
        rhoe=None,
        medium="infinite",
    ):
        self.y_start = y_start
        self.y_end = y_end
        self.radius = radius
        self.orbits = orbits
        self.samples = samples
        self.phase = phase
        super(Helix, self).__init__(rhoe=rhoe, medium=medium)

    def generate_trajectory(self):
        theta_max = self.orbits * 2 * np.pi
        theta = np.linspace(0, theta_max, self.samples)
        y = np.linspace(self.y_start, self.y_end, self.samples)
        z = self.radius * np.sin(theta + self.phase)
        x = self.radius * np.cos(theta + self.phase)
        return np.stack((x, y, z), axis=1)


class Arc(Helix):
    """Construct an arc line source."""

    name = "arc"

    def __init__(self, y, radius, orbit, phase, samples, rhoe=None, medium="infinite"):
        if orbit > 1 or orbit < 0:
            raise ValueError(
                "An arc must have a positive orbit and cannot "
                + "consist of more than one full orbit."
            )
        super().__init__(y, y, radius, orbit, phase, samples, rhoe, medium)


class Line3D(Line):
    """Construct a straight line source in 3D."""

    name = "line3d"

    def __init__(self, start, end, samples, rhoe=None, medium="infinite"):
        self.start = start
        self.end = end
        self.samples = samples
        super(Line3D, self).__init__(rhoe=rhoe, medium=medium)

    def generate_trajectory(self):
        x_ = np.linspace(self.start[0], self.end[0], self.samples)
        y_ = np.linspace(self.start[1], self.end[1], self.samples)
        z_ = np.linspace(self.start[2], self.end[2], self.samples)
        return np.stack((x_, y_, z_), axis=1)
