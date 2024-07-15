"""Point sources."""

from abc import abstractmethod

import numpy as np

from cajal.nrn.sources.base import Source
from cajal.units import unitdispatch, strip_units, um, ohm, cm, rad
from cajal.nrn.__backend import Backend as N


class Point(Source):
    """Generic base pointsource potential field engine."""

    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__position = (x, y, z)
        Source.__init__(self)

    @property
    def x(self):
        """x coordinate"""
        return self.__x

    @property
    def y(self):
        """y coordinate"""
        return self.__y

    @property
    def z(self):
        """z coordinate"""
        return self.__z

    @property
    def position(self):
        """(x, y, z)"""
        return self.__position

    @abstractmethod
    def ve_space_array(self, loc):
        """Calculate voltage distributions produced by a 1mA current source
        at this electrode location at every location in loc.

        Parameters
        ----------
        loc : array
            Array of section locations.
        """

    def init_ve_space(self):
        """Populate and return a list of voltage distributions produced by a
        a 1mA current source at this electrode location at every section
        in every axon in self.axons.
        """
        out = [self.ve_space_array(a.sec_loc) for a in self.axons]
        return out


# pylint: disable=bad-whitespace
class IsotropicPoint(Point):
    """Homogeneous, isotropic medium."""

    name = "isotropic_point"
    _valid_medium = {"infinite": 4, "semi_infinite": 2}

    @unitdispatch
    def __init__(
        self, x: "um", y: "um", z: "um", rhoe: "ohm*cm" = None, medium="infinite"
    ):
        Point.__init__(self, x, y, z)
        self.rhoe = rhoe or N.rhoe
        self.den = self._valid_medium.get(medium)
        self.medium = medium
        if self.den is None:
            raise ValueError(f"{medium} is not a valid medium description.")

    def ve_space_array(self, loc):
        r = np.sqrt(np.sum(np.square(loc - self.position_), axis=-1))
        ve_space = (10**4) * (self.rhoe_ / (self.den * np.pi * r))
        return ve_space


class IsotropicPolar(IsotropicPoint):
    """Homogeneous, isotropic medium. Polar coordinates."""

    name = "isotropic_point_polar"

    @unitdispatch
    def __init__(
        self, r: "um", theta: "rad", y: "um", rhoe: "ohm*cm" = None, medium="infinite"
    ):
        self.r = r
        self.theta = theta
        super().__init__(r * np.cos(theta), y, r * np.sin(theta), rhoe, medium)


class AnisotropicPoint(Point):
    """Homogeneous, anisotropic medium."""

    name = "anisotropic_point"
    _valid_medium = {"infinite": 4, "semi_infinite": 2}

    @unitdispatch
    def __init__(
        self, x: "um", y: "um", z: "um", rhoe: "ohm*cm" = None, medium="infinite"
    ):
        Point.__init__(self, x, y, z)
        self.rhoe = rhoe if rhoe is not None else N.rhoe
        rhoe = strip_units(self.rhoe)
        self.mod = 1 / np.array(
            [rhoe[1] * rhoe[2], rhoe[0] * rhoe[2], rhoe[0] * rhoe[1]]
        )
        self.den = self._valid_medium.get(medium)
        self.medium = medium
        if self.den is None:
            raise ValueError(f"{medium} is not a valid medium description.")

    def ve_space_array(self, loc):
        r = np.sqrt(np.sum(np.square(loc - self.position_) * self.mod, axis=-1))
        ve_space = (10**4) / (self.den * np.pi * r)
        return ve_space


class AnisotropicPolar(AnisotropicPoint):
    """Homogeneous, anisotropic medium. Polar coordinates."""

    name = "anisotropic_point_polar"

    @unitdispatch
    def __init__(
        self, r: "um", theta: "rad", y: "um", rhoe: "ohm*cm" = None, medium="infinite"
    ):
        self.r = r
        self.theta = theta
        super().__init__(r * np.cos(theta), y, r * np.sin(theta), rhoe, medium)
