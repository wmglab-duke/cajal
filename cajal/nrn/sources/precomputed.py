"""Precomputed fields."""

import numpy as np
from scipy.interpolate import interp1d

from cajal.units import Unit, mV
from .base import Source


class PreComputed(Source):
    """Generic abstract base class for PreComputed field potential sources."""

    def __init__(self, data, in_memory=True, is_numpy=False, units="mV", options=None):
        super(PreComputed, self).__init__()
        self._fem_scale = Unit(units).get_conversion_factor(mV)[0]
        if in_memory:
            data = np.asarray(data)
            self.data = np.atleast_2d(data) if data.dtype.kind in "biufc" else data
            self.get = getattr(self, "get_in_memory")
        else:
            self.data = data
            self.get = getattr(self, "get_from_disk")
            if is_numpy:
                self.load = np.load
            else:
                self.load = np.loadtxt
            self.options = options or {}

    def init_ve_space(self):
        """Populate and return a list of voltage distributions produced by a
        a 1mA current source at this electrode location at every section
        in every axon in self.axons.
        """
        raise NotImplementedError()

    def get_in_memory(self, gid):
        return self._fem_scale * np.asarray(self.data[gid])

    def get_from_disk(self, gid):
        loc = self.data.format(gid=gid)
        vals = self.load(loc, **self.options)
        return self._fem_scale * vals


class PreComputedExact(PreComputed):
    """Use PreComputed data with exact values for every section in every model."""

    name = "pre_computed_exact"

    def init_ve_space(self):
        out = [self.get(axon.gid) for axon in self.axons]
        return out


class PreComputedInterpolate1D(PreComputed):
    """Use PreComputed data sampled along the y-axis at different
    (x, z) locations."""

    name = "pre_computed_interpolate_1d"

    def __init__(
        self,
        data=None,
        y=None,
        method="quadratic",
        fill_value="extrapolate",
        truncate=None,
        in_memory=True,
        is_numpy=False,
        units="mV",
        options=None,
    ):
        super(PreComputedInterpolate1D, self).__init__(
            data, in_memory, is_numpy, units, options
        )
        self.x = np.atleast_2d(np.asarray(y))
        self.method = method
        self.use_point_source = False
        if truncate is not None and truncate > 0.4:
            raise ValueError("Cannot truncate by more than 40%.")
        self.truncate = truncate
        self.fill_value = fill_value
        self.fv = fill_value
        if fill_value == "point_source":
            self.fv = 0
            self.use_point_source = True

    def init_ve_space(self):
        out = []
        for a in self.axons:
            try:
                x = self.x[a.gid]
            except IndexError:
                x = self.x[0]
            out.append(self.interpolate(self.get(a.gid), x, a))
        return out

    def interpolate(self, fem_vec, y_vec, axon):
        """Interpolate voltage values along axon."""
        y_points = axon.y_loc
        return self._interpolate(fem_vec, y_vec, y_points)

    def _interpolate(self, fem_vec, y_vec, y_points):
        if self.truncate:
            t_n = int(len(y_vec) * self.truncate)
            fem_vec = fem_vec[t_n:-t_n]
            y_vec = y_vec[t_n:-t_n]
        if self.use_point_source:
            bounds_error = False
        else:
            bounds_error = None
        interp_func = interp1d(
            y_vec,
            fem_vec,
            kind=self.method,
            assume_sorted=True,
            fill_value=self.fv,
            bounds_error=bounds_error,
        )
        interp = interp_func(y_points)
        if self.use_point_source:
            self.point_source_fill(y_vec, y_points, interp)
        return interp

    def point_source_fill(self, y_vec, y_points, interp):
        peak_y = y_points[np.argmax(interp)]
        peak_v = interp.max()

        # do for y < y_min
        end_v = interp[np.searchsorted(y_points, y_vec.min())]
        end_y = y_points[np.searchsorted(y_points, y_vec.min())]
        pe = peak_v**2 / end_v**2
        x = end_y - peak_y
        d = np.sqrt(-(x**2) / (1 - pe))
        sigma = 1 / (2 * np.pi * d * peak_v)
        y_fill = y_points[y_points <= end_y]
        interp[y_points <= end_y] = 1 / (
            2 * np.pi * sigma * np.sqrt(d**2 + (peak_y - y_fill) ** 2)
        )

        # do for y > y_max
        end_v = interp[np.searchsorted(y_points, y_vec.max(), side="right") - 1]
        end_y = y_points[np.searchsorted(y_points, y_vec.max(), side="right") - 1]
        pe = peak_v**2 / end_v**2
        x = end_y - peak_y
        d = np.sqrt(-(x**2) / (1 - pe))
        sigma = 1 / (2 * np.pi * d * peak_v)
        y_fill = y_points[y_points >= end_y]
        interp[y_points >= end_y] = 1 / (
            2 * np.pi * sigma * np.sqrt(d**2 + (peak_y - y_fill) ** 2)
        )


class RANDPreComputedInterpolate1D(PreComputedInterpolate1D):
    def get_in_memory(self, gid):
        return (
            self._fem_scale
            * self.data[np.random.randint(self.data.shape[0], size=1)[0], :]
        )


# -- aliases --
FEMExact = PreComputedExact
FEMInterpolate1D = PreComputedInterpolate1D
