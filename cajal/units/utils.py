"""Utilities for handling units."""

import inspect

import decorator

import cajal.units
from cajal.units import unyt_quantity, unyt_array, Unit


def apply_units(value, unit):
    """Apply units to arrays / numbers."""
    if value is not None:
        if isinstance(unit, str):
            unit = getattr(cajal.units, unit, None) or Unit(unit)
            try:
                if value.units == unit:
                    return value
                value = value.to(unit)
            except AttributeError:
                try:
                    value = unyt_quantity(value, unit)
                except RuntimeError:
                    value = unyt_array(value, unit)
    return value


@decorator.decorator
def unitdispatch(f, *args, **kwargs):
    """Transform function inputs into quantities with units.
    If used with more than one decorator, @unitdispatch should
    be the innermost decorator.
    """
    sig = inspect.signature(f)
    params = sig.parameters
    bound = sig.bind(*args, **kwargs)
    dispatch = {
        k: apply_units(v, params[k].annotation) for k, v in bound.arguments.items()
    }
    return f(**dispatch)


def strip_units(array):
    """Retrieve unitless array."""
    try:
        if array.ndim == 0:
            return float(array.ndarray_view())
        return array.ndarray_view()
    except AttributeError:
        return array


class DimensionlessReturn:
    def __getattr__(self, item):
        if item[-1] == "_":
            return strip_units(getattr(self, item[:-1]))

        raise AttributeError(
            "{} object has no attribute named {}".format(self.__class__.__name__, item)
        )

    def get_dimensionless(self, item, default):
        return getattr(self, f"{item}_", default)


class DimensionlessReturnMeta(type):
    def __getattr__(cls, item):
        if item[-1] == "_":
            return strip_units(getattr(cls, item[:-1]))

        raise AttributeError(
            "{} object has no attribute named {}".format(cls.__name__, item)
        )
