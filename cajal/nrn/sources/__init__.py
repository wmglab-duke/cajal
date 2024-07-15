"""
The `sources` modules defines how extracellular potential fields
are calculated.
"""

__all__ = [
    "Source",
    "Point",
    "IsotropicPoint",
    "IsotropicPolar",
    "AnisotropicPoint",
    "AnisotropicPolar",
    "PreComputed",
    "PreComputedInterpolate1D",
    "PreComputedExact",
    "RANDPreComputedInterpolate1D",
    "Line",
    "Arbitrary",
    "Helix",
    "Arc",
    "Line3D",
]

from cajal.common.logging import logger
from cajal.nrn.sources.base import Source
from cajal.nrn.sources.point import (
    Point,
    IsotropicPoint,
    IsotropicPolar,
    AnisotropicPoint,
    AnisotropicPolar,
)
from cajal.nrn.sources.precomputed import (
    PreComputed,
    PreComputedInterpolate1D,
    PreComputedExact,
    RANDPreComputedInterpolate1D,
)
from cajal.nrn.sources.line import Line, Arbitrary, Helix, Arc, Line3D


_SOURCE_NAMES = {
    "isotropic_point": IsotropicPoint,
    "isotropic_point_polar": IsotropicPolar,
    "anisotropic_point": AnisotropicPoint,
    "anisotropic_point_polar": AnisotropicPolar,
    "pre_computed_exact": PreComputedExact,
    "pre_computed_interpolate_1d": PreComputedInterpolate1D,
    "helix": Helix,
    "arc": Arc,
    "arbitrary": Arbitrary,
}

__SOURCE_NAMES_BACKUP = _SOURCE_NAMES.copy()


def all_valid_source_names():
    """Return the names of all field sources registered to the API."""
    return _SOURCE_NAMES.keys()


def source_dictionary():
    """Return dictionary relating names to field source classes."""
    return _SOURCE_NAMES.copy()


def check_valid_source(source_name):
    """Determine if a given potential field source is implemented and
    available for use.

    Parameters
    ----------
    source_name : str
        String alias of source.

    Returns
    -------
    str
        The source alias name.

    Raises
    ------
    TypeError
        Exception if source name is not recognised.
    """
    if source_name not in _SOURCE_NAMES:
        raise TypeError(
            "{} is not a valid source type. Choose from: {}".format(
                source_name, _SOURCE_NAMES.keys()
            )
        )
    return source_name


def get_source_class(source):
    """Convert an source string alias to the source class.

    Parameters
    ----------
    source : str, type, Engine
        String alias of the source (defined under the class
        attribute 'name'.) / Engine class definition / Engine
        instance.

    Returns
    -------
    Source
        Engine class

    Raises
    ------
    ValueError
        Error if a class with the given alias is not implemented
        and registered with the sources interface.
    """
    if isinstance(source, str):
        try:
            return _SOURCE_NAMES[source]
        except KeyError:
            raise ValueError("{} is not a valid source type".format(source)) from None
    elif isinstance(source, Source):
        return source.__class__
    elif issubclass(source, Source):
        return source
    else:
        raise TypeError("The source is not of valid type.")


def get_source_params(source_name):
    """Get all arguments required to instantiate an source.

    Parameters
    ----------
    source_name : str
        String alias of source.

    Returns
    -------
    list
        List of parameter names.

    Raises
    ------
    ValueError
        Error if a class with the given alias is not implemented
        and registered with the sources interface.
    """
    _eng = get_source_class(source_name)
    return _eng.params()


# -- registration --


def register_source(source):
    """Register a custom source class with the Engine interface
    in order to use it with the parallel compute source and optimisation
    platforms.

    Parameters
    ----------
    source
        Engine class (subclass cajal.nrn.electrodes.Engine)

    Raises
    ------
    ValueError
        Error if the source class is not named.

    TypeError
        Error if the source does not subclass Engine.
    """
    if source.name is None:
        raise ValueError("The source to be registered must be given a name.")
    if not issubclass(source, Source):
        raise TypeError(
            "All registered sources must subclass cajal.nrn.electrodes.Engine"
        )
    if source.name in _SOURCE_NAMES:
        txt = (
            "A source with the name {} is already registered.".format(source.name)
            + " This registration overwrites the "
            + "previous registration."
        )
        logger.warning(txt)
    _SOURCE_NAMES.update({source.name: source})


def restore_source_defaults():
    """Restore the defaults of the field sources API."""
    global _SOURCE_NAMES  # pylint: disable=global-statement
    _SOURCE_NAMES = __SOURCE_NAMES_BACKUP.copy()
