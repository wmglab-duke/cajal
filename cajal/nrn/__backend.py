"""NEURON Backend."""

from neuron import h

from cajal.units import ms, ohm, cm, C, DimensionlessReturn, unitdispatch

__all__ = "Backend"

# def _create_type(meta, name, attrs):
#     type_name = f'{name}Type'
#     type_attrs = {}
#     for k, v in attrs.items():
#         if type(v) is _ClassPropertyDescriptor:
#             type_attrs[k] = v
#     return type(type_name, (meta,), type_attrs)
#
#
# class ClassPropertyType(type):
#     """ClassProperty type."""
#     def __new__(meta, name, bases, attrs):
#         Type = _create_type(meta, name, attrs)
#         cls = super().__new__(meta, name, bases, attrs)
#         cls.__class__ = Type
#         return cls
#
#
# class _ClassPropertyDescriptor:
#     def __init__(self, fget, fset=None):
#         self.fget = fget
#         self.fset = fset
#
#     def __get__(self, obj, owner):
#         if self in obj.__dict__.values():
#             return self.fget(obj)
#         return self.fget(owner)
#
#     def __set__(self, obj, value):
#         if not self.fset:
#             raise AttributeError("can't set attribute")
#         return self.fset(obj, value)
#
#     def setter(self, func):
#         self.fset = func
#         return self
#
#
# def classproperty(func):
#     """Class property wrapper."""
#     return _ClassPropertyDescriptor(func)
#
#
# class Uninstantiable:
#     """Prevent instantiation of Class."""
#     def __new__(cls, *args, **kwargs):
#         raise TypeError("Class {} may not be instantiated.".format(cls))
#
#
# BackendType = type('BackendType', (ClassPropertyType, DimensionlessReturnMeta), {})


# pylint: disable=no-self-argument
class __Backend(DimensionlessReturn):
    """
    NEURON backend. Set global tstop, dt, rhoe, and temp
    parameters.

    DEFAULTS
    --------
    dt      = 0.005 [ms]

    tstop   = 5     [ms]

    rhoe    = 1000  [ohm-cm]

    temp    = 37    [C]

    """

    __defaults__ = {
        "dt": 0.005 * ms,
        "tstop": 5 * ms,
        "rhoe": 1000 * ohm * cm,
        "temp": 37 * C,
        "chunksize": 50 * ms,
    }

    _instance = None  # Keep instance reference

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self._dt = 0.005 * ms
        self._tstop = 5 * ms
        self._rhoe = 1000 * ohm * cm
        self._temp = 37 * C

        self._chunksize = 50 * ms

        h.celsius = float(self._temp)
        h.dt = float(self._dt)

        self.EARLY_STOPPING = True
        self.log_early_stopping = False

    @property
    def dt(self):
        """Global simulation timestep."""
        return self._dt

    @dt.setter
    @unitdispatch
    def dt(self, value: "ms"):
        """Set dt globally."""
        self._dt = value
        h.dt = float(value)

    @property
    def tstop(self):
        """Global simulation duration [ms]."""
        return self._tstop

    @tstop.setter
    @unitdispatch
    def tstop(self, value: "ms"):
        """Set tstop globally."""
        self._tstop = value

    @property
    def rhoe(self):
        """Global simulation isotropic medium resistivity."""
        return self._rhoe

    @rhoe.setter
    @unitdispatch
    def rhoe(self, value: "ohm*cm"):
        """Set rhoe globally."""
        self._rhoe = value

    @property
    def temp(self):
        """Global simulation temperature."""
        return self._temp

    @temp.setter
    @unitdispatch
    def temp(self, value: "C"):
        """Set temp globally."""
        self._temp = value
        h.celsius = float(value)

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    @unitdispatch
    def chunksize(self, value: "ms"):
        self._chunksize = value

    def reset(self, quantity):
        try:
            setattr(self, quantity, self.__defaults__[quantity])
        except KeyError:
            raise ValueError(f"nrn Backend has no default for {quantity}") from None


Backend = __Backend()
