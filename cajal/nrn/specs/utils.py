import inspect
from itertools import chain
import pickle
from typing import Type, List, Tuple

import decorator
import numpy as np

from cajal.common.logging import logger
from cajal.exceptions import Uninitialised, Unoptimisable
from cajal.mpi.utils import master_only
from cajal.nrn.__backend import Backend as N
from cajal.nrn.specs.mutable import Mutable
from cajal.units import DimensionlessReturn


UNINITIALISED = "UNINIT"


class Dummy:
    value = None


class Specable(DimensionlessReturn):
    """Subclasses may be used to construct Specs for optimisation
    and parallel model execution."""

    _spec = None

    def __init_subclass__(cls, *args, **kwargs):
        cls.__init__ = cls.save_args(cls.__init__)

    @classmethod
    def params(cls):
        """Retrieve the parameters that are used when initialising
        an instance.
        """
        return list(inspect.signature(cls).parameters.keys())

    @classmethod
    def defaults(cls):
        """Get parameter defaults."""
        signature = inspect.signature(cls.__init__)
        return {
            k: v.default if v.default is not None else getattr(N, k, None)
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    @staticmethod
    @decorator.decorator
    def save_args(f, *args, **kwargs):
        def _get_args_dict(fn, *arg, **kw):
            arg_names = list(inspect.signature(fn).parameters.keys())
            dct = {**dict(zip(arg_names, arg)), **kw}
            if "self" in dct:
                dct.pop("self")
            return dct

        local_dct = _get_args_dict(f, *args, **kwargs)
        object.__setattr__(
            args[0], "__init_params__", getattr(args[0], "__init_params__", {})
        )
        # args[0].__init_params__ = getattr(args[0], '__init_params__', {})
        args[0].__init_params__.update(local_dct)
        return f(*args, **kwargs)

    def parameters_dict(self):
        return {k: getattr(self, k, self.__init_params__[k]) for k in self.params()}

    def parameters_dict_str(self):
        try:
            dct = {
                k: str(getattr(self, k, self.__init_params__[k])) for k in self.params()
            }
        except KeyError:
            dct = {k: str(getattr(self, k, None)) for k in self.params()}
        return dct

    def non_defaults_parameters(self):
        defaults = self.defaults()
        return {
            k: v
            for k, v in self.parameters_dict().items()
            if v != defaults.get(k, None)
        }

    @classmethod
    def argdims(cls):
        """Get the dimensionality of each constructor parameter."""
        return {k: getattr(cls, "{}_dim".format(k), 1) for k in cls.params()}

    @classmethod
    def _check_kwargs(cls, **kwargs):
        params = set(cls.params())
        total_names = set(kwargs.keys())
        if not params.issubset(total_names):
            missing = params - total_names
            defaults = cls.defaults()
            available = set(defaults.keys())
            total_missing = missing - available
            if total_missing:
                raise AttributeError(
                    f"These parameters are missing and no "
                    f"defaults are specified: {total_missing}"
                )

        if not total_names.issubset(params):
            extraneous = total_names - params
            logger.info(
                f"One or more supplied parameters are not "
                f"expected and will be ignored: {extraneous}."
            )
            for n in extraneous:
                del kwargs[n]

        return kwargs

    @classmethod
    def _construct_mutable(cls, **kwargs):
        # kwargs = cls._check_kwargs(**kwargs)
        mutable = {}
        for k, v in kwargs.items():
            if isinstance(v, Mutable):
                mutable[k] = v
                kwargs[k] = UNINITIALISED
        return mutable, kwargs

    @classmethod
    def SPEC(cls, *args, **kwargs):
        bound = inspect.signature(cls).bind(*args, **kwargs)
        mutable, kwargs = cls._construct_mutable(**bound.arguments)
        spec_cls = cls._spec or Spec
        return spec_cls(cls, mutable, **kwargs)

    def to_spec(self):
        return self._spec(self.__class__, **self.non_defaults_parameters())

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.parameters_dict() == other.parameters_dict()
        return NotImplemented


class Spec:
    """Generic Spec template class."""

    __slots__ = "cls", "_parameters", "_mutable", "_ndim", "_optimisable"

    def __init__(self, cls, mutable=None, **kwargs):
        self.cls = cls
        self._parameters = kwargs
        self._mutable = mutable or {}
        self._ndim = None
        self._optimisable = None

    def save(self, path: str):
        """Save spec as pickled object."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load spec from pickled binary."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def parameters(self):
        def get(key, value):
            val = self._mutable.get(key, Dummy).value
            if val is None:
                return value
            return val

        return {k: get(k, v) for k, v in self._parameters.items()}

    @property
    def initialised(self) -> bool:
        return not any([v is UNINITIALISED for v in self.parameters.values()])

    @property
    def ndim(self) -> int:
        if self._ndim is None:
            self._ndim = sum([v.ndim for v in self._mutable.values()])
        return self._ndim

    @property
    def optimisable(self) -> bool:
        if not self._mutable:
            return True
        if self._optimisable is None:
            self._optimisable = all([e.optimisable for e in self._mutable.values()])
        return self._optimisable

    @property
    def can_be_optimised(self) -> bool:
        return self.optimisable and self.ndim > 0

    @property
    def mutable(self) -> bool:
        return bool(self._mutable)

    @property
    def bounds(self) -> List[Tuple[float]]:
        if self.optimisable:
            return list(
                chain.from_iterable([opt.bounds for opt in self._mutable.values()])
            )
        return []

    def dicts(self) -> Tuple[Type, dict]:
        return self.cls, self.parameters

    def numpy(self) -> np.ndarray:
        if not self.initialised:
            raise Uninitialised(
                "Not all spec parameters have been assigned a "
                "numerical value so a numerical array cannot be "
                "constructed."
            )
        return np.concatenate([np.array(v).flatten() for v in self.parameters.values()])

    def __lshift__(self, other):
        if self._mutable:
            other = np.asarray(other)
            if other.dtype.kind in "biufc":
                length, ndim = other.size, self.ndim
                if length != ndim:
                    raise ValueError(
                        f"Incorrect number of parameter values passed to "
                        f"specification. Expected {ndim}, received {length}"
                    )
                if length != 0:
                    dims = np.cumsum([v.ndim for v in self._mutable.values()])
                    to_push = np.split(other, dims[:-1])
                    for p, (k, v) in zip(to_push, self._mutable.items()):
                        push = v.parse(p)
                        self.parameters[k] = push
            else:
                return NotImplemented
        else:
            logger.info_once(f"{self} has no mutable parameters.")

    def build(self, **kwargs) -> Specable:
        """Builds an instance of a Specable class from a Spec."""
        if not self.initialised:
            raise Uninitialised(
                "Specs cannot be built if any parameters are uninitialised."
            )
        if kwargs:
            return self.cls(
                **{k: v for k, v in kwargs.items() if k not in self.parameters},
                **self.parameters,
            )
        else:
            return self.cls(**self.parameters)

    def __repr__(self):
        return f"{self.cls.__name__} : {self.parameters}"

    def parameters_dict(self, pvec: np.ndarray = None) -> Tuple[Type, dict]:
        dims = np.cumsum([v.ndim for v in self._mutable.values()])
        to_push = np.split(pvec if pvec is not None else [], dims[:-1])

        def generator():
            for p, (k, v) in zip(to_push, self._mutable.items()):
                push = v.parse(p)
                yield np.asscalar(push) if np.size(push) == 1 else push

        gen = generator()

        if pvec is not None:
            param_dict = {
                k: v if k not in self._mutable else next(gen)
                for k, v in self.parameters.items()
            }
        else:
            param_dict = self.parameters
        return self.cls, param_dict


class ContainerSpec(Spec):
    def __init__(self, **kwargs):
        mutable = {}
        for k, v in kwargs.items():
            if isinstance(v, Mutable):
                mutable[k] = v
                kwargs[k] = UNINITIALISED
        super().__init__(None, mutable, **kwargs)

    def build(self):
        if not self.initialised:
            raise Uninitialised(
                "Specs cannot be built if any parameters are uninitialised."
            )
        return self.parameters

    def __repr__(self):
        return f"Container :: {self.parameters}"


class SpecList:
    __slots__ = ("_specs", "_ndim", "_optimisable", "_mutable_specs", "_mutable_inputs")

    def __init__(self, *args):
        if not args or (len(args) == 1 and args[0] is None):
            self._specs = []
        elif (
            len(args) == 1
            and hasattr(args[0], "__iter__")
            and not isinstance(args[0], SpecList)
        ):
            self._specs = list(args[0])
        else:
            self._specs = [arg for arg in args if arg is not None]
        self._ndim = None
        self._optimisable = None
        self._mutable_specs = list(
            dict.fromkeys(
                chain.from_iterable(
                    [
                        e._mutable_specs
                        if isinstance(e, SpecList)
                        else ([e] if e.mutable else [])
                        for e in self
                    ]
                )
            )
        )
        self._mutable_inputs = None

    def __iter__(self):
        return iter(self._specs)

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return self.__class__([self._specs[i] for i in idx])
        elif isinstance(idx, int):
            return self._specs[idx]
        else:
            return self.__class__(self._specs[idx])

    def __setitem__(self, idx, item):
        self._specs[idx] = item

    def __len__(self):
        return len(self._specs)

    def __contains__(self, item):
        return item in self._specs

    def __repr__(self):
        return self._specs.__repr__()

    def append(self, item):
        if isinstance(item, Spec):
            self._specs.append(item)
            if item.mutable:
                self._mutable_specs.append(item)
            self._ndim = self._optimisable = None
        elif isinstance(item, SpecList):
            self._specs.append(item)
            self._mutable_specs.extend(item._mutable_specs)
            self._ndim = self._optimisable = None
        else:
            raise TypeError("Cannot append non-spec object to SpecList.")

    @property
    def initialised(self):
        return all([s.initialised for s in self._mutable_specs])

    @property
    def ndim(self):
        if self._ndim is None:
            self._ndim = np.sum([v.ndim for v in self.mutable_inputs()])
        return self._ndim

    @property
    def optimisable(self):
        if not self._specs:
            return True
        if self._optimisable is None:
            self._optimisable = all([e.optimisable for e in self._mutable_specs])
        return self._optimisable

    def bounds(self):
        return list(chain.from_iterable([opt.bounds for opt in self.mutable_inputs()]))

    @property
    def can_be_optimised(self):
        return (self.ndim > 0) and self.optimisable

    def dicts(self):
        return [spec.dicts() for spec in self._mutable_specs]

    def numpy(self):
        if self._mutable_specs:
            return np.concatenate([spec.numpy() for spec in self._mutable_specs])

    def mutable_inputs(self):
        if self._mutable_inputs is None:
            self._mutable_inputs = list(
                dict.fromkeys(
                    chain.from_iterable(
                        [e._mutable.values() for e in self._mutable_specs]
                    )
                )
            )
        return self._mutable_inputs

    def __lshift__(self, other):
        if self._mutable_specs:
            other = np.asarray(other)
            if other.dtype.kind in "biufc":
                length, ndim = other.size, self.ndim
                if length != ndim:
                    raise ValueError(
                        f"Incorrect number of parameter values passed "
                        f"to specification. Expected {ndim}, received "
                        f"{length}"
                    )
                if length != 0:
                    dims = np.cumsum([v.ndim for v in self.mutable_inputs()])
                    to_push = np.split(other, dims[:-1])
                    for mut, params in zip(self.mutable_inputs(), to_push):
                        mut.parse(params)
            else:
                return NotImplemented
        else:
            logger.info("SpecList contains no mutable Specs.")

    def build(self):
        return [spec.build() for spec in self]

    def parameters_dict(self, pvec=None):
        pvec = np.asarray(pvec if pvec is not None else [])
        length, ndim = pvec.size, self.ndim
        if length != ndim:
            raise ValueError(
                f"Incorrect number of parameter values passed "
                f"to specification. Expected {ndim}, received "
                f"{length}"
            )
        # dims = np.cumsum([v.ndim for v in self._mutable_specs])
        # to_push = np.split(pvec, dims[:-1])
        self << pvec
        return [spec.parameters_dict() for spec in self._mutable_specs]


class ModelSpec(SpecList):
    __slots__ = "axon_specs", "extra_specs", "intra_specs", "size"

    def __init__(self, axon_specs, extra_specs=None, intra_specs=None):
        from cajal.nrn.cells import AxonSpecList

        self.axon_specs = AxonSpecList(axon_specs)
        self.extra_specs = SpecList(extra_specs)
        self.intra_specs = SpecList(intra_specs)
        self.size = len(self.axon_specs)
        super().__init__(self.axon_specs, self.extra_specs, self.intra_specs)

    """
    @master_only
    def summary(self):
        axon_summary = "".join([f"  {i}. {a}\n"
                                for i, a in
                                enumerate(self.axon_specs)]) \
            if self.axon_specs else ""

        extra_summary = "".join([f"  {i}. {a}\n\n"
                                 for i, a in
                                 enumerate(self.extra_specs)]) \
            if self.extra_specs else ""

        intra_summary = "".join([f"  {i}. {a}\n"
                                 for i, a in
                                 enumerate(self.intra_specs)]) \
            if self.intra_specs else ""

        mutable_summary = "".join([f"  {i}. {a}\n"
                                   for i, a in
                                   enumerate(self._mutable_specs)]) \
            if self._mutable_specs else ""

        n_axons = len(self.axon_specs)
        n_extra = len(self.extra_specs)
        n_intra = len(self.intra_specs)
        n_mutab = len(self._mutable_specs)

        summary_str = (
            f"\n{self.ndim} dimensions."
            f"\n{n_axons} axon{'' if n_axons == 1 else 's'}"
            f"{':' if n_axons != 0 else '.'}\n"
            f"{axon_summary}\n"
            f"{n_extra} extracellular source{'' if n_extra == 1 else 's'}"
            f"{':' if n_extra != 0 else '.'}\n"
            f"{extra_summary}\n"
            f"{n_intra} intracellular source{'' if n_intra == 1 else 's'}"
            f"{':' if n_intra != 0 else '.'}\n"
            f"{intra_summary}\n"
            f"{n_mutab} mutable{':' if n_mutab != 0 else '.'}\n"
            f"{mutable_summary}\n"
        )
        print(summary_str)
    """

    def bounds(self):
        if self.can_be_optimised:
            return SpecList.bounds(self)
        raise Unoptimisable(
            "All mutable parameters have not been defined as optimisable."
        )

    @master_only
    def summary(self):
        mutables_list = self.mutable_inputs()

        def parameters(spec, mutables):
            def get(key, value):
                val = getattr(spec, "_mutable").get(key, Dummy)
                if val is Dummy:
                    return value
                if val.value is None:
                    return f"{value}:[{mutables.index(val)}]"
                return f"{val.value}:[{mutables.index(val)}]"

            return {k: get(k, v) for k, v in getattr(spec, "_parameters").items()}

        def summary(spec, mutables):
            return f"{spec.cls.__name__} : {parameters(spec, mutables)}"

        def extra_summary(spec, mutables):
            if isinstance(spec.stim_spec, list):
                return f"{summary(spec.source_spec, mutables)}" + "".join(
                    [f"\n\t|--- << {summary(sp, mutables)}" for sp in spec.stim_spec]
                )
            return (
                f"{summary(spec.source_spec, mutables)}\n\t"
                f"|--- << {summary(spec.stim_spec, mutables)}"
            )

        def intra_summary(spec, mutables):
            axon_cls_name = spec.section_spec.axon_spec.cls.__name__
            axon_position = self.axon_specs._specs.index(spec.section_spec.axon_spec)
            subset = spec.section_spec.subset
            if subset is None:
                subset = "compartment"
            idx = spec.section_spec.idx
            axon = f"{axon_cls_name} #{axon_position}, {subset} {idx}"
            return f"{axon} << {summary(spec.stimulus_spec, mutables)}"

        axon_summary = (
            "".join(
                [
                    f"  {i}. {summary(a, mutables_list)}\n"
                    for i, a in enumerate(self.axon_specs)
                ]
            )
            if self.axon_specs
            else ""
        )

        extra_summary = (
            "".join(
                [
                    f"  {i}. {extra_summary(a, mutables_list)}\n\n"
                    for i, a in enumerate(self.extra_specs)
                ]
            )
            if self.extra_specs
            else ""
        )

        intra_summary = (
            "".join(
                [
                    f"  {i}. {intra_summary(a, mutables_list)}\n"
                    for i, a in enumerate(self.intra_specs)
                ]
            )
            if self.intra_specs
            else ""
        )

        mutable_summary = (
            "".join(
                [
                    f"  {i}. {summary(a, mutables_list)}\n"
                    for i, a in enumerate(self._mutable_specs)
                ]
            )
            if self._mutable_specs
            else ""
        )

        n_axons = len(self.axon_specs)
        n_extra = len(self.extra_specs)
        n_intra = len(self.intra_specs)
        n_mutab = len(self._mutable_specs)

        summary_str = (
            f"\n{self.ndim} dimensions."
            f"\n{n_axons} axon{'' if n_axons == 1 else 's'}"
            f"{':' if n_axons != 0 else '.'}\n"
            f"{axon_summary}\n"
            f"{n_extra} extracellular source{'' if n_extra == 1 else 's'}"
            f"{':' if n_extra != 0 else '.'}\n"
            f"{extra_summary}\n"
            f"{n_intra} intracellular source{'' if n_intra == 1 else 's'}"
            f"{':' if n_intra != 0 else '.'}\n"
            f"{intra_summary}\n"
            f"{n_mutab} mutable{':' if n_mutab != 0 else '.'}\n"
            f"{mutable_summary}\n"
        )
        print(summary_str)
