# pylint: disable=too-many-lines
"""
Cell and Axon classes.
"""

import gc
import inspect
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
from neuron import h

from cajal.common.logging import logger
from cajal.common.math import get_nearest_indices
from cajal.nrn.monitors import APMonitor
from cajal.nrn.simrun import SimulationEnvironment
from cajal.nrn.specs import Specable, Spec, SpecList
from cajal.nrn.stimuli import StimulusSpec, RepeatStimulusSpec
from cajal.units import um, unitdispatch  # pylint: disable=no-name-in-module


__all__ = ("Cell", "Axon", "AxonSpec", "AxonSpecList")


# -- specs --
class AxonSpec(Spec):
    __slots__ = "__intras__", "gid"

    def __init__(self, cls, mutable=None, **kwargs):
        super().__init__(cls, mutable, **kwargs)
        self.__intras__ = []
        self.gid = None

    def __getitem__(self, item):
        return SectionSpec(self, item)

    def __getattr__(self, item):
        if item not in self.__slots__:
            return SubsetSpec(self, item)

    def add_intra(self, intra):
        self.__intras__.append(intra)

    def add_record(self, record):
        self.__records__.append(record)

    def subset(self, subset):
        return SubsetSpec(self, subset)

    def build_with_intras(self, intra_spec_list=None, **kwargs):
        axon = self.build(**kwargs)

        def gen():
            for i in self.__intras__:
                if intra_spec_list:
                    if i in intra_spec_list:
                        yield i
                else:
                    yield i

        intras = []
        for intra in gen():
            subset = (
                getattr(axon, intra.subset)[intra.idx]
                if intra.subset
                else axon[intra.idx]
            )
            if isinstance(subset, list):
                for stim in intra:
                    intras.extend([s(intra.segment) << stim.build() for s in subset])
            else:
                intras.extend([subset(intra.segment) << stim.build() for stim in intra])
        return axon, intras

    def middle(self, n, subset=None):
        return SectionSpec(self, n, subset, middle=True)


class AxonSpecList(SpecList):
    def __init__(self, *args):
        super().__init__(*args)
        if not all([isinstance(spec, AxonSpec) for spec in self]):
            raise TypeError("AxonSpecList may only contain AxonSpecs.")
        for i, spec in enumerate(self):
            spec.gid = i


class SectionSpec:
    __slots__ = "axon_spec", "subset", "segment", "idx", "_middle", "_len"

    def __init__(self, AxonSpec: AxonSpec, idx, subset=None, middle=False):
        self.axon_spec = AxonSpec
        self.subset = subset
        self.segment = 0.5
        self.idx = idx
        self._middle = middle
        self._len = None

    def __call__(self, loc):
        if not 0 <= loc <= 1:
            raise ValueError("Segment position range is in [0.0, 1.0]")
        self.segment = loc
        return self

    def __lshift__(self, other):
        if isinstance(other, (StimulusSpec, RepeatStimulusSpec)):
            return IntraSpec(self, other)
        return NotImplemented

    @property
    def gid(self):
        return self.axon_spec.gid

    def build(self, axon):
        if self._middle:
            return axon.middle(self.idx, self.subset)
        return getattr(axon, self.subset)[self.idx] if self.subset else axon[self.idx]

    def __len__(self):
        if self._middle:
            return self._middle

        def calc_length():
            return len(self.build(self.axon_spec.build()))

        if self._len is None:
            self._len = calc_length()
        gc.collect()
        return self._len


class SubsetSpec:
    __slots__ = "axon_spec", "subset"

    def __init__(self, AxonSpec, subset):
        self.axon_spec = AxonSpec
        self.subset = subset

    def __getitem__(self, item):
        return SectionSpec(self.axon_spec, item, self.subset)

    def middle(self, n):
        return SectionSpec(self.axon_spec, n, self.subset, True)


class IntraSpec(SpecList):
    def __init__(self, SectionSpec, StimulusSpec):
        self.section_spec = SectionSpec
        self.stimulus_spec = StimulusSpec
        self.section_spec.axon_spec.add_intra(self)
        super().__init__(self.stimulus_spec)

    def build(self):
        logger.info(
            "IntraSpec objects cannot execute build() independently"
            "of the AxonSpec with which they are associated."
        )
        pass

    @property
    def subset(self):
        return self.section_spec.subset

    @property
    def segment(self):
        return self.section_spec.segment

    @property
    def idx(self):
        return self.section_spec.idx


class RecordSpec:
    def __init__(self, SectionSpec, variables):
        self.section_spec = SectionSpec
        self.variables = variables
        self.section_spec.axon_spec.add_record(self)


# -- core implementations --


def attribute_constraint(attr):
    """Raise an error if the class calling the decorated method is
    missing an attribute.

    Parameters
    ----------
    attr : str
        Attribute that the class must have.

    Returns
    -------
    func
        Decorated Function.

    Raises
    ------
    AttributeError
        Error if the class is missing the required attribute.
    """

    def _attribute_constraint(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not hasattr(args[0], attr):
                raise AttributeError(
                    f"{f.__qualname__}() method is only available "
                    f"for cells that instantiate a {attr} attribute."
                )
            return f(*args, **kwargs)

        return wrapper

    return _attribute_constraint


class Cell(ABC, Specable):
    """Generic cell template"""

    name = None

    @unitdispatch
    def __init__(self, x: "um" = 0, y: "um" = 0, z: "um" = 0, include_AP_monitors=True):
        self.x = x
        self.y = y
        self.z = z
        self.apm = []
        self._pp = []

        self.all = None
        self.py_all = None

        self.build()
        self.build_subsets()
        self.shape_3D()
        self.init_voltages()
        if include_AP_monitors:
            self.init_AP_monitors()

    @abstractmethod
    def build(self):
        """Create all the model compartments, assign the compartmental
        geometric properties, insert relevant mechanisms and connect them to
        one another."""

    def build_subsets(self):
        """Build subset lists. This defines 'all', but subclasses may
        want to define others. If overridden, call super() to include 'all'."""
        self.all = h.SectionList()
        self.all.wholetree(sec=self.wholetree_section())
        self.py_all = list(self.all)

    @unitdispatch
    def set_position(self, x: "um", y: "um", z: "um"):
        """Set the base location in 3D and move all other parts of the cell
        relative to that location."""
        xs = float(x - self.x)
        ys = float(y - self.y)
        zs = float(z - self.z)

        for sec in self.all:
            for i in range(sec.n3d()):
                h.pt3dchange(
                    i,
                    xs + sec.x3d(i),
                    ys + sec.y3d(i),
                    zs + sec.z3d(i),
                    sec.diam3d(i),
                    sec=sec,
                )

        self.x = x
        self.y = y
        self.z = z
        self._reset_locs()
        return self

    @abstractmethod
    def init_voltages(self):
        """Assign initial voltage conditions for every section
        in the model."""

    @abstractmethod
    def init_AP_monitors(self, distance, threshold, t):
        """Add APMonitor objects to compartments where you want to record for
        action potentials. This is necessary to use early stopping in parallel
        simulations. All APMonitor objects must be gathered under the list
        attribute self.apm"""

    @abstractmethod
    def count_APs(self):
        """Count propagating action potentials."""

    @abstractmethod
    def wholetree_section(self):
        """Return a NEURON section from which to construct a collection
        of all connected sections in this model (self.all)"""

    @abstractmethod
    def shape_3D(self):
        """Assign internal x,y,z locations for all sections in model.
        Every section must have a set ofs x,y,z coordinate for both ends and
        one set of x,y,z coordinates for the centre."""

    def check_early_stop_single(self):
        """Check if the axon has propagated an action potential. If so,
        the simulation."""
        for apm in self.apm:
            apm.check_early_stop()  # pylint: disable=protected-access

    def check_early_stop_multi(self):
        each_n = [apm._n > 0 for apm in self.apm]
        if sum(each_n) > 2:
            raise StopIteration("More than 3 nodes with AP.")

    @classmethod
    def params(cls):
        """Parameters accepted when instantiating this cell."""
        fullargspec = inspect.getfullargspec(cls.__init__)
        params = fullargspec.args[1:]
        return params

    def _reset_locs(self):
        """Reset locations cache."""

    def advance(self):
        """Custom code to execute on each timestep."""

    def initialize(self):
        """Custom code to execute on h.finitialize"""

    def ppadvance(self, i):
        for pp in self._pp:
            pp.advance(i)

    def ppappend(self, pp):
        self._pp.append(pp)

    def ppclear(self):
        self._pp.clear()

    def ppinit(self, t):
        for pp in self._pp:
            pp.init(t)


class Axon(Cell):
    """Generic axon template."""

    n = 0
    _spec = AxonSpec

    def __init__(
        self,
        x,
        y,
        z,
        diameter,
        length,
        gid=None,
        include_AP_monitors=True,
        check_single=True,
    ):
        self.diameter = diameter
        self.length = length
        self._cv = None

        # -- implicit numbering --
        if gid is not None:
            self.gid = int(gid)
        else:
            self.gid = self.__class__.n
        self.__class__.n += 1

        # -- AP checking attributes --
        self.__propagated_ap = False
        self.has_AP_monitors = False
        self.count_node_indices = None
        self.count_node_threshold = None
        self.count_node_t = None

        # -- x,y,z coordinate data
        self._sec_loc = None
        self._node_loc = None
        self._x_loc = None
        self._y_loc = None
        self._z_loc = None
        self._x_node = None
        self._y_node = None
        self._z_node = None

        self.check_single = check_single

        # -- build object --
        super().__init__(x, y, z, include_AP_monitors)

        # -- membrane potential recording? --
        self._check_nodes()

    def check_early_stop(self):
        if self.check_single:
            self.check_early_stop_single()
        self.check_early_stop_multi()

    def build(self):
        self.create_sections()
        self.build_topology()
        self.define_geometry()
        self.define_biophysics()

    @abstractmethod
    def create_sections(self):
        """Create the sections of the axon"""

    @abstractmethod
    def build_topology(self):
        """Connect the sections of the axon to build a tree."""

    @abstractmethod
    def define_geometry(self):
        """Set the 3D geometry of the axon."""

    @abstractmethod
    def define_biophysics(self):
        """Assign the membrane properties across the axon."""

    def shape_3D(self):
        """Assign internal x,y,z locations for all sections in model."""
        axonlength = self.L
        x, z = self.x_, self.z_
        running_total = 0.0
        for sec in self.all:
            seclength = sec.L
            h.pt3dclear(sec=sec)
            h.pt3dadd(
                x, -(axonlength / 2) + running_total + self.y_, z, sec.diam, sec=sec
            )
            running_total += seclength / 2
            h.pt3dadd(
                x, -(axonlength / 2) + running_total + self.y_, z, sec.diam, sec=sec
            )
            running_total += seclength / 2
            h.pt3dadd(
                x, -(axonlength / 2) + running_total + self.y_, z, sec.diam, sec=sec
            )

    def mark_as_active(self):
        """
        Convenience function to set status as having achieved threshold.
        """
        self.__propagated_ap = True

    def mark_as_inactive(self):
        """
        Convenience function to set status as not having achieved threshold.
        """
        self.__propagated_ap = False

    def print_coordinates(self):
        """
        Print coordinates.
        """
        print("{0}, {1}, {2}".format(self.x, self.y, self.z))

    def __iter__(self):
        return iter(self.py_all)

    def __repr__(self):
        return self.__class__.__name__ + "[{}][{}um]".format(
            self.gid, float(self.diameter.to(um))
        )

    def __len__(self):
        return len(self.py_all)

    def __getitem__(self, value):
        return self.py_all[value]

    @property
    def propagated_ap(self):
        """
        Property: whether the cell has generated an action potential.
        """
        return self.__propagated_ap

    @propagated_ap.setter
    def propagated_ap(self, status):
        self.__propagated_ap = bool(status)

    @property
    @attribute_constraint("node")
    def v(self):
        """
        Array of voltages at each node along the axon.
        """
        node = getattr(self, "node")
        v = np.array([sec.v for sec in node])
        return v

    # --------------- x, y, z coordinate location attributes ---------------
    @property
    def x_loc(self):
        """
        x coordinates of all compartments in model
        """
        if self._x_loc is None:
            self._x_loc = x_locs(self.py_all)
        return self._x_loc

    @property
    def y_loc(self):
        """
        y coordinates of all compartments in model
        """
        if self._y_loc is None:
            self._y_loc = y_locs(self.py_all)
        return self._y_loc

    @property
    def z_loc(self):
        """
        z coordinates of all compartments in model
        """
        if self._z_loc is None:
            self._z_loc = z_locs(self.py_all)
        return self._z_loc

    @property
    def sec_loc(self):
        """Return x,y,z coordinates of all sections in model.

        Returns
        -------
        np.ndarray
            [n_sections x 3] array of coordinates.
        """
        if self._sec_loc is None:
            self._sec_loc = np.vstack([self.x_loc, self.y_loc, self.z_loc]).T
        return self._sec_loc

    @property
    @attribute_constraint("node")
    def x_node(self):
        """
        x coordinates of all nodal compartments in model
        """
        node = getattr(self, "node")
        if self._x_node is None:
            self._x_node = x_locs(node)
        return self._x_node

    @property
    @attribute_constraint("node")
    def y_node(self):
        """
        y coordinates of all nodal compartments in model
        """
        node = getattr(self, "node")
        if self._y_node is None:
            self._y_node = y_locs(node)
        return self._y_node

    @property
    @attribute_constraint("node")
    def z_node(self):
        """
        z coordinates of all nodal compartments in model
        """
        node = getattr(self, "node")
        if self._z_node is None:
            self._z_node = z_locs(node)
        return self._z_node

    @property
    def node_loc(self):
        """Return x,y,z coordinates of all nodes in model.
        Can only be called on models that have a 'node' attribute.

        Returns
        -------
        np.ndarray
            [n_nodes x 3] array of coordinates.
        """
        if self._node_loc is None:
            self._node_loc = np.vstack([self.x_node, self.y_node, self.z_node]).T
        return self._node_loc

    def _reset_locs(self):
        self._sec_loc = None
        self._node_loc = None
        self._x_loc = None
        self._y_loc = None
        self._z_loc = None
        self._x_node = None
        self._y_node = None
        self._z_node = None

    @property
    def L(self):
        """True modelled length of the axon model.

        Returns
        -------
        float
            sum of L over all sections in model.
        """
        return sum([sec.L for sec in self.all])

    def _check_nodes(self):
        if not hasattr(self, "node"):
            logger.warning_once(
                f"The model attribute `node` has not been populated "
                f"for {self.__class__.__name__} . Without this, the "
                f"default APMonitor initialisation interface and various "
                f"convenience properties cannot used."
            )

    @attribute_constraint("node")
    def wholetree_section(self):
        """Default section from which to construct tree for all models
        that subclass Axon is the first node in the model.

        Returns
        -------
        h.Section
            self.node[0]
        """
        node = getattr(self, "node")
        return node[0]

    # -----------------------------------
    # -- default AP counting interface --
    # -----------------------------------

    @attribute_constraint("node")
    @unitdispatch
    def set_AP_monitors(
        self, distance: "um" = None, axonnodes=None, threshold=0, t=None
    ):
        """Add AP monitors at specified nodes along the axon.

        Parameters
        ----------
        distance : list
            List of distances (in mm) from the axon 3D reference
            (point on the axon at axon.x, axon.y, axon.z) along the y-axis.
            An APMonitor is inserted in the nearest nodes at each distance.
        axonnodes: list (optional)
            List of node indices in which to insert APMonitors. Overrides
            distance. By default None.
        threshold : float (optional)
            Threshold Vm value for AP monitors. By default 0 (mV)
        t : float (optional)
            Time after which to count for action potentials at the
            given nodes. By default None.
        """

        def _extend_for_multi(param, n):
            if not isinstance(param, list):
                param = [param] * n
                return param
            if len(param) != n:
                raise ValueError("List size incorrect.")
            return param

        if distance is not None or axonnodes is not None:
            node = getattr(self, "node")
            nod = (
                np.unique(np.atleast_1d(axonnodes))
                if axonnodes is not None
                else np.unique(
                    get_nearest_indices(self.y_node[:] - self.y_, np.asarray(distance))
                )
            )
            n = len(nod)
            threshold = _extend_for_multi(threshold, n)
            t = _extend_for_multi(t, n)
            if not self.has_AP_monitors:
                if any(
                    [
                        (self.node[n] is self.node[0] or self.node[n] is self.node[-1])
                        for n in nod
                    ]
                ):
                    logger.info(
                        f"The selected node indices {nod} place one or more "
                        f"APMonitors within the node at the end of the "
                        f"simulated axon. There may be boundary effects."
                    )
                self.count_node_indices = nod
                self.count_node_threshold = threshold
                self.count_node_t = t
                count_nodes = [node[i] for i in nod]
                self.apm = [
                    APMonitor(n, thresh, time)
                    for n, thresh, time in zip(
                        count_nodes, self.count_node_threshold, self.count_node_t
                    )
                ]
                self.has_AP_monitors = True
            else:
                self.apm.clear()
                self.has_AP_monitors = False
                self.set_AP_monitors(distance, axonnodes, threshold, t)

        return self

    def count_APs(self):
        """Count the number of action potentials and set status as achieved
        threshold if count > 0.

        Returns
        -------
        int
            Number of recorded action potentials.
        """
        if self.apm:
            n_aps = sum([apm.n() for apm in self.apm])
            if n_aps > 0:
                self.mark_as_active()
            else:
                self.mark_as_inactive()
            return n_aps
        return None

    def middle(self, num_elements, subset=None):
        subset = getattr(self, subset) if subset else self.py_all
        rec_idx = np.arange(0, len(subset))
        start = (len(rec_idx) - num_elements) // 2
        return subset[start : start + num_elements]

    def cv(self, tstop=20, stim_node=5, amp=1, pw=0.1, rec1=None, rec2=None):
        from cajal.nrn.stimuli import MonophasicPulse

        self.ppclear()
        if self._cv is None:
            cached = (
                self.count_node_indices,
                self.count_node_threshold,
                self.count_node_t,
            )
            rec1 = rec1 or (stim_node + 10)
            rec2 = rec2 or (len(self.node) - 5)
            d = h.distance(self.node[rec1](0.5), self.node[rec2](0.5))
            stim = (
                MonophasicPulse(amp, pw, 0.5).vset().ignore(0) >> self.node[stim_node]
            )
            self.set_AP_monitors(axonnodes=[rec1, rec2])
            env = SimulationEnvironment([self], intra_stim=[stim])
            env.run(tstop, early_stopping=False, progressbar=False)
            self._cv = 0.001 * d / (self.apm[1].spikes()[0] - self.apm[0].spikes()[0])
            self.set_AP_monitors(axonnodes=cached[0], threshold=cached[1], t=cached[2])
        return self._cv


def x_locs(seclist):
    """Calculate x-coordinates of NEURON sections."""
    return np.array(
        [np.mean([sec.x3d(i) for i in range(sec.n3d())]) for sec in seclist]
    )


def y_locs(seclist):
    """Calculate y-coordinates of NEURON sections."""
    return np.array(
        [np.mean([sec.y3d(i) for i in range(sec.n3d())]) for sec in seclist]
    )


def z_locs(seclist):
    """Calculate z-coordinates of NEURON sections."""
    return np.array(
        [np.mean([sec.z3d(i) for i in range(sec.n3d())]) for sec in seclist]
    )
