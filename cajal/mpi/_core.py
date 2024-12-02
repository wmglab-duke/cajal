from abc import ABC, abstractmethod
from collections import namedtuple
import datetime
import gc
import os
from typing import Union, Iterable, List, Tuple, Callable, Sequence
import sys

import h5py
from mpi4py import MPI
import numpy as np
from numpy import dtype as dt
from tqdm import tqdm

from cajal.common.logging import logger
from cajal.mpi.__backend import Backend as MPISRC
from cajal.nrn.__backend import Backend as N
from cajal.nrn.monitors import StateMonitor
from cajal.nrn.simrun import SimulationEnvironment
from cajal.nrn.specs import ModelSpec
from cajal.mpi.utils import master_only
from cajal.units import ms


Numerical = Union[int, float, np.ndarray]

COMM = MPISRC.COMM
RANK = MPISRC.RANK
SIZE = MPISRC.SIZE

COMPUTE_TAG = 0
END_TAG = 99

ModelFragment = namedtuple("ModelFragment", ["pid", "gid", "params"])


class Recorder:
    __dtypes__ = {
        dt("int8"): MPI.INT8_T,
        dt("int16"): MPI.INT16_T,
        dt("int32"): MPI.INT32_T,
        dt("int64"): MPI.INT64_T,
        dt("uint8"): MPI.UINT8_T,
        dt("uint16"): MPI.UINT16_T,
        dt("uint32"): MPI.UINT32_T,
        dt("uint64"): MPI.UINT64_T,
        dt("float32"): MPI.FLOAT,
        dt("float64"): MPI.DOUBLE,
        dt("i"): MPI.INT,
        dt("f"): MPI.FLOAT,
        dt("d"): MPI.DOUBLE,
    }

    def __init__(
        self,
        shapes: Iterable[Union[int, Tuple[int, ...]]],
        dtypes: Union[dt, Tuple[dt, ...]],
        aliases: Sequence[str] = None,
    ):
        aliases_it = aliases if isinstance(aliases, (tuple, list)) else [aliases]
        self.results_shapes = {}
        self.results_dtypes = {}
        self.results_aliases = []
        self.mpi_dtypes = {}
        self.__results__ = {}
        for i, shape in enumerate(shapes):
            attr = aliases_it[i] if aliases else f'results{i if i > 0 else ""}'
            dtype = dt(dtypes[i]) if isinstance(dtypes, (list, tuple)) else dt(dtypes)
            self.results_shapes[attr] = shape
            self.results_dtypes[attr] = dtype
            self.results_aliases.append(attr)
            self.mpi_dtypes[attr] = self.__dtypes__[dtype]

        self._mpi_file = None
        self._mpi_model_group = None
        self._mpi_run_group = None

    def _construct_records(self, n_parameter_sets: int, x: np.ndarray, **kwargs):
        aliases = self.results_aliases
        dtypes = self.results_dtypes
        shapes = self.results_shapes
        if self.using_mpio:
            if MPISRC.MASTER():
                time = str(datetime.datetime.now())
            else:
                time = None
            time = COMM.bcast(time, root=MPISRC.MASTER_RANK)
            n = self._mpi_model_group.attrs["n_eval"]
            COMM.barrier()
            self._mpi_run_group = self._mpi_model_group.create_group(str(n))
            self._mpi_run_group.attrs["time"] = time
            self._mpi_run_group.create_dataset("parameters", data=x)
            for i, attr in enumerate(aliases):
                dtype = dtypes[attr]
                setattr(
                    self,
                    attr,
                    self._mpi_run_group.create_dataset(
                        attr, (n_parameter_sets, *shapes[attr]), dtype=dtype
                    ),
                )
                self.__results__[attr] = getattr(self, attr)
        else:
            for i, attr in enumerate(aliases):
                dtype = dtypes[attr]
                setattr(
                    self, attr, np.empty((n_parameter_sets, *shapes[attr]), dtype=dtype)
                )
                self.__results__[attr] = getattr(self, attr)

    def make_available_everywhere(self, *args: str):
        for var in args:
            try:
                COMM.Bcast(
                    [getattr(self, var), self.mpi_dtypes[var]], root=MPISRC.MASTER_RANK
                )
            except AttributeError:
                raise

    def mpio(self, handle, mode="a", attrs: dict = None):
        if not h5py.get_config().mpi and MPISRC.ENABLED:
            if MPISRC.MASTER():
                logger.warning(
                    "h5py has not been built with MPI support. mpio() is not available. "
                    f"Standard data reassembly on master rank {MPISRC.MASTER_RANK} will "
                    f"be used instead \n"
                )
            return self
        if isinstance(handle, str):
            if MPISRC.ENABLED:
                handle = h5py.File(handle, mode, driver="mpio", comm=COMM)
            else:
                handle = h5py.File(handle, mode)
        elif not isinstance(handle, h5py.File):
            raise TypeError(
                "File handle must either be a string (a location on disk), or "
                "an instance of h5py.File."
            )
        self._mpi_file = handle
        self._mpi_model_group = self._mpi_file
        if "n_eval" not in self._mpi_model_group.attrs.keys():
            self._mpi_model_group.attrs["n_eval"] = 0
        if "script" not in self._mpi_model_group.keys():
            file = os.path.abspath(sys.argv[0])
            with open(file, "r") as f:
                txt = f.read()
            self._mpi_model_group.attrs["script"] = txt
        if "created" not in self._mpi_model_group.keys():
            if MPISRC.MASTER():
                time = str(datetime.datetime.now())
            else:
                time = None
            time = COMM.bcast(time, root=MPISRC.MASTER_RANK)
            self._mpi_model_group.attrs["created"] = time
        if attrs is not None:
            for k, v in attrs.items():
                self._mpi_model_group.attrs[k] = v
        return self

    def standard_io(self):
        self.mpio_close()
        return self

    def mpio_close(self):
        if self._mpi_file:
            COMM.barrier()
            self._mpi_file.close()

    @property
    def using_mpio(self):
        return bool(self._mpi_file)


class Reporter:
    # noinspection PyTypeChecker
    def __init__(self):
        self.pbar: tqdm = None
        self.n_eval = 0
        self._persistent_pbar = False

    @master_only
    def init_report(self, n: int):
        self.pbar = tqdm(total=n, bar_format="{desc}{percentage:3.0f}%|{bar:50}{r_bar}")
        self.pbar.set_description(f"[{self.n_eval}]")
        self.n_eval += 1

    def init_report_persistent(self, n: int):
        self.init_report(n)
        self._persistent_pbar = True

    def close(self):
        if self.pbar and not self._persistent_pbar:
            self.pbar.close()

    def update_report(self, n=1):
        if self.pbar:
            self.pbar.update(n)


class MPIRunner(ABC, Recorder, Reporter):
    """
    Parameters
    ----------
    shapes: int, tuple(int)
        Shapes of outputs from func() method.
    dtypes: np.dtype, str, type
    """

    _lb_methods = {
        "static": "_run_static_load_balancing",
        "dynamic": "_run_dynamic_load_balancing",
    }

    def __init__(
        self,
        shapes: Iterable[Union[int, Tuple[int, ...]]],
        dtypes: Union[dt, str, type, Sequence[Union[dt, str, type]]],
        aliases: Sequence[str] = None,
        load_balancing: str = "dynamic",
        retval: Union[str, Sequence[str]] = None,
    ):
        aliases = aliases or self.func.__annotations__.get("return", None)
        Recorder.__init__(self, shapes, dtypes, aliases)
        Reporter.__init__(self)
        self._run_method: Callable[..., None] = (
            getattr(self, self._lb_methods[load_balancing])
            if MPISRC.ENABLED
            else self._single_process
        )
        self.retval = retval or self.results_aliases

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    # -- core run routine --
    def run(self, x: np.ndarray, progressbar=True, **kwargs) -> list:
        """
        Run a set of parameters through your model.

        Parameters
        ----------
        x : np.ndarray
            2D Array of parameters.
        progressbar : bool
            Show progressbar or not. Optional, by default True.
        kwargs
            Additional keyword arguments to pass to core modelling
            function.

        Returns
        -------
        list
            Attributes defined in 'retval' passed to initializer.
        """
        x = np.atleast_2d(x)
        n_parameter_sets = np.size(x, 0)
        self._construct_records(n_parameter_sets, x, **kwargs)
        if MPISRC.MASTER():
            self._check_return_values()
        if (not self._persistent_pbar) and progressbar:
            self.init_report(self.n_problem_pieces(n_parameter_sets))
        self._run_method(x, **kwargs)
        COMM.barrier()
        if self.using_mpio:
            self._mpi_model_group.attrs["n_eval"] += 1
        self.close()
        return self.__return__()

    @abstractmethod
    def func(self, x, **kwargs):
        pass

    @staticmethod
    def n_problem_pieces(n):
        return n

    def _single_process(self, x: np.ndarray, **kwargs):
        for i, params in enumerate(x):
            results = self.func(params, **kwargs)
            try:
                for result, buffer in zip(results, self.__results__.values()):
                    buffer[i, :] = result
            except TypeError:
                getattr(self, self.results_aliases[0])[i, :] = results
            self.update_report()

    def _run_static_load_balancing(self, x: np.ndarray, **kwargs):
        FINISHED = None
        shapes = self.results_shapes
        dtypes = self.results_dtypes
        param_id = np.empty(1, dtype=np.int32)

        available_ranks = list(range(SIZE))
        available_ranks.remove(MPISRC.MASTER_RANK)

        using_mpio = self.using_mpio

        if not using_mpio:
            local_rec = [
                np.empty(shapes[var], dtype=dtypes[var]) for var in self.__results__
            ]
        else:
            local_rec = [ref for ref in self.__results__.values()]

        if RANK == MPISRC.MASTER_RANK:
            remaining = SIZE - 1
            while remaining > 0:
                s = MPI.Status()
                COMM.Probe(status=s)
                if s.tag == COMPUTE_TAG:
                    COMM.Recv([param_id, MPI.INT32_T], source=s.source)
                    if not using_mpio:
                        for rec, (var, buffer) in zip(
                            local_rec, self.__results__.items()
                        ):
                            COMM.Recv(
                                [rec, self.mpi_dtypes[var]],
                                source=s.source,
                                tag=COMPUTE_TAG,
                            )
                            buffer[param_id, :] = rec
                    self.update_report()
                elif s.tag == END_TAG:
                    COMM.recv(tag=END_TAG)
                    remaining -= 1
        else:
            for i, params in enumerate(x):
                if available_ranks[i % (SIZE - 1)] == RANK:
                    param_id[:] = i
                    results = self.func(x, **kwargs)
                    gc.collect()
                    try:
                        for rec, res in zip(local_rec, results):
                            rec[:] = res
                    except TypeError:
                        local_rec[0][:] = results
                    COMM.Send([param_id, MPI.INT32_T], dest=MPISRC.MASTER_RANK)
                    if not using_mpio:
                        for rec, var in zip(local_rec, self.__results__):
                            COMM.Send(
                                [rec, self.mpi_dtypes[var]],
                                dest=MPISRC.MASTER_RANK,
                                tag=COMPUTE_TAG,
                            )
            COMM.send(FINISHED, dest=MPISRC.MASTER_RANK, tag=END_TAG)

    def _run_dynamic_load_balancing(self, x: np.ndarray, **kwargs):
        if RANK == MPISRC.MASTER_RANK:
            self._master(x)
        else:
            self._worker(x, **kwargs)

    def _master(self, x: np.ndarray):
        n_parameter_sets = np.size(x, 0)
        complete = 0
        deploy = np.empty(1, dtype="i")
        finish = np.array(-1, dtype="i")
        shapes = self.results_shapes
        dtypes = self.results_dtypes
        param_id = np.empty(1, dtype=np.int32)

        using_mpio = self.using_mpio

        local_rec = None
        if not using_mpio:
            local_rec = [
                np.empty(shapes[var], dtype=dtypes[var]) for var in self.__results__
            ]

        # distribute initial jobs
        self._distribute_initial_jobs(n_parameter_sets, deploy)

        # do rest
        while complete < n_parameter_sets:
            s = MPI.Status()
            COMM.Probe(status=s)
            COMM.Recv([param_id, MPI.INT16_T], source=s.source)
            if not using_mpio:
                for rec, (var, buffer) in zip(local_rec, self.__results__.items()):
                    COMM.Recv(
                        [rec, self.mpi_dtypes[var]], source=s.source, tag=COMPUTE_TAG
                    )
                    buffer[param_id, :] = rec
            if deploy < n_parameter_sets - 1:
                deploy += 1
                COMM.Send([deploy, MPI.INT], dest=s.source)
            else:
                COMM.Send([finish, MPI.INT], dest=s.source)
            self.update_report()
            complete += 1

    def _worker(self, x: np.ndarray, **kwargs):
        deploy = np.empty(1, dtype="i")
        counter = -1
        shapes = self.results_shapes
        dtypes = self.results_dtypes
        gen = enumerate(x)
        param_id = np.empty(1, dtype=np.int32)

        using_mpio = self.using_mpio

        if not using_mpio:
            local_rec = [
                np.empty(shapes[var], dtype=dtypes[var]) for var in self.__results__
            ]
        else:
            local_rec = [ref for ref in self.__results__.values()]

        while True:
            COMM.Recv([deploy, MPI.INT], source=MPISRC.MASTER_RANK)
            if deploy == -1:
                break
            r = None
            while counter != deploy:
                r = next(gen)
                counter += 1
            results = self.func(r[1], **kwargs)
            gc.collect()
            param_id[:] = r[0]
            COMM.Send([param_id, MPI.INT32_T], dest=MPISRC.MASTER_RANK)
            self._submit_results(local_rec, results, using_mpio)

    def _submit_results(
        self,
        local_rec: List[np.ndarray],
        results: Union[Numerical, Tuple[Numerical]],
        using_mpio: bool,
    ):
        try:
            for rec, res in zip(local_rec, results):
                rec[:] = res
        except TypeError:
            local_rec[0][:] = results
        if not using_mpio:
            for rec, var in zip(local_rec, self.__results__):
                COMM.Send(
                    [rec, self.mpi_dtypes[var]],
                    dest=MPISRC.MASTER_RANK,
                    tag=COMPUTE_TAG,
                )

    @staticmethod
    def _distribute_initial_jobs(n: int, deploy: np.ndarray):
        available = SIZE - 1
        size = n if n < available else available
        available_ranks = list(range(SIZE))
        available_ranks.remove(MPISRC.MASTER_RANK)
        for i in range(size):
            deploy[:] = i
            COMM.Send([deploy, MPI.INT], dest=available_ranks[i])

        if size < available:
            deploy[:] = -1
            for j in range(size, available):
                COMM.Send([deploy, MPI.INT], dest=available_ranks[j])
            deploy[:] = size - 1

    def return_attrs(self) -> Sequence[str]:
        ret = self.retval
        return ret if isinstance(ret, (list, tuple)) else (ret,)

    def __return__(self) -> List[Numerical]:
        return [getattr(self, value) for value in self.return_attrs()]

    def _check_return_values(self):
        bad_ref = [v for v in self.return_attrs() if not hasattr(self, v)]
        if bad_ref:
            raise AttributeError(
                f"{self.__class__.__name__} object has no " f"{bad_ref} attribute(s)."
            )


class ANNGPURunner(MPIRunner):
    def _single_process(self, x: np.ndarray, **kwargs):
        try:
            results = self.func(x, **kwargs)
            try:
                for result, buffer in zip(results, self.__results__.values()):
                    buffer[:] = result
            except TypeError:
                getattr(self, self.results_aliases[0])[:] = results
            self.update_report(x.shape[0])
        except:
            for i, params in enumerate(x):
                results = self.func(params, **kwargs)
                try:
                    for result, buffer in zip(results, self.__results__.values()):
                        buffer[i, :] = result
                except TypeError:
                    getattr(self, self.results_aliases[0])[i, :] = results
                self.update_report()

    def _run_static_load_balancing(self, x: np.ndarray, **kwargs):
        return NotImplementedError()

    def _run_dynamic_load_balancing(self, x: np.ndarray, **kwargs):
        return NotImplementedError()


class MPIModelSpec(ModelSpec, MPIRunner):
    """
    This class inherits from both the core MPIRunner and the
    ModelSpec type. This can be used to implement custom
    approximation functions for the behaviours of neural models
    and evaluate them efficiently using message passing primitives
    on HPC clusters.
    """

    def __init__(
        self,
        shape,
        dtype,
        aliases,
        axon_spec,
        extra_spec=None,
        intra_spec=None,
        load_balancing="dynamic",
    ):
        ModelSpec.__init__(self, axon_spec, extra_spec, intra_spec)
        MPIRunner.__init__(self, shape, dtype, aliases, load_balancing)

    @abstractmethod
    def func(self, x, **kwargs):
        pass


class NeuronModel(ModelSpec, MPIRunner):
    _rec_submission = {
        0: "_submit_recordings_mpi_disabled",
        1: "_submit_recordings_mpi_enabled",
        2: "_submit_recordings_mpio",
    }

    def __init__(
        self,
        axon_spec,
        extra_spec=None,
        intra_spec=None,
        shape=None,
        dtype=np.int8,
        aliases="activations",
        load_balancing="dynamic",
        retval=None,
    ):
        aliases = aliases or self.model.__annotations__.get("return", None)
        ModelSpec.__init__(self, axon_spec, extra_spec, intra_spec)
        rec_shape = self._prepare_shape(shape)
        self.__results_shapes = shape if shape is not None else [1]
        MPIRunner.__init__(self, rec_shape, dtype, aliases, load_balancing, retval)

        # -- data structures for optional state variable recording --
        self.record_length = None
        self._record_gids = {}
        self.record_dims = {}
        self.record_vars = {}
        self._record_compartments = {}
        self.record_dtypes = {}
        self.recording(self.axon_specs)

        submission = 1 if MPISRC.ENABLED else 0
        self._submit_recordings = getattr(self, self._rec_submission[submission])

    def _prepare_shape(self, shapes):
        if shapes is None:
            return [(self.size,)]
        out = []
        for shape in shapes:
            if shape == 1 or shape == (1,):
                out.append((self.size,))
            else:
                try:
                    out.append((self.size, *shape))
                except TypeError:  # Handle exception if shape is not iterable
                    out.append((self.size, shape))
        return out

    def run(self, x=None, progressbar=True, **kwargs):
        """
        Core run method. Evaluate the set of parameters 'x'
        using the model specified.

        Parameters
        ----------
        x : np.ndarray
            2D array of parameters.
        progressbar : bool
            Choose whether to track progress of model evaluations
            using a progressbar.
        kwargs
            Additional keyword arguments to pass to model func.

        Returns
        -------
        np.ndarray
            Array of results.
        """
        x = [None] if x is None else x
        if self.ndim:
            x = np.reshape(x, (-1, self.ndim))
        if self.record_vars:
            kwargs["early_stopping"] = False
        return super().run(x, progressbar, **kwargs)

    def distribute(self, x):
        for pid, params in enumerate(x):
            for gid in range(self.size):
                yield ModelFragment(pid, gid, params)

    def n_problem_pieces(self, n):
        return n * self.size

    # -- run method if running on only a single CPU --
    # noinspection PyTypeChecker
    def _single_process(self, x, **kwargs):
        for pid, gid, params in self.distribute(x):
            results = self.func(params, gid=gid, pid=pid, **kwargs)
            gc.collect()
            try:
                for result, buffer in zip(results, self.__results__.values()):
                    buffer[pid, gid] = result
            except TypeError:
                getattr(self, self.results_aliases[0])[pid, gid] = results
            self.update_report()

    # -- different load-balancing paradigms --
    # -- processes are distributed statically across all available CPUs --
    # noinspection PyTypeChecker
    def _run_static_load_balancing(self, x, **kwargs):
        FINISHED = None
        shapes = self.__results_shapes
        dtypes = self.results_dtypes
        pid_gid = np.empty(2, dtype=np.int32)

        available_ranks = list(range(SIZE))
        available_ranks.remove(MPISRC.MASTER_RANK)

        using_mpio = self.using_mpio

        if not using_mpio:
            local_rec = [
                np.empty(shape, dtypes[var])
                for shape, var in zip(shapes, self.__results__)
            ]
        else:
            local_rec = [ref for ref in self.__results__.values()]

        if RANK == MPISRC.MASTER_RANK:
            remaining = SIZE - 1
            while remaining > 0:
                s = MPI.Status()
                COMM.Probe(status=s)
                if s.tag == COMPUTE_TAG:
                    COMM.Recv([pid_gid, MPI.INT32_T], source=s.source)
                    if not using_mpio:
                        self._receive_recordings(pid_gid, s.source)
                        for rec, (var, buffer) in zip(
                            local_rec, self.__results__.items()
                        ):
                            COMM.Recv(
                                [rec, self.mpi_dtypes[var]],
                                source=s.source,
                                tag=COMPUTE_TAG,
                            )
                            buffer[pid_gid[0], pid_gid[1]] = rec
                    self.update_report()
                elif s.tag == END_TAG:
                    COMM.recv(tag=END_TAG)
                    remaining -= 1
        else:
            for pid, gid, params in self.distribute(x):
                if available_ranks[(pid * gid) % (SIZE - 1)] == RANK:
                    pid_gid[:] = pid, gid
                    results = self.func(
                        params, gid=gid, pid_gid_buffer=pid_gid, **kwargs
                    )
                    gc.collect()
                    try:
                        for rec, res in zip(local_rec, results):
                            rec[:] = res
                    except TypeError:
                        local_rec[0][:] = results
                    COMM.Send([pid_gid, MPI.INT32_T], dest=MPISRC.MASTER_RANK)
                    if not using_mpio:
                        for rec, var in zip(local_rec, self.__results__):
                            COMM.Send(
                                [rec, self.mpi_dtypes[var]],
                                dest=MPISRC.MASTER_RANK,
                                tag=COMPUTE_TAG,
                            )
            COMM.send(FINISHED, dest=MPISRC.MASTER_RANK, tag=END_TAG)

    # -- processes are distributed dynamically across all CPUs --
    def _run_dynamic_load_balancing(self, x, **kwargs):
        if RANK == MPISRC.MASTER_RANK:
            self._master(x)
        else:
            self._worker(x, **kwargs)

    def _master(self, x):
        # instantiate buffers
        n = np.size(x, 0) * self.size
        complete = 0
        deploy = np.empty(1, dtype="i")
        finish = np.array(-1, dtype="i")
        shapes = self.__results_shapes
        dtypes = self.results_dtypes
        pid_gid = np.empty(2, dtype=np.int32)

        using_mpio = self.using_mpio

        local_rec = None
        if not using_mpio:
            local_rec = [
                np.empty(shape, dtypes[var])
                for shape, var in zip(shapes, self.__results__)
            ]

        # distribute initial jobs
        self._distribute_initial_jobs(n, deploy)
        # do rest
        while complete < n:
            s = MPI.Status()
            COMM.Probe(status=s)

            # get pid and gid
            COMM.Recv([pid_gid, MPI.INT32_T], source=s.source)

            # get any state variable recordings
            if not using_mpio:
                self._receive_recordings(pid_gid, s.source)

                # finally, get core activation information and update
                for rec, (var, buffer) in zip(local_rec, self.__results__.items()):
                    COMM.Recv(
                        [rec, self.mpi_dtypes[var]], source=s.source, tag=COMPUTE_TAG
                    )
                    buffer[pid_gid[0], pid_gid[1]] = rec

            if deploy < n - 1:
                deploy += 1
                COMM.Send([deploy, MPI.INT], dest=s.source)
            else:
                COMM.Send([finish, MPI.INT], dest=s.source)
            self.update_report()
            complete += 1

    def _worker(self, x, **kwargs):
        counter = -1
        gen = self.distribute(x)

        # instantiate buffers
        deploy = np.empty(1, dtype="i")
        shapes = self.__results_shapes
        dtypes = self.results_dtypes
        pid_gid = np.empty(2, dtype=np.int32)

        using_mpio = self.using_mpio

        if not using_mpio:
            local_rec = [
                np.empty(shape, dtypes[var])
                for shape, var in zip(shapes, self.__results__)
            ]
        else:
            local_rec = [ref for ref in self.__results__.values()]

        pid = gid = params = None
        while True:
            COMM.Recv([deploy, MPI.INT], source=MPISRC.MASTER_RANK)
            if deploy == -1:
                break
            while counter != deploy:
                pid, gid, params = next(gen)
                counter += 1
            pid_gid[:] = pid, gid
            results = self.func(
                params, pid=pid, gid=gid, pid_gid_buffer=pid_gid, **kwargs
            )
            gc.collect()
            self._submit_results(local_rec, results, using_mpio)

    # -- core modeling apparatus --
    def func(self, params, pid=None, gid=None, pid_gid_buffer=None, **kwargs):
        axon, extras, intras, recs = self._build_sim_components(params, pid, gid)
        state_monitors = [rec[0] for rec in recs]
        results = self.model(axon, extras, intras, state_monitors, **kwargs)
        self._submit_recordings(recs, pid_gid_buffer)
        return results

    def _build_sim_components(self, params, pid, gid):
        if params is not None:
            self << params
        axon, intras = self.axon_specs[gid].build_with_intras(self.intra_specs, gid=gid)
        rec_spec = self._record_gids.get(gid, None)
        recs = []
        if rec_spec:
            for var, _, compartment_spec in rec_spec:
                compartments = compartment_spec.build(axon)
                recs.append(
                    (
                        StateMonitor(
                            compartments,
                            self.record_vars[var],
                            dtype=self.record_dtypes[var],
                        ),
                        var,
                        gid,
                        pid,
                    )
                )
        extras = self.extra_specs.build()
        return axon, extras, intras, recs

    @staticmethod
    def model(axon, extras, intras, monitors, **kwargs):
        env = SimulationEnvironment([axon], extras, intras, monitors=monitors)
        env.smartrun(progressbar=False, **kwargs)
        return int(axon.propagated_ap)

    # -- state variable recording --

    def record(
        self,
        state_variable_name,
        variable,
        compartment_spec: list,
        n_compartments,
        dtype="f",
    ):
        """

        Parameters
        ----------
        state_variable_name : str
            Name of state variable in NEURON model to record.
        variable : str
            Name of NeuronModel instance attribute into which to
            record the state variable specified by state_variable_name.
        compartment_spec : list
            list of SectionSpec / CompartmentSpec objects specifying
            from which NEURON Sections / Segments to record.
        n_compartments : int
            Total number of compartments to be recorded from
            per axon, as defined by compartment_spec (this cannot
            be inferred from the compartment_spec directly).
        dtype : str, dtype (optional)
            dtype of record array. By default 'f'.

        Returns
        -------
        None
        """
        if variable in self.record_vars or variable in self.results_aliases:
            raise ValueError(f"{variable} is already being written to.")

        for i, spec in enumerate(compartment_spec):
            # make a record of which state variables will be
            # recorded from this axon and where in the respective
            # array the record will live
            self._record_gids.setdefault(spec.gid, []).append((variable, i, spec))

        # size of the array required to record the state variable
        self.record_dims[variable] = (len(compartment_spec), n_compartments)

        # associate the name of the array with the state-variable
        # that will be recorded within it
        self.record_vars[variable] = state_variable_name

        # array datatype for recording, by default single-precision
        # float
        self.record_dtypes[variable] = dtype

        # MPI datatype for use when message passing between buffers
        # during reassembly on master rank
        self.mpi_dtypes[variable] = self.__dtypes__[dt(dtype)]

    def recording(self, axons):
        """
        Define which state-variables will be recorded and how
        they will be accessed after parallel evaluation of model.
        Use a series of record method calls (one for each state-variable +
        instance attribute into which the variable is to be recorded pair)

        Parameters
        ----------
        axons : list
            List of axon specifications in model.

        Returns
        -------
        None
        """
        pass

    def _submit_recordings_mpio(self, recs, pid_gid):
        COMM.Send([pid_gid, MPI.INT32_T], dest=MPISRC.MASTER_RANK)
        self._submit_recordings_mpi_disabled(recs)

    def _submit_recordings_mpi_enabled(self, recs, pid_gid):
        COMM.Send([pid_gid, MPI.INT32_T], dest=MPISRC.MASTER_RANK)
        for rec in recs:
            monitor, var, _, _ = rec
            COMM.Send(
                [getattr(monitor, self.record_vars[var]), self.mpi_dtypes[var]],
                dest=MPISRC.MASTER_RANK,
            )

    def _submit_recordings_mpi_disabled(self, *args):
        for i, rec in enumerate(args[0]):
            monitor, var, gid, pid = rec
            rec_array = getattr(self, var)
            rec_array[pid, self._record_gids[gid][i][1], :, :] = np.ascontiguousarray(
                getattr(monitor, self.record_vars[var])
            )

    def _receive_recordings(self, pid_gid_buffer, source):
        if self.record_vars:
            pid, gid = pid_gid_buffer
            length = self.record_length
            try:
                for i, var_tuple in enumerate(self._record_gids[gid]):
                    var_name, idx, _ = var_tuple
                    receiver_buff = np.empty(
                        (self.record_dims[var_name][1], length),
                        dtype=self.record_dtypes[var_name],
                    )
                    COMM.Recv([receiver_buff, self.mpi_dtypes[var_name]], source=source)
                    getattr(self, var_name)[pid, idx, :, :] = receiver_buff[:, :]
            except KeyError:
                pass

    def _construct_records(self, np_sets, x, **kwargs):
        super()._construct_records(np_sets, x)
        if self.record_vars:
            runtime = kwargs.get("runtime", getattr(N, "tstop"))
            if isinstance(runtime, float):
                runtime = runtime * ms
            self.record_length = int(runtime / getattr(N, "dt")) + 1
            if self.using_mpio:
                self._mpi_run_group.attrs["tstop"] = runtime
                self._mpi_run_group.attrs["tstop_units"] = str(runtime.units)
                self._mpi_run_group.attrs["dt"] = N.dt
                self._mpi_run_group.attrs["dt_units"] = str(N.dt.units)
            for var in self.record_vars:
                dims = self.record_dims[var]
                dtype = self.record_dtypes[var]
                if not self.using_mpio:
                    setattr(
                        self,
                        var,
                        np.empty(
                            (np_sets, dims[0], dims[1], self.record_length), dtype=dtype
                        ),
                    )
                else:
                    vg = self._mpi_run_group.create_dataset(
                        var,
                        (np_sets, dims[0], dims[1], self.record_length),
                        dtype=dtype,
                    )
                    setattr(self, var, vg)

    def records_as_contiguous_array(self):
        return np.stack([getattr(self, var) for var in self.record_vars], axis=-2)

    def attrs(self):
        attrs_ = {
            "axons": [repr(spec) for spec in self.axon_specs],
            "intras": [repr(spec) for spec in self.intra_specs]
            if self.intra_specs
            else None,
            "extras": [repr(spec) for spec in self.extra_specs]
            if self.extra_specs
            else None,
            "mutable": [repr(spec) for spec in self._mutable_specs]
            if self._mutable_specs
            else None,
        }
        return attrs_

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mpi_file:
            self._mpi_file.close()

    # -- mpio stuff --

    def mpio(self, handle, mode="a", attrs=None):
        super().mpio(handle, mode=mode, attrs=attrs)
        if self._mpi_model_group:
            self._mpi_model_group.attrs["n_axons"] = self.size
            self._mpi_model_group.attrs["n_extra"] = len(self.extra_specs)
            self._mpi_model_group.attrs["n_intra"] = len(self.extra_specs)
            self._mpi_model_group.attrs["n_dim"] = self.ndim
            self._mpi_model_group.attrs["record_vars"] = f"{self.record_vars}"
            if MPISRC.ENABLED:
                self._submit_recordings = getattr(self, self._rec_submission[2])
            else:
                self._submit_recordings = getattr(self, self._rec_submission[0])
        return self

    def standard_io(self):
        self.mpio_close()
        return self

    def mpio_close(self):
        super().mpio_close()
        self._submit_recordings = getattr(
            self, self._rec_submission[1 if MPISRC.ENABLED else 0]
        )

    # -- alternative constructor --

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kwargs):
        return cls(spec.axon_specs, spec.extra_specs, spec.intra_specs, **kwargs)
