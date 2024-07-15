"""Classes for composition of Parameter and Prediction losses."""

from abc import ABC, abstractmethod
from itertools import chain

from iteration_utilities import deepflatten
import numpy as np

from cajal.common.logging import logger
from cajal.opt.loss._parameterloss import ParameterLoss, Lambda, CustomParameterLoss


class LossWrapper(ABC):
    """Generic abstract Wrapper class."""

    loss_type = "wrapper"

    def __init__(self, *args):
        self.losses = args
        self.required = list(
            set(chain.from_iterable([loss.required for loss in self.losses]))
        )
        self.__scale = 1

    def __call__(self, **kwargs):
        self._check_call_args(**kwargs)
        return self.__scale * self.call(
            self._flatten([self._call_hook(loss, **kwargs) for loss in self.losses])
        )

    def _check_call_args(self, **kwargs):
        if not all([r in kwargs for r in self.required]):
            raise KeyError(
                "Not all required arguments {} were found.".format(self.required)
            )

    @staticmethod
    def _flatten(iterable):
        return list(deepflatten(iterable))

    def _call_hook(self, loss, **kwargs):
        if loss.loss_type == "output":
            return self._output_loss_hook(loss, **kwargs)
        elif loss.loss_type == "parameter":
            return self._parameter_loss_hook(loss, **kwargs)
        elif loss.loss_type == "vector":
            return self._vector_loss_hook(loss, **kwargs)
        else:
            return loss(**kwargs)

    @staticmethod
    def _output_loss_hook(loss, **kwargs):
        try:
            out = loss(*kwargs["output"])
        except KeyError:
            txt = (
                "OutputLoss losses expect an `output` "
                + "argument but none was found. Returning 0."
            )
            out = 0
            logger.warning(txt)
        return out

    @staticmethod
    def _parameter_loss_hook(loss, **kwargs):
        try:
            parameter_args = kwargs["parameter_args"]
            out = [
                loss(p[0], **p[1])
                for p in parameter_args
                if (loss.apply_to is None or p[0] in loss.apply_to)
            ]
        except KeyError:
            txt = (
                "ParameterLoss losses expect a `parameter_args` "
                "argument but none was given. Returning 0."
            )
            out = 0
            logger.warning(txt)
        return out

    @staticmethod
    def _vector_loss_hook(loss, **kwargs):
        try:
            out = loss(kwargs["pop_vec"])
        except KeyError:
            txt = (
                "VectorLoss losses expect a `pop_vec` argument "
                "but none was given. Returning 0."
            )
            out = 0
            logger.warning(txt)
        return out

    def __mul__(self, val):
        self.__scale *= val
        return self

    @abstractmethod
    def call(self, losses):
        """Function to apply to results of dispatched losses."""


class Sum(LossWrapper):
    """Sum over losses."""

    def call(self, losses):
        return np.sum(losses)


class WeightedSum(LossWrapper):
    """Weights over all losses in wrapper."""

    def __init__(self, *args, weights=None):
        super().__init__(*args)
        self.weights = np.array(weights)

    def call(self, losses):
        return np.sum(np.array(losses) * self.weights)


class ParameterLossWrapper(LossWrapper):
    """Wrap ParameterLoss objects to control dispatch."""

    def __init__(self, *args):
        super(ParameterLossWrapper, self).__init__(*args)
        self._check_losses()

    def _check_losses(self):
        if not all([isinstance(loss, ParameterLoss) for loss in self.losses]):
            raise TypeError("This wrapper can only contain ParameterLoss objects.")

    def call(self, losses):
        return np.array(losses)


class SharedContext(ParameterLossWrapper):
    """Deliver parameters over all electrodes as arrays to Lambda
    loss objects."""

    def _check_losses(self):
        if not all(
            [isinstance(loss, (Lambda, CustomParameterLoss)) for loss in self.losses]
        ):
            raise TypeError(
                f"{self.__class__.__name__} can only contain Lambda "
                f"or CustomParameterLoss objects."
            )

    @staticmethod
    def _to_arrays(p_args, args, stimtypes=None):
        dct = {
            k: np.concatenate(
                [
                    np.atleast_1d(p[1][k])
                    for p in p_args
                    if stimtypes is None or p[0] in stimtypes
                ]
            )
            for k in args
        }
        return dct

    def _call_hook(self, loss, **kwargs):
        parameter_args = kwargs.get("parameter_args", None)
        if not parameter_args:
            txt = (
                "SharedContext wrapper expects `parameter_args` "
                "argument but none was given. Returning 0."
            )
            out = 0
            logger.warning(txt)
            return out
        return loss(**self._to_arrays(parameter_args, loss.args, loss.apply_to))


class MultiObjective(LossWrapper):
    """Return list of objective function values for multi-objective
    optimisation."""

    def __init__(self, *args, ndim):
        super().__init__(*args)
        self.ndim = ndim

    def call(self, losses):
        return np.array(losses)
