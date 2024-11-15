"""
Losses that can be applied to stimulus parameters.
"""
import inspect
from abc import ABC, abstractmethod

import numpy as np
from sympy import sympify, lambdify
from scipy import integrate
from neuron import h


class ParameterLoss(ABC):
    """Abstract base ParameterLoss class."""

    loss_type = "parameter"
    required = ["parameter_args"]

    def __init__(self, apply_to=None):
        self.__scale = 1
        self.apply_to = None
        if apply_to is not None:
            if not isinstance(apply_to, (list, tuple)):
                apply_to = (apply_to,)
            self.apply_to = apply_to

    def __call__(self, *args, **kwargs):
        return self.__scale * self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        """Loss function."""

    def __rmul__(self, val):
        self.__scale = val
        return self


class CustomParameterLoss(ParameterLoss):
    """
    Create your own ParameterLoss.
        ```
        class MyParameterLoss(CustomParameterLoss):
            def call(parameter_name_1, parameter_name_2, ...):
                ...define operations on parameters...
                return loss
        ```
    """

    def __init__(self, apply_to=None):
        super().__init__(apply_to)
        self.args = inspect.getfullargspec(self.call).args[1:]

    def __call__(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self.args}
        return super().__call__(**kwargs)

    @abstractmethod
    def call(self, *args):  # pylint: disable=arguments-differ
        """Define custom loss based on parameter names."""


class Lambda(ParameterLoss):
    """Define custom losses over stimulus parameters."""

    def __init__(self, equation, apply_to=None, **kwargs):
        super(Lambda, self).__init__(apply_to)

        self.__func = sympify(equation)
        self._check_symbols(**kwargs)
        self.args = inspect.getfullargspec(self.__func).args

    def _check_symbols(self, **kwargs):
        self.__func = self.__func.subs(kwargs)
        self.__func = lambdify(self.__func.free_symbols, self.__func, "numpy")

    def call(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self.args}
        return self.__func(**kwargs)


class VectorLoss(ParameterLoss):
    """Abstract base class to define custom losses applied directly to
    population vectors."""

    loss_type = "vector"
    required = ["pop_vec"]

    @abstractmethod
    def call(self, x):  # pylint: disable=arguments-differ
        """Loss logic."""


class TimecourseLoss(ParameterLoss):
    """Abstract base class for losses that operate on the stimulus
    timecourse vector itself."""

    def call(self, stim_class, **kwargs):  # pylint: disable=arguments-differ
        st_cls = stim_class(**kwargs)
        t_course = st_cls.timecourse_
        return self.timecourse_func(t_course)

    @abstractmethod
    def timecourse_func(self, t_course):
        """Function that is applied to timecourse vector."""


class Integral(TimecourseLoss):
    """Numerically integrate over timecourse array of a given stimulus."""

    def __init__(self, apply_to=None, method="trapezoid"):
        super(Integral, self).__init__(apply_to)
        self.integrate_func = getattr(integrate, method)

    def timecourse_func(self, t_course):
        return self.integrate_func(t_course, dx=h.dt)


class Energy(Integral):
    """Calculate total energy from stimulus timecourse."""

    def timecourse_func(self, t_course):
        return super().timecourse_func(np.square(t_course))
