"""
`mutable` provides classes to specify which input parameters
to your model can be manipulated at runtime (e.g. during optimisation
or parameter sweeps).
"""
import numpy as np

from cajal.common.math import round_nearest


class Mutable:
    optimisable = False

    def __init__(self, shape=None, order="C", dtype="f"):
        self.shape = shape
        self.order = order
        self.ndim = np.prod(self.shape or 1)
        self._value = None
        self.dtype = dtype

    def parse(self, values):
        val = np.reshape(values, self.shape, self.order).astype(self.dtype)
        self._value = val.item() if np.size(val) == 1 else val
        return self._value

    @property
    def value(self):
        return self._value


class OptNumerical(Mutable):
    optimisable = True

    def __init__(self, bounds, shape=None, order="C", rounding=None, dtype="f"):
        super().__init__(shape, order, dtype)
        if self.ndim != len(bounds):
            raise ValueError(
                f"shape {shape or 1} and # of bounds ({len(bounds)}) " f"disagree."
            )
        self.bounds = bounds
        self.rounding = rounding

    def parse(self, values):
        return super().parse(round_nearest(values, self.rounding))


class OptCategorical(OptNumerical):
    def __init__(self, categories):
        super().__init__([(0, 1)] * categories, categories)

    def parse(self, values):
        return np.argmax(np.exp(values) / np.sum(np.exp(values)))
