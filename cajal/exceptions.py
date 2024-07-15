"""Custom Exceptions."""


class ArgumentError(Exception):
    pass


class Uninitialised(Exception):
    pass


class StimTypeMismatch(Exception):
    pass


class BlockThresholdTopInitError(Exception):
    pass


class BlockThresholdBottomInitError(Exception):
    pass


class ThresholdTopInitError(Exception):
    pass


class ThresholdBottomInitError(Exception):
    pass


class Unoptimisable(Exception):
    pass
