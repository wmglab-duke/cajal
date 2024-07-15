from ._differentialevolution import DEBASE


class CheckPoint:
    def __init__(self, optimizer: DEBASE):
        self.limits = optimizer.limits
        self.population = optimizer.population
