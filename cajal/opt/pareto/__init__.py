"""Functions for MOOPs"""

__all__ = ["nds", "dominates", "crowding", "tournament_crowding"]

from cajal.opt.pareto.nds import nds, dominates
from cajal.opt.pareto.ops import crowding, tournament_crowding
