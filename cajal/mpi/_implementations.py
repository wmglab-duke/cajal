from cajal.mpi._core import NeuronModel
from cajal.nrn import SimulationEnvironment


__all__ = "Thresholder", "BlockThresholder"


class Thresholder(NeuronModel):

    """Calculate fibre thresholds in parallel."""

    def __init__(
        self, axon_spec, extra_spec=None, intra_spec=None, load_balancing="dynamic"
    ):
        super().__init__(
            axon_spec,
            extra_spec,
            intra_spec,
            dtype="f",
            aliases="thresholds",
            load_balancing=load_balancing,
        )

    @staticmethod
    def model(axon, extras, intras, recs, **kwargs):
        thresh = SimulationEnvironment([axon], extras, intras).find_thresh(**kwargs)
        return thresh


class BlockThresholder(NeuronModel):
    def __init__(
        self, axon_spec, extra_spec=None, intra_spec=None, load_balancing="dynamic"
    ):
        super().__init__(
            axon_spec,
            extra_spec,
            intra_spec,
            dtype="f",
            aliases="block_thresholds",
            load_balancing=load_balancing,
        )

    @staticmethod
    def model(axon, extras, intras, recs, **kwargs):
        thresh = SimulationEnvironment([axon], extras, intras).find_block_thresh(
            **kwargs
        )
        return thresh
