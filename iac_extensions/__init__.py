"""Optional research extensions for the IAC training stack.

The modules in this package are deliberately dependency-light and opt-in.  An
existing IAC config behaves exactly as before unless it enables an extension.
"""

from .dino_motion_head import (
    CandidateBlindDinoMotionHead,
    UncertaintyAwareTrajectoryComparator,
    uncertainty_weighted_motion_loss,
)
from .flow_evidence import (
    ClassicFlowExtractor,
    FLOW_METHODS,
    RidgeSpeedHead,
    TorchvisionRaftFlowExtractor,
    flow_statistics,
    make_flow_extractor,
    speed_energy,
    trajectory_speed_targets,
)
from .rgb_motion_head import CandidateBlindRgbDiffMotionHead

__all__ = [
    "CandidateBlindDinoMotionHead",
    "CandidateBlindRgbDiffMotionHead",
    "UncertaintyAwareTrajectoryComparator",
    "uncertainty_weighted_motion_loss",
    "ClassicFlowExtractor",
    "FLOW_METHODS",
    "RidgeSpeedHead",
    "TorchvisionRaftFlowExtractor",
    "flow_statistics",
    "make_flow_extractor",
    "speed_energy",
    "trajectory_speed_targets",
]
