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
    RidgeSpeedHead,
    flow_statistics,
    speed_energy,
    trajectory_speed_targets,
)

__all__ = [
    "CandidateBlindDinoMotionHead",
    "UncertaintyAwareTrajectoryComparator",
    "uncertainty_weighted_motion_loss",
    "ClassicFlowExtractor",
    "RidgeSpeedHead",
    "flow_statistics",
    "speed_energy",
    "trajectory_speed_targets",
]
