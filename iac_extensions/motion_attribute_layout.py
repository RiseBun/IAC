"""Names and evidence families for IAC's structured motion attributes.

The trajectory-rule target in ``train_dinov2_v5_minimal.py`` contains twelve
global attributes followed by eight attributes for every temporal segment.
Historically those values were exposed only as an anonymous vector.  This
module gives every dimension a stable name and groups dimensions into four
human-readable evidence families:

``longitudinal``
    Forward progress, path length, mean step length and speed change.
``lateral``
    Signed/absolute lateral displacement.
``heading``
    Signed/absolute yaw change and curvature.
``path_shape``
    Directness and when a turn occurs.

The families are exhaustive and disjoint.  Consequently, family
contributions can be summed exactly back to the scalar motion energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


MOTION_FAMILY_NAMES: Tuple[str, ...] = (
    "longitudinal",
    "lateral",
    "heading",
    "path_shape",
)

GLOBAL_ATTRIBUTE_LAYOUT: Tuple[Tuple[str, str], ...] = (
    ("final_forward", "longitudinal"),
    ("final_lateral", "lateral"),
    ("abs_final_lateral", "lateral"),
    ("path_length", "longitudinal"),
    ("directness", "path_shape"),
    ("mean_step_length", "longitudinal"),
    ("late_minus_early_speed", "longitudinal"),
    ("yaw_change", "heading"),
    ("abs_yaw_change", "heading"),
    ("max_abs_lateral", "lateral"),
    ("curvature", "heading"),
    ("turn_timing", "path_shape"),
)

SEGMENT_ATTRIBUTE_LAYOUT: Tuple[Tuple[str, str], ...] = (
    ("forward_delta", "longitudinal"),
    ("lateral_delta", "lateral"),
    ("end_lateral", "lateral"),
    ("path_length", "longitudinal"),
    ("mean_step_length", "longitudinal"),
    ("yaw_change", "heading"),
    ("abs_yaw_change", "heading"),
    ("curvature", "heading"),
)


@dataclass(frozen=True)
class MotionAttributeLayout:
    """Stable names and family membership for one configured motion vector."""

    attribute_names: Tuple[str, ...]
    attribute_families: Tuple[str, ...]
    family_names: Tuple[str, ...] = MOTION_FAMILY_NAMES

    @property
    def attribute_dim(self) -> int:
        return len(self.attribute_names)

    def family_indices(self) -> Dict[str, Tuple[int, ...]]:
        return {
            family: tuple(
                index
                for index, value in enumerate(self.attribute_families)
                if value == family
            )
            for family in self.family_names
        }


def build_motion_attribute_layout(segment_count: int) -> MotionAttributeLayout:
    """Return the named layout for ``12 + 8 * segment_count`` attributes."""

    segment_count = max(0, int(segment_count))
    names: List[str] = [name for name, _ in GLOBAL_ATTRIBUTE_LAYOUT]
    families: List[str] = [family for _, family in GLOBAL_ATTRIBUTE_LAYOUT]
    for segment in range(segment_count):
        for name, family in SEGMENT_ATTRIBUTE_LAYOUT:
            names.append(f"segment_{segment}_{name}")
            families.append(family)
    return MotionAttributeLayout(tuple(names), tuple(families))


def aggregate_motion_family_contributions(
    weighted_component_energy: torch.Tensor,
    *,
    segment_count: int,
) -> torch.Tensor:
    """Aggregate weighted component energies without losing additivity.

    ``UncertaintyAwareTrajectoryComparator`` defines total energy as the mean
    over all weighted attributes.  Each returned family value is therefore the
    sum of that family's weighted components divided by the full attribute
    dimension.  Summing the final axis exactly reconstructs total energy.
    """

    if weighted_component_energy.ndim < 1:
        raise ValueError("weighted_component_energy must have at least one axis")
    layout = build_motion_attribute_layout(segment_count)
    if weighted_component_energy.shape[-1] != layout.attribute_dim:
        raise ValueError(
            "motion attribute width does not match the configured layout: "
            f"{weighted_component_energy.shape[-1]} vs {layout.attribute_dim}"
        )
    family_values: List[torch.Tensor] = []
    indices_by_family = layout.family_indices()
    denominator = float(layout.attribute_dim)
    for family in layout.family_names:
        indices: Sequence[int] = indices_by_family[family]
        index = torch.tensor(
            indices,
            dtype=torch.long,
            device=weighted_component_energy.device,
        )
        family_values.append(
            weighted_component_energy.index_select(-1, index).sum(dim=-1)
            / denominator
        )
    return torch.stack(family_values, dim=-1)
