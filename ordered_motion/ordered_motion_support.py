"""Visibility-aware, three-state decisions for ordered-motion evidence.

This module consumes the segment ledger emitted by the ordered-motion scorer.
It never reads source labels.  Labels may be used by separate validation-only
calibration and reporting tools, but inference depends only on visual evidence,
candidate residuals, visibility, and predictive uncertainty.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class SupportDecisionConfig:
    """Frozen validation-set thresholds for three-state inference."""

    support_energy_max: float
    unsupported_energy_min: float
    min_evidence_coverage: float = 0.6
    max_mean_normalized_uncertainty: float | None = None
    require_uncertainty: bool = False

    def validate(self) -> None:
        values = (self.support_energy_max, self.unsupported_energy_min)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("energy thresholds must be finite")
        if self.support_energy_max >= self.unsupported_energy_min:
            raise ValueError(
                "support_energy_max must be lower than unsupported_energy_min"
            )
        if not 0.0 <= self.min_evidence_coverage <= 1.0:
            raise ValueError("min_evidence_coverage must be in [0, 1]")
        maximum = self.max_mean_normalized_uncertainty
        if maximum is not None and (
            not math.isfinite(float(maximum)) or float(maximum) <= 0.0
        ):
            raise ValueError(
                "max_mean_normalized_uncertainty must be positive and finite"
            )
        if self.require_uncertainty and maximum is None:
            raise ValueError(
                "require_uncertainty needs max_mean_normalized_uncertainty"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SupportDecisionConfig":
        raw = value.get("support_decision_config", value)
        if not isinstance(raw, Mapping):
            raise ValueError("support decision config must be an object")
        config = cls(
            support_energy_max=float(raw["support_energy_max"]),
            unsupported_energy_min=float(raw["unsupported_energy_min"]),
            min_evidence_coverage=float(raw.get("min_evidence_coverage", 0.6)),
            max_mean_normalized_uncertainty=(
                float(raw["max_mean_normalized_uncertainty"])
                if raw.get("max_mean_normalized_uncertainty") is not None
                else None
            ),
            require_uncertainty=bool(raw.get("require_uncertainty", False)),
        )
        config.validate()
        return config


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unit_weight(value: Any, *, default: float = 1.0) -> float:
    result = _finite_float(value)
    if result is None:
        result = default
    return min(max(result, 0.0), 1.0)


def _segment_visibility(
    row: Mapping[str, Any],
    segment: Mapping[str, Any],
    segment_index: int,
) -> tuple[float, str]:
    row_values = row.get("ordered_motion_segment_visibility")
    if isinstance(row_values, Sequence) and not isinstance(
        row_values, (str, bytes)
    ):
        if segment_index < len(row_values):
            return _unit_weight(row_values[segment_index]), "row"
    if "visibility" in segment:
        return _unit_weight(segment.get("visibility")), "segment"
    return 1.0, "implicit_full"


def _normalized_uncertainty(component: Mapping[str, Any]) -> float | None:
    for key in (
        "normalized_uncertainty",
        "visual_normalized_standard_deviation",
    ):
        value = _finite_float(component.get(key))
        if value is not None:
            return max(value, 0.0)
    return None


def aggregate_segment_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate visible segment evidence while preserving the energy scale.

    With all visibility weights equal to one, ``visibility_aware_energy`` is
    exactly the sum of per-family mean segment contributions used by the
    ordered-motion model.  Partial visibility removes unobserved components and
    renormalizes by observed mass, so missing evidence does not look like a
    good match.
    """

    ledger = row.get("ordered_motion_segment_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError(
            "row needs a non-empty ordered_motion_segment_ledger; rerun the "
            "upstream scorer with --include-segment-ledger"
        )

    weighted_energy = 0.0
    observed_mass = 0.0
    component_count = 0
    family_count = 0
    uncertainty_weighted_sum = 0.0
    uncertainty_mass = 0.0
    visibility_sources: Counter[str] = Counter()

    for segment_index, raw_segment in enumerate(ledger):
        if not isinstance(raw_segment, Mapping):
            raise ValueError(f"segment {segment_index} must be an object")
        components = raw_segment.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"segment {segment_index} has no components")
        family_count = max(family_count, len(components))
        segment_weight, source = _segment_visibility(
            row, raw_segment, segment_index
        )
        visibility_sources[source] += 1

        for component_index, raw_component in enumerate(components):
            if not isinstance(raw_component, Mapping):
                raise ValueError(
                    f"segment {segment_index} component {component_index} "
                    "must be an object"
                )
            energy = _finite_float(raw_component.get("energy_contribution"))
            if energy is None or energy < 0.0:
                raise ValueError(
                    f"segment {segment_index} component {component_index} "
                    "needs a finite non-negative energy_contribution"
                )
            component_weight = _unit_weight(
                raw_component.get("visibility"), default=1.0
            )
            weight = segment_weight * component_weight
            component_count += 1
            observed_mass += weight
            weighted_energy += weight * energy

            uncertainty = _normalized_uncertainty(raw_component)
            if uncertainty is not None and weight > 0.0:
                uncertainty_weighted_sum += weight * uncertainty
                uncertainty_mass += weight

    coverage = observed_mass / component_count if component_count else 0.0
    energy = (
        weighted_energy / observed_mass * family_count
        if observed_mass > 0.0
        else None
    )
    uncertainty = (
        uncertainty_weighted_sum / uncertainty_mass
        if uncertainty_mass > 0.0
        else None
    )
    return {
        "visibility_aware_energy": energy,
        "evidence_coverage": coverage,
        "mean_normalized_uncertainty": uncertainty,
        "uncertainty_coverage": (
            uncertainty_mass / observed_mass if observed_mass > 0.0 else 0.0
        ),
        "segment_count": len(ledger),
        "component_count": component_count,
        "visibility_sources": dict(sorted(visibility_sources.items())),
    }


def classify_support(
    evidence: Mapping[str, Any],
    config: SupportDecisionConfig,
) -> tuple[str, str]:
    """Return the state and a stable machine-readable reason."""

    config.validate()
    energy = _finite_float(evidence.get("visibility_aware_energy"))
    coverage = _finite_float(evidence.get("evidence_coverage")) or 0.0
    uncertainty = _finite_float(evidence.get("mean_normalized_uncertainty"))

    if energy is None:
        return INSUFFICIENT_EVIDENCE, "no_observed_evidence"
    if coverage < config.min_evidence_coverage:
        return INSUFFICIENT_EVIDENCE, "low_evidence_coverage"
    if config.max_mean_normalized_uncertainty is not None:
        if uncertainty is None and config.require_uncertainty:
            return INSUFFICIENT_EVIDENCE, "missing_uncertainty"
        if (
            uncertainty is not None
            and uncertainty > config.max_mean_normalized_uncertainty
        ):
            return INSUFFICIENT_EVIDENCE, "high_predictive_uncertainty"
    if energy <= config.support_energy_max:
        return SUPPORTED, "energy_below_support_threshold"
    if energy >= config.unsupported_energy_min:
        return UNSUPPORTED, "energy_above_unsupported_threshold"
    return INSUFFICIENT_EVIDENCE, "decision_margin"


def score_row(
    row: Mapping[str, Any],
    config: SupportDecisionConfig,
) -> dict[str, Any]:
    evidence = aggregate_segment_evidence(row)
    state, reason = classify_support(evidence, config)
    result = dict(row)
    result.update(
        {
            "ordered_motion_support_state": state,
            "ordered_motion_support_reason": reason,
            "ordered_motion_support_config": config.to_dict(),
            **{f"ordered_motion_{key}": value for key, value in evidence.items()},
        }
    )
    return result


def summarize_scored_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    coverage_by_state: dict[str, list[float]] = defaultdict(list)
    energy_by_state: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        state = str(row["ordered_motion_support_state"])
        reason = str(row["ordered_motion_support_reason"])
        state_counts[state] += 1
        reason_counts[reason] += 1
        coverage = _finite_float(row.get("ordered_motion_evidence_coverage"))
        energy = _finite_float(row.get("ordered_motion_visibility_aware_energy"))
        if coverage is not None:
            coverage_by_state[state].append(coverage)
        if energy is not None:
            energy_by_state[state].append(energy)

    total = len(rows)
    decided = state_counts[SUPPORTED] + state_counts[UNSUPPORTED]
    return {
        "rows": total,
        "state_counts": dict(sorted(state_counts.items())),
        "state_fractions": {
            key: value / total if total else float("nan")
            for key, value in sorted(state_counts.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_coverage": decided / total if total else float("nan"),
        "mean_evidence_coverage_by_state": {
            key: sum(values) / len(values)
            for key, values in sorted(coverage_by_state.items())
            if values
        },
        "mean_energy_by_state": {
            key: sum(values) / len(values)
            for key, values in sorted(energy_by_state.items())
            if values
        },
    }


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    """Return the Wilson score lower bound for a Bernoulli precision."""

    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("correct and total must satisfy 0 <= correct <= total")
    if z <= 0.0:
        raise ValueError("z must be positive")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def calibrate_energy_thresholds(
    records: Sequence[tuple[float, bool]],
    *,
    min_supported_precision: float,
    min_unsupported_precision: float,
    min_supported_precision_lower_bound: float | None = None,
    min_unsupported_precision_lower_bound: float | None = None,
    confidence_z: float = 1.96,
) -> dict[str, Any]:
    """Select disjoint energy tails with frozen minimum precision.

    ``True`` labels mean supported.  Low energy predicts supported and high
    energy predicts unsupported.  The selected pair maximizes classified rows
    while meeting both precision constraints; the middle remains abstained.
    """

    if not 0.0 < min_supported_precision <= 1.0:
        raise ValueError("min_supported_precision must be in (0, 1]")
    if not 0.0 < min_unsupported_precision <= 1.0:
        raise ValueError("min_unsupported_precision must be in (0, 1]")
    for name, value in (
        ("min_supported_precision_lower_bound", min_supported_precision_lower_bound),
        ("min_unsupported_precision_lower_bound", min_unsupported_precision_lower_bound),
    ):
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
    cleaned = [
        (float(energy), bool(label))
        for energy, label in records
        if math.isfinite(float(energy))
    ]
    if not cleaned or not any(label for _, label in cleaned) or not any(
        not label for _, label in cleaned
    ):
        raise ValueError("calibration needs finite supported and unsupported rows")

    thresholds = sorted({energy for energy, _ in cleaned})
    support_options: list[dict[str, Any]] = []
    unsupported_options: list[dict[str, Any]] = []
    supported_total = sum(label for _, label in cleaned)
    unsupported_total = len(cleaned) - supported_total
    for threshold in thresholds:
        low = [label for energy, label in cleaned if energy <= threshold]
        if low:
            correct = sum(low)
            precision = correct / len(low)
            lower_bound = wilson_lower_bound(correct, len(low), confidence_z)
            if precision >= min_supported_precision and (
                min_supported_precision_lower_bound is None
                or lower_bound >= min_supported_precision_lower_bound
            ):
                support_options.append(
                    {
                        "threshold": threshold,
                        "count": len(low),
                        "precision": precision,
                        "precision_lower_bound": lower_bound,
                        "recall": sum(low) / supported_total,
                    }
                )
        high = [label for energy, label in cleaned if energy >= threshold]
        if high:
            true_unsupported = sum(not label for label in high)
            precision = true_unsupported / len(high)
            lower_bound = wilson_lower_bound(
                true_unsupported, len(high), confidence_z
            )
            if precision >= min_unsupported_precision and (
                min_unsupported_precision_lower_bound is None
                or lower_bound >= min_unsupported_precision_lower_bound
            ):
                unsupported_options.append(
                    {
                        "threshold": threshold,
                        "count": len(high),
                        "precision": precision,
                        "precision_lower_bound": lower_bound,
                        "recall": true_unsupported / unsupported_total,
                    }
                )

    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for support in support_options:
        for unsupported in unsupported_options:
            if support["threshold"] >= unsupported["threshold"]:
                continue
            classified = support["count"] + unsupported["count"]
            score = (
                float(classified),
                min(float(support["precision"]), float(unsupported["precision"])),
                float(unsupported["threshold"] - support["threshold"]),
            )
            candidates.append((score, support, unsupported))
    if not candidates:
        raise ValueError(
            "no disjoint support/unsupported thresholds meet the precision gates"
        )
    _, support, unsupported = max(candidates, key=lambda item: item[0])
    classified = int(support["count"] + unsupported["count"])
    return {
        "support_energy_max": float(support["threshold"]),
        "unsupported_energy_min": float(unsupported["threshold"]),
        "labeled_rows": len(cleaned),
        "classified_rows": classified,
        "decision_coverage": classified / len(cleaned),
        "abstained_rows": len(cleaned) - classified,
        "supported_tail": support,
        "unsupported_tail": unsupported,
        "minimum_precision": {
            "supported": min_supported_precision,
            "unsupported": min_unsupported_precision,
        },
        "minimum_precision_lower_bound": {
            "supported": min_supported_precision_lower_bound,
            "unsupported": min_unsupported_precision_lower_bound,
            "confidence_z": confidence_z,
        },
    }
