"""Per-label, per-horizon metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from clif_protoecg.evaluation.metrics.auroc import compute_auroc
from clif_protoecg.evaluation.metrics.calibration import (
    compute_auprc,
    compute_brier,
    compute_ece,
)


@dataclass
class PredictionResult:
    """A single prediction for one label at one window."""

    patient_id: str
    window_id: int
    label: str
    horizon_hours: int
    predicted_prob: float
    actual_positive: bool


@dataclass
class LabelMetrics:
    """Aggregated metrics for one label at one horizon."""

    label: str
    horizon_hours: int
    auroc: float | None
    auprc: float | None
    brier: float
    ece: float
    n_positive: int
    n_negative: int
    n_total: int


def aggregate_metrics(
    results: list[PredictionResult],
) -> list[LabelMetrics]:
    """Compute metrics per label per horizon."""
    groups: dict[tuple[str, int], list[PredictionResult]] = defaultdict(list)
    for r in results:
        groups[(r.label, r.horizon_hours)].append(r)

    metrics: list[LabelMetrics] = []
    for (label, horizon), group in sorted(groups.items()):
        preds = [r.predicted_prob for r in group]
        labels = [r.actual_positive for r in group]
        n_pos = sum(labels)

        metrics.append(
            LabelMetrics(
                label=label,
                horizon_hours=horizon,
                auroc=compute_auroc(preds, labels),
                auprc=compute_auprc(preds, labels),
                brier=compute_brier(preds, labels),
                ece=compute_ece(preds, labels),
                n_positive=n_pos,
                n_negative=len(labels) - n_pos,
                n_total=len(labels),
            )
        )

    return metrics
