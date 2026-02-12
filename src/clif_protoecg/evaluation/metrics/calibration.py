"""Calibration metrics: Brier score, ECE, AUPRC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def compute_brier(predictions: list[float], labels: list[bool]) -> float:
    """Brier score (mean squared error between predicted prob and outcome)."""
    return float(np.mean([(p - float(y)) ** 2 for p, y in zip(predictions, labels)]))


def compute_ece(
    predictions: list[float],
    labels: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
    if not predictions:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(predictions)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # Last bin includes right boundary
        if i == n_bins - 1:
            mask = [(lo <= p <= hi) for p in predictions]
        else:
            mask = [(lo <= p < hi) for p in predictions]
        count = sum(mask)

        if count == 0:
            continue

        bin_preds = [p for p, m in zip(predictions, mask) if m]
        bin_labels = [float(y) for y, m in zip(labels, mask) if m]

        avg_pred = np.mean(bin_preds)
        avg_label = np.mean(bin_labels)

        ece += abs(avg_pred - avg_label) * (count / n_total)

    return float(ece)


def compute_auprc(predictions: list[float], labels: list[bool]) -> float | None:
    """Area Under Precision-Recall Curve."""
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    return float(average_precision_score(labels, predictions))
