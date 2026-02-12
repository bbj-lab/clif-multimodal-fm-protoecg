"""AUROC metric with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(
    predictions: list[float],
    labels: list[bool],
) -> float | None:
    """Compute AUROC, returning None for degenerate cases."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float(roc_auc_score(labels, predictions))


def bootstrap_auroc(
    predictions: list[float],
    labels: list[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap confidence interval.

    Returns (auroc_median, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(seed)
    n = len(predictions)
    aurocs: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_preds = [predictions[i] for i in idx]
        boot_labels = [labels[i] for i in idx]
        auroc = compute_auroc(boot_preds, boot_labels)
        if auroc is not None:
            aurocs.append(auroc)

    if not aurocs:
        return 0.0, 0.0, 0.0

    alpha = (1 - confidence) / 2
    return (
        float(np.median(aurocs)),
        float(np.percentile(aurocs, alpha * 100)),
        float(np.percentile(aurocs, (1 - alpha) * 100)),
    )
