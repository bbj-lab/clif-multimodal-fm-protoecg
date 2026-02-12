"""Evaluation visualization: AUROC comparison, calibration curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_auroc_comparison(
    route_metrics: dict[str, list],
    horizon: int = 48,
    output_path: Path | str = "figures/auroc_comparison.pdf",
) -> None:
    """Bar chart comparing AUROCs across routes for each label.

    Args:
        route_metrics: Dict of route_name -> list of LabelMetrics.
        horizon: Which horizon to plot.
        output_path: Where to save the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect labels present at this horizon
    all_labels: set[str] = set()
    for metrics_list in route_metrics.values():
        for m in metrics_list:
            if m.horizon_hours == horizon and m.auroc is not None:
                all_labels.add(m.label)

    labels = sorted(all_labels)
    if not labels:
        return

    routes = list(route_metrics.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(routes)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))

    for i, route in enumerate(routes):
        aurocs = []
        for label in labels:
            metric = next(
                (
                    m
                    for m in route_metrics[route]
                    if m.label == label and m.horizon_hours == horizon
                ),
                None,
            )
            aurocs.append(metric.auroc if metric and metric.auroc else 0)
        ax.bar(x + i * width, aurocs, width, label=route)

    ax.set_xticks(x + width * (len(routes) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Label Prediction Performance ({horizon}h Horizon)")
    ax.legend()
    ax.set_ylim(0.5, 1.0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(
    predictions: list[float],
    labels: list[bool],
    n_bins: int = 10,
    output_path: Path | str = "figures/calibration.pdf",
    title: str = "Calibration Curve",
) -> None:
    """Plot predicted probability vs observed frequency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_mids = []
    bin_freqs = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = [(lo <= p <= hi) for p in predictions]
        else:
            mask = [(lo <= p < hi) for p in predictions]
        count = sum(mask)
        if count == 0:
            continue
        bin_preds = [p for p, m in zip(predictions, mask) if m]
        bin_labels = [float(y) for y, m in zip(labels, mask) if m]
        bin_mids.append(np.mean(bin_preds))
        bin_freqs.append(np.mean(bin_labels))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_mids, bin_freqs, "o-", label="Model")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.legend()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
