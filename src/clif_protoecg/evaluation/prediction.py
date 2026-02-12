"""Prediction window creation and probability extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PredictionWindow:
    """A midnight-aligned prediction window."""

    window_id: int
    context_ids: list[int]
    context_end_time: datetime
    hospitalization_id: str


def create_midnight_windows(
    token_ids: list[int],
    timestamps: list[datetime],
    midnight_token_id: int,
    hospitalization_id: str = "",
    max_days: int = 7,
) -> list[PredictionWindow]:
    """Create prediction windows at each midnight clock token.

    Each window uses all tokens up to (and including) midnight as context.
    """
    windows: list[PredictionWindow] = []

    for i, (tid, ts) in enumerate(zip(token_ids, timestamps)):
        if tid == midnight_token_id and len(windows) < max_days:
            windows.append(
                PredictionWindow(
                    window_id=len(windows),
                    context_ids=token_ids[: i + 1],
                    context_end_time=ts,
                    hospitalization_id=hospitalization_id,
                )
            )

    return windows


def extract_predictions(
    generated_samples: list[list[int]],
    label_token_ids: dict[str, int],
) -> dict[str, float]:
    """Extract predicted probabilities via Monte Carlo proportion.

    P(label) = (# samples containing label token) / N
    """
    n_samples = len(generated_samples)
    if n_samples == 0:
        return {name: 0.0 for name in label_token_ids}

    predictions: dict[str, float] = {}
    for label_name, label_id in label_token_ids.items():
        count = sum(1 for sample in generated_samples if label_id in sample)
        predictions[label_name] = count / n_samples

    return predictions


def extract_predictions_with_censoring(
    samples: list[list[int]],
    label_token_ids: dict[str, int],
    death_token_ids: set[int],
    discharge_token_ids: set[int],
) -> dict[str, float]:
    """Extract predictions with death/discharge censoring.

    For mortality labels: death token presence = positive.
    For other labels: label token presence = positive.
    """
    n_samples = len(samples)
    if n_samples == 0:
        return {name: 0.0 for name in label_token_ids}

    predictions: dict[str, float] = {}
    for label_name, label_id in label_token_ids.items():
        if "mortality" in label_name:
            count = sum(
                1
                for s in samples
                if any(t in death_token_ids for t in s)
            )
        else:
            count = sum(1 for s in samples if label_id in s)
        predictions[label_name] = count / n_samples

    return predictions


def get_ground_truth(
    token_ids: list[int],
    timestamps: list[datetime],
    context_end_time: datetime,
    horizon_hours: int,
    label_token_ids: dict[str, int],
) -> dict[str, bool]:
    """Extract ground truth labels within the prediction horizon.

    Returns dict of label_name -> whether label occurred within
    [context_end_time, context_end_time + horizon_hours].
    """
    horizon_end = context_end_time + timedelta(hours=horizon_hours)
    ground_truth = {name: False for name in label_token_ids}

    for tid, ts in zip(token_ids, timestamps):
        if ts <= context_end_time:
            continue
        if ts > horizon_end:
            break
        for label_name, label_id in label_token_ids.items():
            if tid == label_id:
                ground_truth[label_name] = True

    return ground_truth
