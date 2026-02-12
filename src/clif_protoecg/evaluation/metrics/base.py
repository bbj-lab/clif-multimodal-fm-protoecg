"""Base metric protocol."""

from __future__ import annotations

from typing import Protocol


class Metric(Protocol):
    """Protocol for evaluation metrics."""

    def compute(
        self, predictions: list[float], labels: list[bool]
    ) -> float | None: ...
