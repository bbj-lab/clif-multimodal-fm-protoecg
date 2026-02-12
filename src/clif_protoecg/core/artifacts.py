"""Serialisable artifacts produced by pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class SplitInfo:
    """Patient-level train/val/test split info."""

    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> SplitInfo:
        data = json.loads(path.read_text())
        return cls(**data)


@dataclass
class CodeProfile:
    """Code frequency profile from the training set."""

    # code -> {event_count, admission_count}
    code_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    included_codes: set[str] = field(default_factory=set)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "code_stats": self.code_stats,
            "included_codes": sorted(self.included_codes),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> CodeProfile:
        data = json.loads(path.read_text())
        return cls(
            code_stats=data["code_stats"],
            included_codes=set(data["included_codes"]),
        )


@dataclass
class BinEdges:
    """Discretisation bin edges per code."""

    edges: dict[str, list[float]] = field(default_factory=dict)

    def get_bin(self, code: str, value: float) -> int | None:
        """Map a numeric value to its bin index for a given code."""
        code_edges = self.edges.get(code)
        if code_edges is None:
            return None
        for i in range(1, len(code_edges)):
            if value <= code_edges[i]:
                return i - 1
        return len(code_edges) - 2

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.edges, indent=2))

    @classmethod
    def load(cls, path: Path) -> BinEdges:
        return cls(edges=json.loads(path.read_text()))


@dataclass
class TrainingMetrics:
    """Metrics captured during model training."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float("inf")
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> TrainingMetrics:
        return cls(**json.loads(path.read_text()))
