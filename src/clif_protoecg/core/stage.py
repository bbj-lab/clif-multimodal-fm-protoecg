"""Base stage protocol and result types for the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of input/output validation for a stage."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class StageResult:
    """Result of running a pipeline stage."""

    success: bool
    output_path: Path | None = None
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseStage(ABC):
    """Abstract base class for pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name."""

    def validate_input(self, path: Path) -> ValidationResult:
        """Validate that required input files exist and are readable."""
        result = ValidationResult()
        if not path.exists():
            result.add_error(f"Input path does not exist: {path}")
        return result

    def validate_output(self, path: Path) -> ValidationResult:
        """Validate that output was produced correctly."""
        result = ValidationResult()
        if not path.exists():
            result.add_error(f"Output path does not exist: {path}")
        return result

    @abstractmethod
    def run(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> StageResult:
        """Execute the stage."""
