"""Structured logging helpers using Rich."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with Rich formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_path=False,
            markup=True,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def log_stage_start(stage_name: str) -> None:
    logger = get_logger("pipeline")
    logger.info(f"[bold cyan]Starting stage:[/bold cyan] {stage_name}")


def log_stage_end(stage_name: str, duration: float) -> None:
    logger = get_logger("pipeline")
    logger.info(
        f"[bold green]Completed stage:[/bold green] {stage_name} "
        f"in {duration:.1f}s"
    )
