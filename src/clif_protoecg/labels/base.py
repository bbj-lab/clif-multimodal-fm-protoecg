"""Label registry with @label decorator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_LABEL_REGISTRY: dict[str, LabelDefinition] = {}


@dataclass
class LabelDefinition:
    """Metadata + compute function for a single label."""

    name: str
    token: str
    category: str  # "clif_only", "clif_partial", "mimic_required"
    description: str
    source_tables: list[str]
    compute_fn: Callable[..., dict | None]


def label(
    name: str,
    token: str,
    category: str,
    description: str,
    source_tables: list[str] | None = None,
) -> Callable:
    """Decorator that registers a label compute function."""

    def decorator(fn: Callable) -> Callable:
        _LABEL_REGISTRY[name] = LabelDefinition(
            name=name,
            token=token,
            category=category,
            description=description,
            source_tables=source_tables or [],
            compute_fn=fn,
        )
        return fn

    return decorator


def get_registry() -> dict[str, LabelDefinition]:
    """Return a copy of the global label registry."""
    return dict(_LABEL_REGISTRY)
