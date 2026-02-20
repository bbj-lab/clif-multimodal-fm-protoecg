"""Abstract base class for inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    n_samples: int = 8
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95


def truncate_at_clock_limit(
    token_ids: list[int],
    clock_ids: set[int],
    max_clocks: int,
) -> list[int]:
    """Truncate a generated sequence at the Nth clock token."""
    count = 0
    for i, tid in enumerate(token_ids):
        if tid in clock_ids:
            count += 1
            if count >= max_clocks:
                return token_ids[: i + 1]
    return token_ids


class InferenceEngine(ABC):
    """Abstract interface for model inference."""

    @abstractmethod
    def generate(
        self,
        context_ids: list[int],
        config: GenerationConfig,
        stop_token_ids: set[int] | None = None,
        clock_token_ids: set[int] | None = None,
        max_clock_tokens: int | None = None,
    ) -> list[list[int]]:
        """Generate N samples from a context.

        Args:
            context_ids: Token IDs for the prefix/context.
            config: Generation parameters.
            stop_token_ids: Tokens that trigger immediate stop (EOS, death, discharge).
            clock_token_ids: Clock tokens for horizon-based stopping.
            max_clock_tokens: Stop after this many clock tokens are generated.

        Returns:
            List of N generated token sequences (excluding context).
        """
        ...
