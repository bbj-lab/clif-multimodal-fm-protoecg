"""Base protocol for modality tokenizers."""

from __future__ import annotations

from typing import Protocol

from clif_protoecg.tokenization.vocabulary import Vocabulary


class BaseModalityTokenizer(Protocol):
    """Protocol for modality-specific tokenizers."""

    def tokenize_event(
        self, event: dict, vocab: Vocabulary
    ) -> list[str]:
        """Convert a single event dict to a list of token strings."""
        ...
