"""Time token tokenizer (clock + gap)."""

from __future__ import annotations

from clif_protoecg.tokenization.vocabulary import Vocabulary


class TimeTokenizer:
    """Pass-through tokenizer for CLOCK// and DT// tokens."""

    def tokenize_event(self, event: dict, vocab: Vocabulary) -> list[str]:
        return [event["code"]]
