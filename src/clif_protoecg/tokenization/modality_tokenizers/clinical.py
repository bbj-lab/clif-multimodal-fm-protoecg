"""Clinical event tokenizer (labs, vitals, meds, etc.)."""

from __future__ import annotations

from clif_protoecg.tokenization.vocabulary import Vocabulary


class ClinicalTokenizer:
    """Tokenize clinical events with optional quantile binning."""

    def __init__(self, fuse_bins: bool = False) -> None:
        self.fuse_bins = fuse_bins

    def tokenize_event(self, event: dict, vocab: Vocabulary) -> list[str]:
        code = event["code"]
        tokens = [code]

        val = event.get("value")
        if val is not None and isinstance(val, (int, float)):
            bin_idx = vocab.quantile_bins.get_bin(code, float(val))
            if bin_idx is not None:
                if self.fuse_bins:
                    tokens = [f"{code}/Q_{bin_idx}"]
                else:
                    tokens.append(f"Q_{bin_idx}")

        return tokens
