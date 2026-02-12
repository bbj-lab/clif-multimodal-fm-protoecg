"""ECG event tokenizer."""

from __future__ import annotations

from clif_protoecg.tokenization.vocabulary import Vocabulary


class ECGTokenizer:
    """Tokenize ECG events with similarity binning."""

    def tokenize_event(self, event: dict, vocab: Vocabulary) -> list[str]:
        code = event["code"]
        tokens = [code]

        # Similarity tokens get quantile bins
        if code.startswith("ECG//Similarity/"):
            val = event.get("value")
            if val is not None:
                bin_idx = vocab.quantile_bins.get_bin(code, float(val))
                if bin_idx is not None:
                    tokens.append(f"Q_{bin_idx}")

        return tokens
