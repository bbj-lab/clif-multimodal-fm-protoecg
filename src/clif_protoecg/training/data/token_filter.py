"""ECG token filter for route-based ablation."""

from __future__ import annotations


class ECGTokenFilter:
    """Filter ECG tokens from sequences based on ablation route.

    Tokenize once with all_branches, then filter per-route at training time.
    """

    ROUTE_ALLOWED_PREFIXES: dict[str, set[str] | None] = {
        "no_ecg": set(),
        "fusion_class": {"ECG//Class", "ECG//Prototype/1D", "ECG//Similarity/1D"},
        "all_branches": None,  # Keep all
    }

    def __init__(self, vocab: dict[str, int], route: str) -> None:
        self.route = route
        allowed = self.ROUTE_ALLOWED_PREFIXES.get(route)
        if allowed is None:
            # Keep all tokens
            self.filtered_ids: frozenset[int] = frozenset()
        else:
            self.filtered_ids = frozenset(
                tid
                for token, tid in vocab.items()
                if token.startswith("ECG//")
                and not any(token.startswith(p) for p in allowed)
            )

    def filter(self, token_ids: list[int]) -> list[int]:
        if not self.filtered_ids:
            return token_ids
        return [t for t in token_ids if t not in self.filtered_ids]
