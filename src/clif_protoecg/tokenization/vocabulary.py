"""Vocabulary builder and encoder."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clif_protoecg.core.constants import SPECIAL_TOKENS


@dataclass
class QuantileBins:
    """Per-code bin edges for numeric value discretization."""

    edges: dict[str, list[float]] = field(default_factory=dict)

    def get_bin(self, code: str, value: float) -> int | None:
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
    def load(cls, path: Path) -> QuantileBins:
        return cls(edges=json.loads(path.read_text()))


@dataclass
class Vocabulary:
    """Token <-> ID mapping with quantile bin support."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    quantile_bins: QuantileBins = field(default_factory=QuantileBins)

    def encode(self, tokens: list[str]) -> list[int]:
        unk_id = self.token_to_id.get("[UNK]", 1)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, "[UNK]") for i in ids]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id.get("[PAD]", 0)

    @property
    def bos_id(self) -> int:
        return self.token_to_id.get("[BOS]", 2)

    @property
    def eos_id(self) -> int:
        return self.token_to_id.get("[EOS]", 3)

    @property
    def cont_id(self) -> int | None:
        return self.token_to_id.get("[CONT]")

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "vocab.json").write_text(
            json.dumps(self.token_to_id, indent=2)
        )
        self.quantile_bins.save(output_dir / "quantile_bins.json")

    @classmethod
    def load(cls, vocab_dir: Path) -> Vocabulary:
        token_to_id = json.loads((vocab_dir / "vocab.json").read_text())
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        qb_path = vocab_dir / "quantile_bins.json"
        qb = QuantileBins.load(qb_path) if qb_path.exists() else QuantileBins()
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, quantile_bins=qb)


class VocabularyBuilder:
    """Build a vocabulary from training event sequences."""

    def __init__(
        self,
        n_bins: int = 10,
        fuse_numeric_bins: bool = False,
        min_count: int = 1,
    ) -> None:
        self.n_bins = n_bins
        self.fuse_numeric_bins = fuse_numeric_bins
        self.min_count = min_count
        self._token_counts: Counter = Counter()
        self._numeric_values: dict[str, list[float]] = {}

    def add_events(self, events: list[dict]) -> None:
        """Process events from one hospitalization."""
        for e in events:
            code = e["code"]
            self._token_counts[code] += 1
            val = e.get("value")
            if val is not None and isinstance(val, (int, float)):
                self._numeric_values.setdefault(code, []).append(float(val))

    def build(self) -> Vocabulary:
        """Build the final vocabulary."""
        # 1. Special tokens
        token_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)

        # 2. Compute quantile bins for numeric codes
        quantile_edges: dict[str, list[float]] = {}
        for code, vals in self._numeric_values.items():
            if len(vals) >= self.n_bins:
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.unique(np.percentile(vals, percentiles)).tolist()
                quantile_edges[code] = edges

        # 3. Add all tokens meeting min_count
        for token, count in sorted(self._token_counts.items()):
            if count >= self.min_count and token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        # 4. Add bin tokens (Q_0 through Q_n-1)
        max_bins = max(
            (len(e) - 1 for e in quantile_edges.values()),
            default=self.n_bins,
        )
        for i in range(max(max_bins, self.n_bins)):
            tok = f"Q_{i}"
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        # 5. If fused, add fused variants
        if self.fuse_numeric_bins:
            for code in quantile_edges:
                n_bins = len(quantile_edges[code]) - 1
                for i in range(n_bins):
                    fused = f"{code}/Q_{i}"
                    if fused not in token_to_id:
                        token_to_id[fused] = len(token_to_id)

        id_to_token = {v: k for k, v in token_to_id.items()}

        return Vocabulary(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            quantile_bins=QuantileBins(edges=quantile_edges),
        )
