"""Tokenization stage: build vocab, tokenize sequences, write output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
from tqdm import tqdm

from clif_protoecg.core.constants import SPECIAL_TOKENS
from clif_protoecg.tokenization.vocabulary import Vocabulary, VocabularyBuilder
from clif_protoecg.tokenization.modality_tokenizers.clinical import ClinicalTokenizer
from clif_protoecg.tokenization.modality_tokenizers.ecg import ECGTokenizer
from clif_protoecg.tokenization.modality_tokenizers.time import TimeTokenizer
from clif_protoecg.utils.streaming import StreamingParquetWriter


def _get_tokenizer(code: str, clinical: ClinicalTokenizer, ecg: ECGTokenizer, time_tok: TimeTokenizer):
    if code.startswith("CLOCK//") or code.startswith("DT//"):
        return time_tok
    if code.startswith("ECG//"):
        return ecg
    return clinical


def tokenize_events(
    events: list[dict],
    vocab: Vocabulary,
    fuse_bins: bool = False,
) -> list[int]:
    """Convert event dicts to token IDs."""
    clinical = ClinicalTokenizer(fuse_bins=fuse_bins)
    ecg = ECGTokenizer()
    time_tok = TimeTokenizer()

    token_strings: list[str] = ["[BOS]"]
    for e in events:
        tokenizer = _get_tokenizer(e["code"], clinical, ecg, time_tok)
        token_strings.extend(tokenizer.tokenize_event(e, vocab))
    token_strings.append("[EOS]")

    return vocab.encode(token_strings)


def tokenize_events_with_timestamps(
    events: list[dict],
    vocab: Vocabulary,
    fuse_bins: bool = False,
) -> tuple[list[int], list[int]]:
    """Convert event dicts to token IDs with parallel timestamps.

    Returns (token_ids, timestamps_us) where timestamps_us are microsecond
    epoch values (one per token).  BOS/EOS inherit the first/last event time.
    """
    from datetime import datetime, timezone

    clinical = ClinicalTokenizer(fuse_bins=fuse_bins)
    ecg = ECGTokenizer()
    time_tok = TimeTokenizer()

    token_strings: list[str] = []
    token_times_us: list[int] = []

    first_time = events[0]["time"] if events else datetime(2000, 1, 1, tzinfo=timezone.utc)
    last_time = events[-1]["time"] if events else first_time

    # BOS
    token_strings.append("[BOS]")
    token_times_us.append(_dt_to_us(first_time))

    for e in events:
        tokenizer = _get_tokenizer(e["code"], clinical, ecg, time_tok)
        toks = tokenizer.tokenize_event(e, vocab)
        ts_us = _dt_to_us(e["time"])
        token_strings.extend(toks)
        token_times_us.extend([ts_us] * len(toks))

    # EOS
    token_strings.append("[EOS]")
    token_times_us.append(_dt_to_us(last_time))

    return vocab.encode(token_strings), token_times_us


def _dt_to_us(dt) -> int:
    """Convert datetime to microseconds since epoch."""
    from datetime import timezone

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def build_vocabulary_from_sequences(
    sequences: list[list[dict]],
    n_bins: int = 10,
    fuse_bins: bool = False,
    min_count: int = 1,
) -> Vocabulary:
    """Build vocabulary from training event sequences."""
    builder = VocabularyBuilder(
        n_bins=n_bins,
        fuse_numeric_bins=fuse_bins,
        min_count=min_count,
    )
    for seq in tqdm(sequences, desc="Building vocabulary", unit="seq", mininterval=0.5):
        builder.add_events(seq)
    return builder.build()


_OUTPUT_SCHEMA = pa.schema([
    ("hospitalization_id", pa.string()),
    ("token_ids", pa.list_(pa.int32())),
    ("timestamps_us", pa.list_(pa.int64())),
    ("token_count", pa.int32()),
])


def tokenize_and_write(
    sequences: dict[str, list[dict]],
    vocab: Vocabulary,
    output_path: Path,
    fuse_bins: bool = False,
) -> int:
    """Tokenize sequences and write to Parquet.

    Saves token_ids with parallel timestamps_us (microsecond epoch) for
    each token, needed by the evaluation stage.

    Args:
        sequences: {hospitalization_id: event_list}
        vocab: Built vocabulary
        output_path: Output parquet file path
        fuse_bins: Whether to use fused bin tokens

    Returns:
        Number of sequences written.
    """
    total_tokens = 0
    with StreamingParquetWriter(output_path, _OUTPUT_SCHEMA) as writer:
        pbar = tqdm(sequences.items(), desc="Tokenizing", unit="seq", total=len(sequences), mininterval=0.5)
        for hid, events in pbar:
            token_ids, timestamps_us = tokenize_events_with_timestamps(
                events, vocab, fuse_bins
            )
            writer.write_row({
                "hospitalization_id": hid,
                "token_ids": token_ids,
                "timestamps_us": timestamps_us,
                "token_count": len(token_ids),
            })
            total_tokens += len(token_ids)
            pbar.set_postfix(tokens=f"{total_tokens:,}")
    return len(sequences)
