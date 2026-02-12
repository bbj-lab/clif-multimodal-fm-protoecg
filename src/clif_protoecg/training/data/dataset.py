"""PyTorch dataset for tokenized medical sequences with chunk support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset


class MedicalEventDataset(Dataset):
    """Dataset of tokenized hospitalization sequences.

    Long sequences are split into overlapping chunks.  Continuation chunks
    (chunk_idx > 0) are prefixed with ``cont_token_id`` so the model can
    distinguish them from new-patient starts.  A ``loss_mask`` is returned
    that zeros out the overlap region to prevent double-counting loss.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        max_seq_len: int = 2048,
        chunk_overlap: int = 256,
        label_token_ids: set[int] | None = None,
        cont_token_id: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.chunk_overlap = chunk_overlap
        self.label_token_ids = label_token_ids or set()
        self.cont_token_id = cont_token_id

        # Build chunk map: list of (seq_idx, start_pos, overlap_start)
        self._chunk_map: list[tuple[int, int, int]] = []
        for seq_idx, seq in enumerate(sequences):
            if len(seq) <= max_seq_len or chunk_overlap == 0:
                self._chunk_map.append((seq_idx, 0, 0))
            else:
                stride = max_seq_len - chunk_overlap
                pos = 0
                chunk_idx = 0
                while pos < len(seq):
                    overlap_start = 0 if chunk_idx == 0 else chunk_overlap
                    self._chunk_map.append((seq_idx, pos, overlap_start))
                    # Stop if this chunk already reaches the end
                    if pos + max_seq_len >= len(seq):
                        break
                    pos += max(stride, 1)
                    chunk_idx += 1

    def __len__(self) -> int:
        return len(self._chunk_map)

    def chunk_lengths(self) -> list[int]:
        """Return the token length of each chunk (for dynamic batching)."""
        lengths = []
        for seq_idx, start, _ in self._chunk_map:
            end = min(start + self.max_seq_len, len(self.sequences[seq_idx]))
            chunk_len = end - start
            if start > 0 and self.cont_token_id is not None:
                chunk_len = min(chunk_len + 1, self.max_seq_len)
            lengths.append(chunk_len)
        return lengths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start, overlap_start = self._chunk_map[idx]
        seq = self.sequences[seq_idx]
        end = min(start + self.max_seq_len, len(seq))
        chunk = seq[start:end]

        # Continuation chunks: prepend [CONT]
        is_continuation = start > 0 and self.cont_token_id is not None
        if is_continuation:
            if len(chunk) == self.max_seq_len:
                chunk = [self.cont_token_id] + chunk[:-1]
            else:
                chunk = [self.cont_token_id] + chunk

        label_mask = [tid in self.label_token_ids for tid in chunk]

        # Loss mask: 0 for overlap region (don't double-count), 1 elsewhere
        if is_continuation and overlap_start > 0:
            # [CONT] at pos 0 contributes to loss; overlap tokens masked
            loss_mask = [1] + [0] * (overlap_start - 1) + [1] * (len(chunk) - overlap_start)
        else:
            loss_mask = [0] * overlap_start + [1] * (len(chunk) - overlap_start)

        return {
            "input_ids": torch.tensor(chunk, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
        }

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        max_seq_len: int = 2048,
        chunk_overlap: int = 256,
        label_token_ids: set[int] | None = None,
        cont_token_id: int | None = None,
    ) -> MedicalEventDataset:
        df = pl.read_parquet(path)
        sequences = df["token_ids"].to_list()
        return cls(sequences, max_seq_len, chunk_overlap, label_token_ids, cont_token_id)
