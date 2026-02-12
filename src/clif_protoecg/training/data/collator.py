"""Dynamic batch collator using token budget."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Sampler


class DynamicBatchCollator:
    """Collate variable-length sequences with padding.

    Pads to the longest sequence in the batch.
    """

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        label_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        loss_mask = torch.zeros(len(batch), max_len, dtype=torch.float)

        for i, b in enumerate(batch):
            seq_len = b["input_ids"].size(0)
            input_ids[i, :seq_len] = b["input_ids"]
            attention_mask[i, :seq_len] = 1
            if "label_mask" in b:
                label_mask[i, :seq_len] = b["label_mask"]
            if "loss_mask" in b:
                loss_mask[i, :seq_len] = b["loss_mask"]
            else:
                loss_mask[i, :seq_len] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "loss_mask": loss_mask,
        }


def create_dynamic_batches(
    lengths: list[int],
    max_tokens_per_batch: int = 32768,
) -> list[list[int]]:
    """Group indices into batches targeting a token budget.

    *lengths* is a list where ``lengths[i]`` is the token count of item *i*.
    """
    indexed = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for idx in indexed:
        seq_len = lengths[idx]
        if current_tokens + seq_len > max_tokens_per_batch and current:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += seq_len

    if current:
        batches.append(current)
    return batches


class DynamicBatchSampler(Sampler[list[int]]):
    """Yields pre-computed batch index lists, optionally shuffled."""

    def __init__(self, batches: list[list[int]], shuffle: bool = False) -> None:
        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self) -> int:
        return len(self.batches)
