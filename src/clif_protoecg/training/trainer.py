"""Training loop with label loss weighting and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from clif_protoecg.utils.logging import get_logger

logger = get_logger("trainer")


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    label_loss_weight: float = 75.0
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-3
    eval_steps: int = 500
    log_steps: int = 50
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    max_seq_len: int | None = None
    chunk_overlap: int = 256
    max_tokens_per_batch: int = 32768
    pad_id: int = 0
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    val_max_batches: int | None = None


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    label_weight: float = 75.0,
    pad_id: int = 0,
) -> torch.Tensor:
    """Cross-entropy with boosted weight for label tokens and loss masking."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_label_mask = label_mask[:, 1:]

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=pad_id,
    ).view(shift_labels.shape)

    weights = torch.where(shift_label_mask, label_weight, 1.0)

    # Zero out loss on overlap regions (loss_mask=0 means skip)
    if loss_mask is not None:
        shift_loss_mask = loss_mask[:, 1:]
        weights = weights * shift_loss_mask

    return (loss * weights).sum() / weights.sum().clamp(min=1)


def train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainerConfig,
    step: int,
) -> float:
    """Single training step. Returns loss value."""
    device = config.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    label_mask = batch["label_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device) if "loss_mask" in batch else None

    with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=config.bf16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        loss = weighted_cross_entropy(
            outputs.logits, input_ids, label_mask,
            loss_mask=loss_mask,
            label_weight=config.label_loss_weight,
            pad_id=config.pad_id,
        )

    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps

    loss.backward()

    if (step + 1) % config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return loss.item()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainerConfig,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_steps = 0
    for step, batch in enumerate(loader):
        total_loss += train_step(model, batch, optimizer, scheduler, config, step)
        n_steps += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TrainerConfig,
    max_batches: int | None = None,
) -> float:
    """Evaluate on validation set. Returns average loss.

    Args:
        max_batches: If set, only evaluate on this many batches (for speed
            with large val sets). Uses config.val_max_batches as fallback.
    """
    model.eval()
    total_loss = 0.0
    n_steps = 0
    device = config.device
    limit = max_batches if max_batches is not None else config.val_max_batches

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_mask = batch["label_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device) if "loss_mask" in batch else None

        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=config.bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss = weighted_cross_entropy(
                outputs.logits, input_ids, label_mask,
                loss_mask=loss_mask,
                label_weight=config.label_loss_weight,
                pad_id=config.pad_id,
            )
        total_loss += loss.item()
        n_steps += 1
        if limit is not None and n_steps >= limit:
            break

    return total_loss / max(n_steps, 1)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: TrainerConfig,
    total_steps: int,
) -> tuple[AdamW, Any]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - config.warmup_steps, 1),
        eta_min=config.learning_rate * 0.1,
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[config.warmup_steps]
    )
    return optimizer, scheduler
