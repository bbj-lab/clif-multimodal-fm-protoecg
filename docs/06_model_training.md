# Step 6: Model Training

## Goal

Train a causal language model on tokenized clinical timelines. The model learns to predict the next token in a hospitalization sequence, including clinical events, labels, ECG prototypes, and temporal markers. Three ECG routes (`no_ecg`, `fusion_class`, `all_branches`) enable ablation comparison of ECG integration strategies.

## Architecture

### Transformer Design (from Reference Repository)

The model uses a **Gemma-style decoder-only transformer** with:

- **Grouped-Query Attention (GQA)** for computational efficiency
- **Rotary Position Embeddings (RoPE)** for sequence position encoding
- **SwiGLU feed-forward networks** (gated linear units with SiLU activation)
- **RMSNorm** for layer normalization
- **Tied embeddings** (input embedding weights shared with output projection)

### HuggingFace Backend (Recommended)

The reference repository wraps LlamaForCausalLM for vLLM compatibility:

```python
from transformers import LlamaConfig, LlamaForCausalLM


def create_model(vocab_size: int, size_name: str = "tiny") -> LlamaForCausalLM:
    """Create a model using HuggingFace LlamaForCausalLM.

    This enables:
    - vLLM inference support out-of-the-box
    - Flash attention via HF's built-in support
    - save_pretrained/from_pretrained for easy model sharing
    """
    SIZE_PRESETS = {
        "tiny":   {"d_model": 128,  "n_layers": 4,  "n_heads": 4,  "n_kv_heads": 2,  "d_ff": 512},
        "small":  {"d_model": 256,  "n_layers": 8,  "n_heads": 8,  "n_kv_heads": 4,  "d_ff": 1024},
        "medium": {"d_model": 512,  "n_layers": 12, "n_heads": 16, "n_kv_heads": 4,  "d_ff": 2048},
        "large":  {"d_model": 768,  "n_layers": 12, "n_heads": 12, "n_kv_heads": 4,  "d_ff": 3072},
    }

    preset = SIZE_PRESETS[size_name]

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=preset["d_model"],
        intermediate_size=preset["d_ff"],
        num_hidden_layers=preset["n_layers"],
        num_attention_heads=preset["n_heads"],
        num_key_value_heads=preset["n_kv_heads"],
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        use_cache=True,
    )

    return LlamaForCausalLM(config)
```

### Model Size Summary

| Size | Params | d_model | Layers | Heads | KV Heads | d_ff | Context |
|------|--------|---------|--------|-------|----------|------|---------|
| Tiny | ~3M | 128 | 4 | 4 | 2 | 512 | 8192 |
| Small | ~8M | 256 | 8 | 8 | 4 | 1024 | 8192 |
| Medium | ~48M | 512 | 12 | 16 | 4 | 2048 | 16384 |
| Large | ~105M | 768 | 12 | 12 | 4 | 3072 | 32768 |

## Tokenization and Vocabulary

### Building the Vocabulary

From the reference repository's approach:

```python
import json
from pathlib import Path
from collections import Counter


def build_vocabulary(
    sequences: list[list[str]],
    min_count: int = 1,
) -> dict[str, int]:
    """Build vocabulary from training sequences.

    Vocabulary is built from training data only to prevent data leakage.
    """
    # Special tokens (reserved IDs 0-4)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "[SEP]": 4,
    }

    # Count token frequencies in training data
    counts = Counter()
    for seq in sequences:
        counts.update(seq)

    # Add tokens meeting minimum frequency
    next_id = len(vocab)
    for token, count in sorted(counts.items()):
        if count >= min_count and token not in vocab:
            vocab[token] = next_id
            next_id += 1

    return vocab
```

### Numeric Value Discretization

Continuous values (lab results, vitals) are discretized into bins (Q_0 through Q_{n-1}) using the bin mode configured in Step 1 (`hybrid`, `clinical_only`, or `quantile`; default `hybrid` with 10 bins). Bin edges are computed on training data only:

```python
import numpy as np


def compute_decile_bins(
    values: list[float],
    n_bins: int = 10,
) -> list[float]:
    """Compute bin edges from training data values.

    Returns n_bins + 1 edges (11 values for 10 bins).
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(values, percentiles).tolist()


def get_bin(value: float, edges: list[float]) -> int:
    """Map a numeric value to its bin index (0 to n_bins-1)."""
    for i in range(1, len(edges)):
        if value <= edges[i]:
            return i - 1
    return len(edges) - 2
```

Two tokenization strategies for numeric values:

1. **Unfused** (default): Two separate tokens
   ```
   LAB_RESULT//glucose Q_7
   ```

2. **Fused**: Single combined token
   ```
   LAB_RESULT//glucose/Q_7
   ```

## Training Configuration

### Optimizer Settings (from Reference Repository)

```python
optimizer_config = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "warmup_steps": 100,       # Linear warmup
    "max_grad_norm": 1.0,      # Gradient clipping
}
```

### Learning Rate Schedule

Linear warmup followed by cosine annealing:

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def create_scheduler(optimizer, warmup_steps, total_steps, min_lr_factor=0.1):
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=optimizer.defaults["lr"] * min_lr_factor,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
```

### Label Loss Weighting

Clinical tokens (labels, ICD codes, discharge) are rare in the training data. The reference repository applies a **label loss weight** (e.g., 75x) to boost their contribution:

```python
LABEL_TOKEN_PREFIXES = ["LABEL//", "DISCH//", "ICD//", "PROC//"]


def create_label_mask(
    token_ids: list[int],
    vocab: dict[str, int],
) -> list[bool]:
    """Create a boolean mask identifying clinical label tokens."""
    id_to_token = {v: k for k, v in vocab.items()}
    return [
        any(id_to_token.get(tid, "").startswith(prefix)
            for prefix in LABEL_TOKEN_PREFIXES)
        for tid in token_ids
    ]


def weighted_cross_entropy(
    logits,       # (batch, seq_len, vocab_size)
    labels,       # (batch, seq_len)
    label_mask,   # (batch, seq_len) bool
    label_weight: float = 75.0,
    pad_id: int = 0,
):
    """Cross-entropy with boosted weight for clinical tokens."""
    import torch
    import torch.nn.functional as F

    # Shift for causal LM (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = label_mask[:, 1:]

    # Per-token cross-entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=pad_id,
    ).view(shift_labels.shape)

    # Apply label weighting
    weights = torch.where(shift_mask, label_weight, 1.0)
    weighted_loss = (loss * weights).sum() / weights.sum()

    return weighted_loss
```

### Dynamic Batching

Variable-length sequences are batched using a **token budget** rather than a fixed batch size:

```python
def create_dynamic_batches(
    sequences: list[list[int]],
    max_tokens_per_batch: int = 32768,
) -> list[list[int]]:
    """Group sequences into batches targeting a token budget.

    This ensures consistent GPU memory usage regardless of
    sequence length distribution.
    """
    # Sort by length for efficient packing
    indexed = sorted(enumerate(sequences), key=lambda x: len(x[1]))

    batches = []
    current_batch = []
    current_tokens = 0

    for idx, seq in indexed:
        seq_len = len(seq)
        if current_tokens + seq_len > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(idx)
        current_tokens += seq_len

    if current_batch:
        batches.append(current_batch)

    return batches
```

### Token Budget by Model Size

From the reference repository, tuned for ~20GB GPU memory with bf16:

| Model Size | Max Tokens/Batch | Workers |
|-----------|-----------------|---------|
| Tiny (3M) | 32,768 | 4 |
| Small (8M) | 32,768 | 4 |
| Medium (48M) | 16,384 | 2 |
| Large (105M) | 8,192 | 2 |

## ECG Route-Based Training

### Tokenize Once, Train Three Times

The efficient approach is to tokenize with `all_branches` and filter at training time:

```python
from training.data.token_filter import ECGTokenFilter


def create_filtered_dataset(
    tokenized_dir: Path,
    route: str,  # "no_ecg", "fusion_class", "all_branches"
) -> Dataset:
    """Load dataset with ECG token filtering based on route."""
    vocab = load_vocab(tokenized_dir / "vocab.json")

    # Create filter
    ecg_filter = ECGTokenFilter(vocab, route)

    # Load sequences and apply filter
    sequences = load_sequences(tokenized_dir / "train" / "sequences.parquet")
    filtered = [ecg_filter.filter(seq) for seq in sequences]

    return MedicalEventDataset(filtered, vocab)
```

### Variant Directories

Alternatively, maintain separate tokenized variants:

```
tokenized/
  base/                    # no_ecg route uses this
    vocab.json
    train/sequences.parquet
    test/sequences.parquet
  plus_prototypes/         # fusion_class and all_branches use this
    vocab.json
    train/sequences.parquet
    test/sequences.parquet
```

## Training Loop

```python
import torch
from torch.cuda.amp import autocast, GradScaler


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
):
    scaler = GradScaler() if not config.bf16 else None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            label_mask = batch.get("label_mask", None)

            with autocast(dtype=torch.bfloat16, enabled=config.bf16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Use custom weighted loss for label token boosting
                loss = weighted_cross_entropy(
                    outputs.logits, input_ids, label_mask,
                    label_weight=config.label_loss_weight,
                    pad_id=config.pad_id,
                )

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        # Validation
        val_loss = evaluate(model, val_loader, config)

        # Early stopping
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, config)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break
```

## Checkpointing and Logging

### Checkpoint Format

Use HuggingFace save_pretrained for vLLM compatibility:

```python
def save_checkpoint(model, optimizer, scheduler, epoch, config):
    checkpoint_dir = config.checkpoint_dir / f"checkpoint_{epoch}"
    model.save_pretrained(checkpoint_dir)  # Saves config.json + model.safetensors

    # Save training state separately
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, checkpoint_dir / "training_state.pt")
```

### Logging

Support both TensorBoard and Weights & Biases:

```python
# W&B logging
import wandb
wandb.init(project="clif-protoecg-fm", name=f"{size}_{route}")
wandb.log({"train/loss": loss, "train/lr": lr}, step=global_step)
```

## Dependencies

- Input: Tokenized sequences from Steps 1-5
- Input: Vocabulary and decile bins
- Output: Trained model checkpoints (HF format)
- Output: Training metrics and logs
- Feeds into: Step 7 (Inference)
