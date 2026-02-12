# Pipeline Configuration

Configuration is specified in YAML. Any omitted key falls back to its default value.

Two configs are provided:

| File | Purpose |
|------|---------|
| `default.yaml` | Full-cohort run with production settings |
| `test.yaml` | Quick smoke-test with a handful of patients |

---

## Sections

### `data` — File paths

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `./data` | Directory containing CLIF v2.1.0 parquet files (`clif_*.parquet`) |
| `ecg_csv` | `./data/ecg_prototypes_with_shifted_dates.csv` | ECG prototype CSV with shifted timestamps |
| `output_dir` | `./data/processed` | Root directory for all pipeline outputs (`sequences.pkl`, tokenized data, checkpoints, results) |
| `id_mapping_path` | *none* | Optional path to a custom MIMIC `hadm_id` -> CLIF `hospitalization_id` mapping file. If omitted, the mapping is derived from the CLIF hospitalization table. |

### `extraction` — Step 1: CLIF -> MEDS sequences

| Key | Default | Description |
|-----|---------|-------------|
| `n_patients` | *none* | Limit extraction to the first N patients (useful for testing). Omit for the full cohort. Also overridable with `--n-patients` on the CLI. |
| `bin_mode` | `hybrid` | How to discretize numeric values. `hybrid` uses clinically-defined bin edges for vitals/key labs (see `constants.CLINICAL_BINS`) and quantile bins for everything else. |
| `n_bins` | `10` | Number of quantile bins for numeric codes that don't have clinical bin edges. |
| `min_events` | `1000` | Minimum total occurrences for a code to be kept during profiling (frequency filter). |
| `min_admissions` | `100` | Minimum distinct admissions a code must appear in to be kept. |
| `seed` | `42` | Random seed for reproducibility. |
| `batch_size` | `1000` | Number of hospitalizations to process per batch during extraction. |

### `split` — Train / Val / Test split

| Key | Default | Description |
|-----|---------|-------------|
| `val_fraction` | `0.1` | Fraction of patients held out for validation. An equal-sized test set is carved out first, then `val_fraction` of the remainder is used for validation. |
| `seed` | `42` | Random seed for the split. |

### `labels` — Steps 2-3: Label definitions

| Key | Default | Description |
|-----|---------|-------------|
| `categories` | `[clif_only, clif_partial]` | Which label categories to evaluate. `clif_only` labels are computable from CLIF tables alone. `clif_partial` labels require extra heuristics or proxy variables. |

### `ecg` — Step 4: ECG prototype injection

| Key | Default | Description |
|-----|---------|-------------|
| `route` | `all_branches` | ECG ablation route. One of: |
| | | `no_ecg` — drop all ECG tokens (clinical-only baseline) |
| | | `fusion_class` — include classification + 1D prototype/similarity |
| | | `all_branches` — include classification + 1D + 2D morphology + 2D global |
| `n_similarity_bins` | `10` | Number of quantile bins for discretizing similarity scores per branch. |
| `lookback_days` | `7` | Include ECGs from up to this many days *before* admission (pre-admission ECGs). ECGs after discharge are always excluded. |

### `time_tokens` — Step 5: Clock & gap tokens

| Key | Default | Description |
|-----|---------|-------------|
| `clock_interval_hours` | `4` | Interval for absolute clock tokens (e.g., `CLOCK//00:00`, `CLOCK//04:00`, ..., `CLOCK//20:00`). |
| `bucket_edges` | `[0, 1, 5, 15, 30, 60, 120, 240, 480]` | Minute boundaries for relative gap tokens between consecutive events. A 7-minute gap becomes `DT//5-15`. |

### `tokenization` — Steps 5-6: Vocabulary & token IDs

| Key | Default | Description |
|-----|---------|-------------|
| `fuse_numeric_bins` | `false` | If `true`, numeric codes and their quantile bin are fused into a single token (e.g., `LAB_RESULT//potassium/Q_3`). If `false`, they are emitted as two separate tokens (`LAB_RESULT//potassium` then `Q_3`). |
| `min_token_count` | `1` | Minimum training-set count for a token to be included in the vocabulary. Tokens below this threshold are mapped to `[UNK]`. |

### `training` — Step 6: Model training

| Key | Default | Description |
|-----|---------|-------------|
| `model_size` | `tiny` | Architecture size preset. One of `tiny`, `small`, `medium`, `large`. Also selectable via `--size` on the CLI. |
| `num_epochs` | `10` | Maximum training epochs. |
| `learning_rate` | `3e-4` | Peak learning rate (after warmup). |
| `weight_decay` | `0.1` | AdamW weight decay. |
| `beta1` | `0.9` | Adam beta1. |
| `beta2` | `0.95` | Adam beta2. |
| `warmup_steps` | `100` | Linear warmup steps before reaching peak learning rate. |
| `max_grad_norm` | `1.0` | Gradient clipping norm. |
| `label_loss_weight` | `75.0` | Loss multiplier applied to label tokens (`LABEL//`, `DISCH//`, etc.) to up-weight prediction of clinical outcomes. |
| `max_tokens_per_batch` | `32768` | Dynamic batching target: pack sequences until this token budget is reached. |
| `gradient_accumulation_steps` | `1` | Number of forward passes before a weight update. Effective batch size = `max_tokens_per_batch * gradient_accumulation_steps`. |
| `bf16` | `true` | Use bfloat16 mixed precision. Set to `false` on hardware without bf16 support (e.g., older GPUs, CPU-only). |
| `early_stopping` | `true` | Stop training when validation loss stops improving. |
| `early_stopping_patience` | `5` | Number of epochs without improvement before stopping. |
| `early_stopping_min_delta` | `1e-4` | Minimum decrease in val loss to count as an improvement. |
| `num_workers` | `4` | DataLoader worker processes. |
| `max_seq_len` | *none* | Optional hard cap on sequence length (tokens). Sequences longer than this are truncated. Omit to keep all tokens. |

### `inference` — Step 7: Autoregressive generation

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `native` | Inference engine. `native` uses PyTorch directly; `vllm` uses vLLM for batched/paged-attention serving. |
| `n_samples` | `8` | Number of autoregressive continuations to sample per hospitalization prefix. |
| `max_new_tokens` | `500` | Maximum tokens to generate per sample. Generation also stops at `[EOS]` or when the horizon is reached. |
| `temperature` | `1.0` | Sampling temperature. Lower = more deterministic. |
| `top_k` | `50` | Top-k filtering (keep only the k highest-probability tokens). |
| `top_p` | `0.95` | Nucleus (top-p) filtering. |
| `gpu_memory_utilization` | `0.9` | Fraction of GPU memory to reserve (vLLM backend only). |

### `evaluation` — Step 8: Window-based AUROCs

| Key | Default | Description |
|-----|---------|-------------|
| `horizons` | `[8, 24, 48]` | Prediction horizons in hours. For each horizon, the model generates continuations up to that many hours and label predictions are evaluated. |
| `max_days` | `7` | Maximum calendar days from admission to evaluate. Hospitalizations shorter than a given horizon are excluded from that horizon's metrics. |
| `n_bootstrap` | `1000` | Number of bootstrap resamples for confidence intervals on AUROC. |
| `confidence` | `0.95` | Confidence level for bootstrap intervals (e.g., 0.95 = 95% CI). |

---

## Derived paths

These are computed from `data.output_dir` and are not set in YAML:

| Path | Location |
|------|----------|
| Sequences | `{output_dir}/sequences.pkl` |
| Metadata | `{output_dir}/metadata/` (splits, code profile, bin edges) |
| Tokenized data | `{output_dir}/tokenized/` (vocab, `all_tokens.parquet`) |
| Checkpoints | `{output_dir}/checkpoints/{route}_{size}/` |
| Results | `{output_dir}/results/` |

## Example: creating a custom config

Start from `default.yaml` and override only what you need:

```yaml
data:
  output_dir: ./data/processed_custom

extraction:
  n_patients: 500
  n_bins: 5

ecg:
  route: fusion_class

training:
  model_size: medium
  num_epochs: 20
  bf16: true
```

Then run:
```bash
protoecg-pipeline extract --config config/custom.yaml
protoecg-pipeline tokenize --config config/custom.yaml
protoecg-pipeline train --config config/custom.yaml
```
