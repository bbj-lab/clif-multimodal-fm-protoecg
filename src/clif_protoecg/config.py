"""Pydantic configuration models for the full pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Paths to CLIF data and ECG prototypes."""

    data_dir: Path = Path("./data")
    ecg_csv: Path = Path("./data/ecg_prototypes_with_shifted_dates.csv")
    output_dir: Path = Path("./data/processed")
    id_mapping_path: Path | None = None


class ExtractionConfig(BaseModel):
    """Step 1: CLIF -> MEDS extraction settings."""

    n_patients: int | None = None
    bin_mode: Literal["hybrid", "clinical_only", "quantile"] = "hybrid"
    n_bins: int = 10
    min_events: int = 1000
    min_admissions: int = 100
    seed: int = 42
    batch_size: int = 1000
    chunk_size: int = 50_000


class SplitConfig(BaseModel):
    """Patient-level split settings."""

    val_fraction: float = 0.1
    seed: int = 42


class LabelConfig(BaseModel):
    """Step 2-3: label settings."""

    categories: list[str] = Field(
        default=["clif_only", "clif_partial"],
        description="Which label categories to compute",
    )


class ECGConfig(BaseModel):
    """Step 4: ECG prototype settings."""

    route: Literal["no_ecg", "fusion_class", "all_branches"] = "all_branches"
    n_similarity_bins: int = 10
    lookback_days: int = 7


class TimeTokenConfig(BaseModel):
    """Step 5: clock + gap token settings."""

    clock_interval_hours: int = 4
    bucket_edges: list[int] = Field(default=[1, 5, 15, 30, 60, 120, 240, 480])


class TokenizationConfig(BaseModel):
    """Step 5-6: tokenization settings."""

    fuse_numeric_bins: bool = False
    min_token_count: int = 1


class TrainingConfig(BaseModel):
    """Step 6: training hyperparameters."""

    model_size: Literal["tiny", "small", "medium", "large"] = "tiny"
    num_epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    label_loss_weight: float = 75.0
    max_tokens_per_batch: int = 32768
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-3
    eval_steps: int = 500
    num_workers: int = 4
    max_seq_len: int | None = None
    chunk_overlap: int = 256
    val_max_batches: int | None = 50
    wandb_project: str | None = "protoecg-fm"
    wandb_run_name: str | None = None


class InferenceConfig(BaseModel):
    """Step 7: inference settings."""

    backend: Literal["native", "vllm", "sglang"] = "vllm"
    n_samples: int = 8
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.9


class EvaluationConfig(BaseModel):
    """Step 8: evaluation settings."""

    horizons: list[int] = Field(default=[8, 24, 48])
    max_days: int = 7
    n_bootstrap: int = 1000
    confidence: float = 0.95
    inference_patients: int | None = None
    samples_per_patient: int | None = None


class PipelineConfig(BaseModel):
    """Root configuration for the full pipeline."""

    data: DataConfig = DataConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    split: SplitConfig = SplitConfig()
    labels: LabelConfig = LabelConfig()
    ecg: ECGConfig = ECGConfig()
    time_tokens: TimeTokenConfig = TimeTokenConfig()
    tokenization: TokenizationConfig = TokenizationConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    # Computed paths
    @property
    def processed_dir(self) -> Path:
        return self.data.output_dir

    @property
    def metadata_dir(self) -> Path:
        return self.processed_dir / "metadata"

    @property
    def tokenized_dir(self) -> Path:
        return self.processed_dir / "tokenized"

    @property
    def checkpoints_dir(self) -> Path:
        return self.processed_dir / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.processed_dir / "results"
