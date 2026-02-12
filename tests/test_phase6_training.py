"""Phase 6: Training tests — model, trainer, dataset, collator, token filter."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from clif_protoecg.training.model import (
    SIZE_PRESETS,
    create_model,
    save_model,
    load_model,
)
from clif_protoecg.training.data.dataset import MedicalEventDataset
from clif_protoecg.training.data.collator import (
    DynamicBatchCollator,
    create_dynamic_batches,
)
from clif_protoecg.training.data.token_filter import ECGTokenFilter
from clif_protoecg.training.trainer import (
    TrainerConfig,
    weighted_cross_entropy,
    train_epoch,
    evaluate,
    create_optimizer_and_scheduler,
)
from clif_protoecg.training.stage import TrainingStage
from clif_protoecg.tokenization.vocabulary import Vocabulary


# ──────────────────────── Fixtures ──────────────────────────


@pytest.fixture
def small_vocab() -> Vocabulary:
    """Vocabulary with 50 tokens including specials, labels, ECG, and clinical."""
    tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    # Clinical tokens
    tokens += [f"VITAL//HR/Q_{i}" for i in range(5)]
    tokens += [f"LAB//glucose/Q_{i}" for i in range(5)]
    tokens += [f"MED_BOLUS//aspirin" for _ in range(1)]
    # ECG tokens
    tokens += [
        "ECG//Class/rhythm_normal",
        "ECG//Class/rhythm_afib",
        "ECG//Prototype/1D/Q_0",
        "ECG//Prototype/1D/Q_1",
        "ECG//Similarity/1D/Q_0",
        "ECG//Similarity/2D_morph/Q_0",
        "ECG//Prototype/2D_morph/Q_0",
        "ECG//Similarity/2D_global/Q_0",
        "ECG//Prototype/2D_global/Q_0",
    ]
    # Time tokens
    tokens += ["CLOCK//00:00", "CLOCK//04:00", "DT//0", "DT//1-5"]
    # Label tokens
    tokens += ["LABEL//mortality", "LABEL//icu_admission", "LABEL//mech_vent"]
    # Pad to exactly 50
    while len(tokens) < 50:
        tokens.append(f"TOKEN_{len(tokens)}")

    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


@pytest.fixture
def sample_sequences() -> list[list[int]]:
    """Several token-id sequences of varying lengths."""
    return [
        [2, 5, 6, 7, 8, 9, 3],           # len 7
        [2, 10, 11, 12, 3],               # len 5
        [2, 15, 16, 17, 18, 19, 20, 3],   # len 8
        [2, 5, 10, 15, 20, 25, 30, 35, 3], # len 9
    ]


@pytest.fixture
def label_token_ids() -> set[int]:
    """Token IDs corresponding to label tokens (indices 30, 31, 32 in small_vocab)."""
    return {30, 31, 32}


@pytest.fixture
def trainer_config() -> TrainerConfig:
    return TrainerConfig(
        num_epochs=2,
        learning_rate=1e-3,
        warmup_steps=2,
        bf16=False,
        device="cpu",
        gradient_accumulation_steps=1,
        early_stopping=False,
    )


# ──────────────────────── Model tests ──────────────────────────


class TestCreateModel:
    def test_creates_tiny_model(self):
        model = create_model(vocab_size=100, size_name="tiny")
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 4
        assert model.config.num_attention_heads == 4
        assert model.config.vocab_size == 100

    def test_creates_small_model(self):
        model = create_model(vocab_size=200, size_name="small")
        assert model.config.hidden_size == 256
        assert model.config.num_hidden_layers == 8
        assert model.config.num_attention_heads == 8

    def test_creates_medium_model(self):
        model = create_model(vocab_size=200, size_name="medium")
        assert model.config.hidden_size == 512
        assert model.config.num_hidden_layers == 12

    def test_creates_large_model(self):
        model = create_model(vocab_size=200, size_name="large")
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12

    def test_invalid_size_raises(self):
        with pytest.raises(KeyError):
            create_model(vocab_size=100, size_name="xlarge")

    def test_custom_max_position_embeddings(self):
        model = create_model(vocab_size=100, size_name="tiny", max_position_embeddings=4096)
        assert model.config.max_position_embeddings == 4096

    def test_default_max_position(self):
        model = create_model(vocab_size=100, size_name="tiny")
        assert model.config.max_position_embeddings == SIZE_PRESETS["tiny"]["max_pos"]

    def test_tied_embeddings(self):
        model = create_model(vocab_size=100, size_name="tiny")
        assert model.config.tie_word_embeddings is True

    def test_model_forward_pass(self):
        model = create_model(vocab_size=100, size_name="tiny")
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        out = model(input_ids=ids)
        assert out.logits.shape == (1, 5, 100)

    def test_all_presets_exist(self):
        assert set(SIZE_PRESETS.keys()) == {"tiny", "small", "medium", "large"}

    def test_model_param_count_tiny(self):
        model = create_model(vocab_size=100, size_name="tiny")
        n_params = sum(p.numel() for p in model.parameters())
        # Tiny should be in the range of ~1-5M params with small vocab
        assert 100_000 < n_params < 10_000_000


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        model = create_model(vocab_size=100, size_name="tiny")
        save_path = tmp_path / "test_model"
        save_model(model, save_path)

        loaded = load_model(save_path)
        assert loaded.config.hidden_size == model.config.hidden_size
        assert loaded.config.vocab_size == model.config.vocab_size
        assert loaded.config.num_hidden_layers == model.config.num_hidden_layers

    def test_save_creates_directory(self, tmp_path: Path):
        model = create_model(vocab_size=100, size_name="tiny")
        save_path = tmp_path / "nested" / "dir" / "model"
        save_model(model, save_path)
        assert save_path.exists()

    def test_loaded_model_produces_same_output(self, tmp_path: Path):
        model = create_model(vocab_size=100, size_name="tiny")
        model.eval()
        ids = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            out1 = model(input_ids=ids).logits

        save_model(model, tmp_path / "m")
        loaded = load_model(tmp_path / "m")
        loaded.eval()

        with torch.no_grad():
            out2 = loaded(input_ids=ids).logits

        assert torch.allclose(out1, out2, atol=1e-5)


# ──────────────────────── Dataset tests ──────────────────────────


class TestMedicalEventDataset:
    def test_length(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        assert len(ds) == 4

    def test_getitem_returns_dict(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        item = ds[0]
        assert "input_ids" in item
        assert "label_mask" in item

    def test_input_ids_dtype(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long

    def test_label_mask_dtype(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        item = ds[0]
        assert item["label_mask"].dtype == torch.bool

    def test_label_mask_all_false_when_no_label_ids(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        item = ds[0]
        assert not item["label_mask"].any()

    def test_label_mask_marks_correct_positions(self, label_token_ids):
        # Sequence containing label tokens at positions 2, 4
        seq = [2, 5, 30, 10, 31, 3]
        ds = MedicalEventDataset([seq], label_token_ids=label_token_ids)
        item = ds[0]
        expected = torch.tensor([False, False, True, False, True, False])
        assert torch.equal(item["label_mask"], expected)

    def test_max_seq_len_truncates(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences, max_seq_len=4)
        item = ds[2]  # Original len 8
        assert item["input_ids"].size(0) == 4

    def test_max_seq_len_no_truncation_when_shorter(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences, max_seq_len=100)
        item = ds[1]  # len 5
        assert item["input_ids"].size(0) == 5

    def test_empty_sequences(self):
        ds = MedicalEventDataset([])
        assert len(ds) == 0

    def test_from_parquet(self, tmp_path: Path):
        import polars as pl

        seqs = [[1, 2, 3], [4, 5, 6, 7]]
        df = pl.DataFrame({"token_ids": seqs})
        path = tmp_path / "tokens.parquet"
        df.write_parquet(path)

        ds = MedicalEventDataset.from_parquet(path)
        assert len(ds) == 2
        assert ds[0]["input_ids"].tolist() == [1, 2, 3]
        assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]


# ──────────────────────── Collator tests ──────────────────────────


class TestDynamicBatchCollator:
    def test_pads_to_max_length(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        collator = DynamicBatchCollator(pad_id=0)
        batch = [ds[0], ds[1]]  # len 7, len 5
        out = collator(batch)
        assert out["input_ids"].shape == (2, 7)

    def test_pad_values(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        collator = DynamicBatchCollator(pad_id=0)
        batch = [ds[0], ds[1]]  # len 7, len 5
        out = collator(batch)
        # Second sequence should be padded with 0 at positions 5, 6
        assert out["input_ids"][1, 5].item() == 0
        assert out["input_ids"][1, 6].item() == 0

    def test_attention_mask_correct(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        collator = DynamicBatchCollator(pad_id=0)
        batch = [ds[0], ds[1]]  # len 7, len 5
        out = collator(batch)
        # First row: all 1s
        assert out["attention_mask"][0].sum().item() == 7
        # Second row: 5 ones, 2 zeros
        assert out["attention_mask"][1].sum().item() == 5

    def test_label_mask_padded(self, label_token_ids):
        seq1 = [2, 30, 5, 3]   # label at pos 1
        seq2 = [2, 5, 3]       # no labels
        ds = MedicalEventDataset([seq1, seq2], label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=0)
        out = collator([ds[0], ds[1]])
        # seq1 label_mask: [F, T, F, F], seq2: [F, F, F, F] (padded)
        assert out["label_mask"][0, 1].item() is True
        assert out["label_mask"][1].sum().item() == 0

    def test_single_item_batch(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        collator = DynamicBatchCollator(pad_id=0)
        out = collator([ds[0]])
        assert out["input_ids"].shape == (1, 7)

    def test_output_dtypes(self, sample_sequences):
        ds = MedicalEventDataset(sample_sequences)
        collator = DynamicBatchCollator(pad_id=0)
        out = collator([ds[0], ds[1]])
        assert out["input_ids"].dtype == torch.long
        assert out["attention_mask"].dtype == torch.long
        assert out["label_mask"].dtype == torch.bool


class TestCreateDynamicBatches:
    def test_single_batch_within_budget(self):
        lengths = [3, 3]  # total 6 tokens
        batches = create_dynamic_batches(lengths, max_tokens_per_batch=10)
        assert len(batches) == 1
        assert set(batches[0]) == {0, 1}

    def test_splits_when_exceeding_budget(self):
        lengths = [5, 5, 5]  # 5 each
        batches = create_dynamic_batches(lengths, max_tokens_per_batch=8)
        assert len(batches) >= 2
        # All indices present exactly once
        all_idx = [i for b in batches for i in b]
        assert sorted(all_idx) == [0, 1, 2]

    def test_sorted_by_length(self):
        lengths = [10, 3, 5]
        batches = create_dynamic_batches(lengths, max_tokens_per_batch=10)
        # After sorting by length: index 1 (3), index 2 (5), index 0 (10)
        # Batch 1: [1, 2] (3+5=8), Batch 2: [0] (10)
        all_idx = [i for b in batches for i in b]
        assert sorted(all_idx) == [0, 1, 2]

    def test_empty_sequences(self):
        batches = create_dynamic_batches([], max_tokens_per_batch=100)
        assert batches == []

    def test_large_budget_single_batch(self, sample_sequences):
        lengths = [len(s) for s in sample_sequences]
        batches = create_dynamic_batches(lengths, max_tokens_per_batch=100000)
        assert len(batches) == 1


# ──────────────────────── Token Filter tests ──────────────────────────


class TestECGTokenFilter:
    @pytest.fixture
    def ecg_vocab(self) -> dict[str, int]:
        return {
            "[PAD]": 0,
            "VITAL//HR": 1,
            "ECG//Class/rhythm_normal": 2,
            "ECG//Prototype/1D/Q_0": 3,
            "ECG//Similarity/1D/Q_0": 4,
            "ECG//Prototype/2D_morph/Q_0": 5,
            "ECG//Similarity/2D_morph/Q_0": 6,
            "ECG//Prototype/2D_global/Q_0": 7,
            "ECG//Similarity/2D_global/Q_0": 8,
            "LAB//glucose": 9,
            "CLOCK//00:00": 10,
        }

    def test_all_branches_keeps_everything(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "all_branches")
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert filt.filter(ids) == ids

    def test_no_ecg_removes_all_ecg(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "no_ecg")
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = filt.filter(ids)
        # Should keep non-ECG: 0, 1, 9, 10
        assert result == [0, 1, 9, 10]

    def test_fusion_class_keeps_class_and_1d(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "fusion_class")
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = filt.filter(ids)
        # Should keep: non-ECG + ECG//Class + ECG//Prototype/1D + ECG//Similarity/1D
        # Keep: 0, 1, 2, 3, 4, 9, 10
        # Remove: 5 (2D_morph proto), 6 (2D_morph sim), 7 (2D_global proto), 8 (2D_global sim)
        assert result == [0, 1, 2, 3, 4, 9, 10]

    def test_no_ecg_empty_sequence(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "no_ecg")
        assert filt.filter([]) == []

    def test_no_ecg_no_ecg_in_sequence(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "no_ecg")
        assert filt.filter([0, 1, 9, 10]) == [0, 1, 9, 10]

    def test_filtered_ids_frozenset(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "no_ecg")
        assert isinstance(filt.filtered_ids, frozenset)

    def test_all_branches_filtered_ids_empty(self, ecg_vocab):
        filt = ECGTokenFilter(ecg_vocab, "all_branches")
        assert len(filt.filtered_ids) == 0


# ──────────────────────── Trainer tests ──────────────────────────


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.num_epochs == 1
        assert cfg.label_loss_weight == 75.0
        assert cfg.early_stopping is True

    def test_custom_values(self):
        cfg = TrainerConfig(learning_rate=1e-4, num_epochs=5)
        assert cfg.learning_rate == 1e-4
        assert cfg.num_epochs == 5


class TestWeightedCrossEntropy:
    def test_basic_loss_nonzero(self):
        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(1, vocab_size, (batch_size, seq_len))  # avoid pad=0
        label_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        loss = weighted_cross_entropy(logits, labels, label_mask, label_weight=75.0, pad_id=0)
        assert loss.item() > 0

    def test_loss_is_scalar(self):
        logits = torch.randn(2, 5, 20)
        labels = torch.randint(1, 20, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)
        loss = weighted_cross_entropy(logits, labels, mask)
        assert loss.dim() == 0

    def test_label_weight_increases_loss(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 10, 50)
        labels = torch.randint(1, 50, (2, 10))
        # Mask with some label positions
        mask_with_labels = torch.zeros(2, 10, dtype=torch.bool)
        mask_with_labels[0, 3] = True
        mask_with_labels[1, 5] = True

        loss_w1 = weighted_cross_entropy(logits, labels, mask_with_labels, label_weight=1.0)
        loss_w75 = weighted_cross_entropy(logits, labels, mask_with_labels, label_weight=75.0)
        # Higher weight on label positions should change the loss
        assert not torch.isclose(loss_w1, loss_w75)

    def test_pad_id_ignored(self):
        logits = torch.randn(1, 5, 20)
        labels = torch.tensor([[1, 2, 0, 0, 0]])  # positions 2-4 are pad
        mask = torch.zeros(1, 5, dtype=torch.bool)
        loss = weighted_cross_entropy(logits, labels, mask, pad_id=0)
        assert loss.item() >= 0  # Should not NaN

    def test_all_pad_returns_zero(self):
        logits = torch.randn(1, 5, 20)
        labels = torch.zeros(1, 5, dtype=torch.long)  # all pad
        mask = torch.zeros(1, 5, dtype=torch.bool)
        loss = weighted_cross_entropy(logits, labels, mask, pad_id=0)
        assert loss.item() == 0.0


class TestCreateOptimizerAndScheduler:
    def test_creates_optimizer_and_scheduler(self, trainer_config):
        model = create_model(vocab_size=50, size_name="tiny")
        optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config, total_steps=100)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert scheduler is not None

    def test_scheduler_steps_without_error(self, trainer_config):
        model = create_model(vocab_size=50, size_name="tiny")
        optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config, total_steps=10)
        for _ in range(10):
            optimizer.step()
            scheduler.step()

    def test_lr_increases_during_warmup(self, trainer_config):
        trainer_config.warmup_steps = 5
        model = create_model(vocab_size=50, size_name="tiny")
        optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config, total_steps=20)
        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        # LR should generally increase during warmup
        assert lrs[-1] > lrs[0]


class TestTrainEpoch:
    def test_returns_float_loss(self, small_vocab, trainer_config, label_token_ids):
        model = create_model(vocab_size=small_vocab.size, size_name="tiny")
        seqs = [[2, 5, 6, 7, 8, 9, 3], [2, 10, 11, 12, 3]]
        ds = MedicalEventDataset(seqs, label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=small_vocab.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collator)

        total_steps = len(loader) * trainer_config.num_epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config, total_steps)

        loss = train_epoch(model, loader, optimizer, scheduler, trainer_config)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_epochs(self, small_vocab, trainer_config, label_token_ids):
        """Training on same data multiple times should reduce loss."""
        model = create_model(vocab_size=small_vocab.size, size_name="tiny")
        seqs = [[2, 5, 6, 7, 8, 9, 3]] * 10  # Repeat for enough data
        ds = MedicalEventDataset(seqs, label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=small_vocab.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collator)

        trainer_config.num_epochs = 10
        total_steps = len(loader) * trainer_config.num_epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config, total_steps)

        losses = []
        for _ in range(5):
            loss = train_epoch(model, loader, optimizer, scheduler, trainer_config)
            losses.append(loss)

        # Loss should generally decrease (first > last)
        assert losses[-1] < losses[0]


class TestEvaluate:
    def test_returns_float_loss(self, small_vocab, trainer_config, label_token_ids):
        model = create_model(vocab_size=small_vocab.size, size_name="tiny")
        seqs = [[2, 5, 6, 7, 8, 9, 3], [2, 10, 11, 12, 3]]
        ds = MedicalEventDataset(seqs, label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=small_vocab.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collator)

        loss = evaluate(model, loader, trainer_config)
        assert isinstance(loss, float)
        assert loss > 0

    def test_eval_loss_deterministic(self, small_vocab, trainer_config, label_token_ids):
        model = create_model(vocab_size=small_vocab.size, size_name="tiny")
        model.eval()
        seqs = [[2, 5, 6, 7, 8, 9, 3]]
        ds = MedicalEventDataset(seqs, label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=small_vocab.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collator)

        loss1 = evaluate(model, loader, trainer_config)
        loss2 = evaluate(model, loader, trainer_config)
        assert abs(loss1 - loss2) < 1e-5


class TestGradientAccumulation:
    def test_gradient_accumulation_steps(self, small_vocab, label_token_ids):
        config = TrainerConfig(
            num_epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            bf16=False,
            device="cpu",
            gradient_accumulation_steps=2,
            early_stopping=False,
        )
        model = create_model(vocab_size=small_vocab.size, size_name="tiny")
        seqs = [[2, 5, 6, 7, 3]] * 4
        ds = MedicalEventDataset(seqs, label_token_ids=label_token_ids)
        collator = DynamicBatchCollator(pad_id=small_vocab.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collator)

        total_steps = len(loader) * config.num_epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
        loss = train_epoch(model, loader, optimizer, scheduler, config)
        assert isinstance(loss, float)
        assert loss > 0


# ──────────────────────── TrainingStage tests ──────────────────────────


class TestTrainingStage:
    def test_stage_name(self):
        stage = TrainingStage()
        assert stage.name == "training"

    def test_runs_training(self, tmp_path: Path, small_vocab, label_token_ids):
        stage = TrainingStage()
        vocab_dir = tmp_path / "vocab"
        small_vocab.save(vocab_dir)

        config = TrainerConfig(
            num_epochs=2,
            learning_rate=1e-3,
            warmup_steps=1,
            bf16=False,
            device="cpu",
            early_stopping=False,
        )
        train_seqs = [[2, 5, 6, 7, 8, 3]] * 4
        val_seqs = [[2, 10, 11, 12, 3]] * 2

        result = stage.run(
            input_path=vocab_dir,
            output_path=tmp_path / "output",
            vocab=small_vocab,
            train_sequences=train_seqs,
            val_sequences=val_seqs,
            config=config,
            size_name="tiny",
            label_token_ids=label_token_ids,
        )
        assert result.success is True
        assert result.output_path == tmp_path / "output"
        assert "train_loss" in result.metrics
        assert (tmp_path / "output" / "final").exists()

    def test_early_stopping(self, tmp_path: Path, small_vocab, label_token_ids):
        stage = TrainingStage()
        config = TrainerConfig(
            num_epochs=100,  # Very high, should stop early
            learning_rate=1e-3,
            warmup_steps=1,
            bf16=False,
            device="cpu",
            early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=1e-10,  # Very strict
        )
        train_seqs = [[2, 5, 6, 7, 8, 3]] * 4
        val_seqs = [[2, 10, 11, 12, 3]] * 2

        result = stage.run(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            vocab=small_vocab,
            train_sequences=train_seqs,
            val_sequences=val_seqs,
            config=config,
            size_name="tiny",
            label_token_ids=label_token_ids,
        )
        assert result.success is True
        # Should have stopped well before 100 epochs
        assert result.duration_seconds < 300  # Sanity check

    def test_empty_val_set(self, tmp_path: Path, small_vocab, label_token_ids):
        stage = TrainingStage()
        config = TrainerConfig(
            num_epochs=1,
            learning_rate=1e-3,
            warmup_steps=1,
            bf16=False,
            device="cpu",
            early_stopping=False,
        )
        train_seqs = [[2, 5, 6, 7, 8, 3]] * 4

        result = stage.run(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            vocab=small_vocab,
            train_sequences=train_seqs,
            val_sequences=[],
            config=config,
            size_name="tiny",
            label_token_ids=label_token_ids,
        )
        assert result.success is True
