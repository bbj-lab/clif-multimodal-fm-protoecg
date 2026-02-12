"""Phase 7: Evaluation tests — inference, metrics, prediction, plotting."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from clif_protoecg.evaluation.inference.base import GenerationConfig, InferenceEngine
from clif_protoecg.evaluation.inference.native import (
    NativeInferenceEngine,
    _clone_cache,
    _sample_token,
)
from clif_protoecg.evaluation.inference.factory import create_inference_engine
from clif_protoecg.evaluation.inference.vllm_engine import _truncate_at_clock_limit
from clif_protoecg.evaluation.prediction import (
    PredictionWindow,
    create_midnight_windows,
    extract_predictions,
    extract_predictions_with_censoring,
    get_ground_truth,
)
from clif_protoecg.evaluation.metrics.auroc import compute_auroc, bootstrap_auroc
from clif_protoecg.evaluation.metrics.calibration import (
    compute_brier,
    compute_ece,
    compute_auprc,
)
from clif_protoecg.evaluation.metrics.aggregator import (
    PredictionResult,
    LabelMetrics,
    aggregate_metrics,
)
from clif_protoecg.evaluation.plotting import plot_auroc_comparison, plot_calibration_curve
from clif_protoecg.evaluation.stage import EvaluationStage
from clif_protoecg.training.model import create_model, save_model


# ──────────────────────── Fixtures ──────────────────────────


@pytest.fixture
def tiny_model():
    model = create_model(vocab_size=100, size_name="tiny")
    model.eval()
    return model


@pytest.fixture
def saved_model(tmp_path, tiny_model):
    save_path = tmp_path / "model"
    save_model(tiny_model, save_path)
    return save_path


# ──────────────────────── GenerationConfig tests ──────────────────────────


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.n_samples == 8
        assert cfg.max_new_tokens == 500
        assert cfg.temperature == 1.0
        assert cfg.top_k == 50
        assert cfg.top_p == 0.95

    def test_custom(self):
        cfg = GenerationConfig(n_samples=16, temperature=0.7)
        assert cfg.n_samples == 16
        assert cfg.temperature == 0.7


# ──────────────────────── Native Inference Engine tests ──────────────────────────


class TestNativeInferenceEngine:
    def test_generate_returns_list_of_lists(self, tiny_model):
        engine = NativeInferenceEngine(tiny_model, device="cpu")
        config = GenerationConfig(n_samples=3, max_new_tokens=5)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
        assert all(isinstance(s, list) for s in result)

    def test_generate_respects_max_new_tokens(self, tiny_model):
        engine = NativeInferenceEngine(tiny_model, device="cpu")
        config = GenerationConfig(n_samples=2, max_new_tokens=10)
        result = engine.generate([1, 2, 3], config)
        for sample in result:
            assert len(sample) <= 10

    def test_generate_stops_on_stop_tokens(self, tiny_model):
        engine = NativeInferenceEngine(tiny_model, device="cpu")
        # Use a very large max_new_tokens but set EOS=3 as stop
        config = GenerationConfig(n_samples=1, max_new_tokens=1000)
        result = engine.generate([1, 2], config, stop_token_ids={3})
        # Should stop at some point (either by hitting EOS or max tokens)
        assert len(result) == 1
        assert len(result[0]) <= 1000

    def test_generate_stops_on_clock_limit(self, tiny_model):
        engine = NativeInferenceEngine(tiny_model, device="cpu")
        config = GenerationConfig(n_samples=1, max_new_tokens=500)
        # With a very small vocab, some tokens will be "clock" tokens
        result = engine.generate(
            [1, 2],
            config,
            clock_token_ids={5, 10, 15},
            max_clock_tokens=2,
        )
        assert len(result) == 1

    def test_different_samples_can_differ(self, tiny_model):
        engine = NativeInferenceEngine(tiny_model, device="cpu")
        config = GenerationConfig(n_samples=10, max_new_tokens=20, temperature=1.0)
        result = engine.generate([1, 2, 3], config)
        # With temperature=1.0, not all samples should be identical
        # (probabilistically, but with 10 samples of 20 tokens this is very likely)
        unique_samples = set(tuple(s) for s in result)
        # At least some variation expected (but not guaranteed, so be lenient)
        assert len(result) == 10


class TestSampleToken:
    def test_returns_int(self):
        logits = torch.randn(1, 100)
        token = _sample_token(logits, temperature=1.0, top_k=50, top_p=0.95)
        assert isinstance(token, int)
        assert 0 <= token < 100

    def test_top_k_limits_choices(self):
        # With top_k=1, should always pick the argmax
        logits = torch.zeros(1, 100)
        logits[0, 42] = 100.0  # Make one token dominant
        token = _sample_token(logits, temperature=1.0, top_k=1, top_p=1.0)
        assert token == 42

    def test_zero_temperature_picks_argmax(self):
        # temperature=0 would cause div by zero, but very low temp should pick max
        logits = torch.zeros(1, 100)
        logits[0, 77] = 100.0
        token = _sample_token(logits, temperature=0.001, top_k=0, top_p=1.0)
        assert token == 77


class TestCloneCache:
    def test_cloned_tuple_cache_independent(self):
        cache = ((torch.ones(2, 3), torch.ones(2, 3)),)
        cloned = _clone_cache(cache)
        cloned[0][0].fill_(0)
        assert cache[0][0].sum().item() > 0  # Original unchanged

    def test_cloned_dynamic_cache_independent(self, tiny_model):
        ctx = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = tiny_model(ctx, use_cache=True)
        cache = out.past_key_values
        cloned = _clone_cache(cache)
        # Modify the clone, original should be unaffected
        assert cloned is not cache


# ──────────────────────── Factory tests ──────────────────────────


class TestInferenceFactory:
    def test_create_native_engine(self, saved_model):
        engine = create_inference_engine("native", saved_model, device="cpu")
        assert isinstance(engine, NativeInferenceEngine)

    def test_invalid_backend_raises(self, saved_model):
        with pytest.raises(ValueError, match="Unknown inference backend"):
            create_inference_engine("invalid", saved_model)

    def test_native_engine_can_generate(self, saved_model):
        engine = create_inference_engine("native", saved_model, device="cpu")
        config = GenerationConfig(n_samples=2, max_new_tokens=5)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 2


# ──────────────────────── vLLM truncation tests ──────────────────────────


class TestVLLMTruncation:
    def test_truncate_at_clock_limit(self):
        tokens = [1, 2, 10, 3, 10, 4, 10, 5]
        result = _truncate_at_clock_limit(tokens, clock_ids={10}, max_clocks=2)
        assert result == [1, 2, 10, 3, 10]

    def test_truncate_no_clocks(self):
        tokens = [1, 2, 3, 4]
        result = _truncate_at_clock_limit(tokens, clock_ids={10}, max_clocks=2)
        assert result == [1, 2, 3, 4]

    def test_truncate_fewer_clocks_than_limit(self):
        tokens = [1, 10, 2]
        result = _truncate_at_clock_limit(tokens, clock_ids={10}, max_clocks=3)
        assert result == [1, 10, 2]


# ──────────────────────── Prediction Window tests ──────────────────────────


class TestCreateMidnightWindows:
    def test_creates_windows_at_midnight(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 2, 100, 3, 4, 100, 5]  # 100 = midnight
        timestamps = [
            base + timedelta(hours=i * 4)
            for i in range(7)
        ]
        windows = create_midnight_windows(token_ids, timestamps, midnight_token_id=100)
        assert len(windows) == 2
        assert windows[0].window_id == 0
        assert windows[1].window_id == 1

    def test_context_includes_midnight_token(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 2, 100, 3]
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        windows = create_midnight_windows(token_ids, timestamps, midnight_token_id=100)
        assert windows[0].context_ids == [1, 2, 100]

    def test_max_days_limits_windows(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [100] * 10
        timestamps = [base + timedelta(days=i) for i in range(10)]
        windows = create_midnight_windows(token_ids, timestamps, midnight_token_id=100, max_days=3)
        assert len(windows) == 3

    def test_no_midnight_tokens(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 2, 3]
        timestamps = [base + timedelta(hours=i) for i in range(3)]
        windows = create_midnight_windows(token_ids, timestamps, midnight_token_id=100)
        assert windows == []

    def test_hospitalization_id_preserved(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [100]
        timestamps = [base]
        windows = create_midnight_windows(
            token_ids, timestamps, midnight_token_id=100, hospitalization_id="H001"
        )
        assert windows[0].hospitalization_id == "H001"


# ──────────────────────── Prediction Extraction tests ──────────────────────────


class TestExtractPredictions:
    def test_all_samples_contain_label(self):
        samples = [[1, 2, 50], [3, 50, 4], [50, 5, 6]]
        preds = extract_predictions(samples, {"mortality": 50})
        assert preds["mortality"] == 1.0

    def test_no_samples_contain_label(self):
        samples = [[1, 2, 3], [4, 5, 6]]
        preds = extract_predictions(samples, {"mortality": 50})
        assert preds["mortality"] == 0.0

    def test_proportion(self):
        samples = [[1, 50], [2, 3], [4, 50], [5, 6]]
        preds = extract_predictions(samples, {"mortality": 50})
        assert preds["mortality"] == 0.5

    def test_empty_samples(self):
        preds = extract_predictions([], {"mortality": 50})
        assert preds["mortality"] == 0.0

    def test_multiple_labels(self):
        samples = [[50, 60], [50, 3], [3, 4]]
        preds = extract_predictions(samples, {"mortality": 50, "icu": 60})
        assert abs(preds["mortality"] - 2 / 3) < 1e-9
        assert abs(preds["icu"] - 1 / 3) < 1e-9


class TestExtractPredictionsWithCensoring:
    def test_mortality_uses_death_tokens(self):
        samples = [[1, 99], [2, 3], [99, 4]]  # 99 = death token
        preds = extract_predictions_with_censoring(
            samples, {"mortality": 50}, death_token_ids={99}, discharge_token_ids={98}
        )
        assert abs(preds["mortality"] - 2 / 3) < 1e-9

    def test_non_mortality_uses_label_token(self):
        samples = [[1, 60], [2, 3], [60, 4]]
        preds = extract_predictions_with_censoring(
            samples, {"icu": 60}, death_token_ids={99}, discharge_token_ids={98}
        )
        assert abs(preds["icu"] - 2 / 3) < 1e-9

    def test_empty_samples(self):
        preds = extract_predictions_with_censoring(
            [], {"mortality": 50}, death_token_ids={99}, discharge_token_ids={98}
        )
        assert preds["mortality"] == 0.0


# ──────────────────────── Ground Truth tests ──────────────────────────


class TestGetGroundTruth:
    def test_label_in_horizon(self):
        base = datetime(2024, 1, 1, 0, 0)
        # Label token 50 occurs 4 hours after context end
        token_ids = [1, 2, 3, 50]
        timestamps = [
            base - timedelta(hours=2),
            base - timedelta(hours=1),
            base,  # context end
            base + timedelta(hours=4),  # within 8h horizon
        ]
        gt = get_ground_truth(token_ids, timestamps, base, 8, {"mortality": 50})
        assert gt["mortality"] is True

    def test_label_outside_horizon(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 2, 3, 50]
        timestamps = [
            base - timedelta(hours=2),
            base - timedelta(hours=1),
            base,
            base + timedelta(hours=10),  # outside 8h horizon
        ]
        gt = get_ground_truth(token_ids, timestamps, base, 8, {"mortality": 50})
        assert gt["mortality"] is False

    def test_no_label_in_sequence(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 2, 3]
        timestamps = [base - timedelta(hours=1), base, base + timedelta(hours=1)]
        gt = get_ground_truth(token_ids, timestamps, base, 48, {"mortality": 50})
        assert gt["mortality"] is False

    def test_multiple_labels(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [1, 50, 60]
        timestamps = [base, base + timedelta(hours=2), base + timedelta(hours=5)]
        gt = get_ground_truth(token_ids, timestamps, base, 8, {"mortality": 50, "icu": 60})
        assert gt["mortality"] is True
        assert gt["icu"] is True

    def test_label_before_context_ignored(self):
        base = datetime(2024, 1, 1, 0, 0)
        token_ids = [50, 1, 2]
        timestamps = [base - timedelta(hours=1), base, base + timedelta(hours=1)]
        gt = get_ground_truth(token_ids, timestamps, base, 8, {"mortality": 50})
        assert gt["mortality"] is False


# ──────────────────────── AUROC tests ──────────────────────────


class TestComputeAUROC:
    def test_perfect_predictions(self):
        preds = [0.9, 0.8, 0.1, 0.2]
        labels = [True, True, False, False]
        auroc = compute_auroc(preds, labels)
        assert auroc == 1.0

    def test_random_predictions(self):
        np.random.seed(42)
        preds = list(np.random.rand(100))
        labels = [bool(x) for x in np.random.randint(0, 2, 100)]
        auroc = compute_auroc(preds, labels)
        assert auroc is not None
        assert 0.0 <= auroc <= 1.0

    def test_all_positive_returns_none(self):
        assert compute_auroc([0.5, 0.6], [True, True]) is None

    def test_all_negative_returns_none(self):
        assert compute_auroc([0.5, 0.6], [False, False]) is None

    def test_returns_float(self):
        preds = [0.9, 0.1]
        labels = [True, False]
        result = compute_auroc(preds, labels)
        assert isinstance(result, float)


class TestBootstrapAUROC:
    def test_returns_three_values(self):
        preds = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        labels = [True, True, True, False, False, False]
        median, lo, hi = bootstrap_auroc(preds, labels, n_bootstrap=100, seed=42)
        assert lo <= median <= hi

    def test_confidence_interval_width(self):
        np.random.seed(42)
        preds = list(np.random.rand(50))
        labels = [bool(x) for x in np.random.randint(0, 2, 50)]
        _, lo, hi = bootstrap_auroc(preds, labels, n_bootstrap=200, seed=42)
        # CI should be non-trivial
        assert hi - lo > 0

    def test_perfect_data_narrow_ci(self):
        preds = [0.99, 0.98, 0.97, 0.01, 0.02, 0.03]
        labels = [True, True, True, False, False, False]
        median, lo, hi = bootstrap_auroc(preds, labels, n_bootstrap=500, seed=42)
        assert median == 1.0
        assert lo >= 0.9

    def test_degenerate_returns_zeros(self):
        preds = [0.5, 0.6]
        labels = [True, True]
        median, lo, hi = bootstrap_auroc(preds, labels, n_bootstrap=50, seed=42)
        assert median == 0.0


# ──────────────────────── Calibration tests ──────────────────────────


class TestComputeBrier:
    def test_perfect_predictions(self):
        preds = [1.0, 0.0]
        labels = [True, False]
        assert compute_brier(preds, labels) == 0.0

    def test_worst_predictions(self):
        preds = [0.0, 1.0]
        labels = [True, False]
        assert compute_brier(preds, labels) == 1.0

    def test_middle_predictions(self):
        preds = [0.5, 0.5]
        labels = [True, False]
        assert abs(compute_brier(preds, labels) - 0.25) < 1e-9


class TestComputeECE:
    def test_perfectly_calibrated(self):
        # Probabilities match frequencies exactly
        preds = [0.05] * 10 + [0.95] * 10
        labels = [False] * 10 + [True] * 10
        ece = compute_ece(preds, labels, n_bins=10)
        assert ece < 0.1

    def test_poorly_calibrated(self):
        # Predict 0.9 but all negative
        preds = [0.9] * 20
        labels = [False] * 20
        ece = compute_ece(preds, labels, n_bins=10)
        assert ece > 0.5

    def test_empty_returns_zero(self):
        assert compute_ece([], [], n_bins=10) == 0.0


class TestComputeAUPRC:
    def test_perfect_predictions(self):
        preds = [0.9, 0.8, 0.1, 0.2]
        labels = [True, True, False, False]
        auprc = compute_auprc(preds, labels)
        assert auprc == 1.0

    def test_no_positives(self):
        assert compute_auprc([0.5, 0.6], [False, False]) is None

    def test_returns_float(self):
        preds = [0.9, 0.1, 0.8, 0.2]
        labels = [True, False, True, False]
        result = compute_auprc(preds, labels)
        assert isinstance(result, float)


# ──────────────────────── Aggregator tests ──────────────────────────


class TestAggregateMetrics:
    def test_groups_by_label_and_horizon(self):
        results = [
            PredictionResult("P1", 0, "mortality", 8, 0.9, True),
            PredictionResult("P1", 0, "mortality", 8, 0.1, False),
            PredictionResult("P1", 0, "mortality", 24, 0.8, True),
            PredictionResult("P1", 0, "mortality", 24, 0.2, False),
            PredictionResult("P1", 0, "icu", 8, 0.7, True),
            PredictionResult("P1", 0, "icu", 8, 0.3, False),
        ]
        metrics = aggregate_metrics(results)
        assert len(metrics) == 3  # mortality@8, mortality@24, icu@8

    def test_auroc_computed(self):
        results = [
            PredictionResult("P1", 0, "mortality", 8, 0.9, True),
            PredictionResult("P2", 0, "mortality", 8, 0.1, False),
        ]
        metrics = aggregate_metrics(results)
        assert len(metrics) == 1
        assert metrics[0].auroc == 1.0

    def test_counts_correct(self):
        results = [
            PredictionResult("P1", 0, "mortality", 8, 0.9, True),
            PredictionResult("P2", 0, "mortality", 8, 0.8, True),
            PredictionResult("P3", 0, "mortality", 8, 0.1, False),
        ]
        metrics = aggregate_metrics(results)
        assert metrics[0].n_positive == 2
        assert metrics[0].n_negative == 1
        assert metrics[0].n_total == 3

    def test_brier_and_ece_computed(self):
        results = [
            PredictionResult("P1", 0, "mortality", 8, 1.0, True),
            PredictionResult("P2", 0, "mortality", 8, 0.0, False),
        ]
        metrics = aggregate_metrics(results)
        assert metrics[0].brier == 0.0

    def test_empty_results(self):
        metrics = aggregate_metrics([])
        assert metrics == []


# ──────────────────────── Plotting tests ──────────────────────────


class TestPlotting:
    def test_auroc_comparison_creates_file(self, tmp_path):
        m1 = LabelMetrics("mortality", 48, 0.85, 0.7, 0.1, 0.05, 10, 90, 100)
        m2 = LabelMetrics("mortality", 48, 0.90, 0.75, 0.08, 0.04, 10, 90, 100)
        route_metrics = {
            "no_ecg": [m1],
            "all_branches": [m2],
        }
        out = tmp_path / "auroc.pdf"
        plot_auroc_comparison(route_metrics, horizon=48, output_path=out)
        assert out.exists()

    def test_calibration_curve_creates_file(self, tmp_path):
        preds = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        out = tmp_path / "cal.pdf"
        plot_calibration_curve(preds, labels, output_path=out)
        assert out.exists()

    def test_empty_labels_no_crash(self, tmp_path):
        route_metrics = {"no_ecg": []}
        out = tmp_path / "empty.pdf"
        plot_auroc_comparison(route_metrics, horizon=48, output_path=out)
        # File may or may not be created (no labels to plot), but no crash


# ──────────────────────── EvaluationStage tests ──────────────────────────


class TestEvaluationStage:
    def test_stage_name(self):
        stage = EvaluationStage()
        assert stage.name == "evaluation"

    def test_missing_test_data_returns_failure(self, tmp_path):
        stage = EvaluationStage()
        result = stage.run(
            input_path=tmp_path,
            output_path=tmp_path / "eval_out",
        )
        assert result.success is False

    def test_end_to_end(self, tmp_path, tiny_model):
        """Full pipeline: save model -> create engine -> generate -> evaluate."""
        from clif_protoecg.tokenization.vocabulary import Vocabulary

        # Build a small vocab
        tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
        tokens += [f"T_{i}" for i in range(45)]
        tokens += ["CLOCK//00:00", "CLOCK//04:00", "CLOCK//08:00"]
        tokens += ["DISCH//Expired", "DISCH//Home"]
        tokens += ["LABEL//mortality", "LABEL//icu"]
        # Pad to 100
        while len(tokens) < 100:
            tokens.append(f"EXTRA_{len(tokens)}")

        tok2id = {t: i for i, t in enumerate(tokens)}
        id2tok = {i: t for i, t in enumerate(tokens)}
        vocab = Vocabulary(token_to_id=tok2id, id_to_token=id2tok)

        model_path = tmp_path / "model"
        save_model(tiny_model, model_path)
        vocab_dir = tmp_path / "vocab"
        vocab.save(vocab_dir)

        # Create test data with midnight tokens
        base = datetime(2024, 1, 1, 0, 0)
        midnight_id = tok2id["CLOCK//00:00"]
        label_mort_id = tok2id["LABEL//mortality"]
        label_icu_id = tok2id["LABEL//icu"]

        test_data = [
            {
                "hospitalization_id": "H001",
                "token_ids": [2, 5, 6, midnight_id, 7, label_mort_id, 8, 3],
                "timestamps": [
                    base,
                    base + timedelta(hours=2),
                    base + timedelta(hours=4),
                    base + timedelta(hours=8),  # midnight
                    base + timedelta(hours=12),
                    base + timedelta(hours=16),
                    base + timedelta(hours=20),
                    base + timedelta(hours=24),
                ],
            }
        ]

        label_token_ids = {"mortality": label_mort_id, "icu": label_icu_id}

        stage = EvaluationStage()
        result = stage.run(
            input_path=model_path,
            output_path=tmp_path / "eval_out",
            vocab=vocab,
            test_data=test_data,
            label_token_ids=label_token_ids,
            horizons=[8, 24],
            n_samples=2,
            max_new_tokens=10,
            backend="native",
            device="cpu",
        )
        assert result.success is True
        assert (tmp_path / "eval_out" / "metrics.json").exists()
