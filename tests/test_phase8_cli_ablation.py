"""Phase 8: CLI, Ablation, Config YAMLs tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from clif_protoecg.cli import app, _load_config
from clif_protoecg.config import PipelineConfig
from clif_protoecg.ablation import AblationRunner, AblationResult


runner = CliRunner()


# ──────────────────────── Config YAML tests ──────────────────────────


class TestConfigYAMLs:
    def test_default_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/default.yaml")
        assert cfg.training.model_size == "small"
        assert cfg.extraction.bin_mode == "hybrid"
        assert cfg.ecg.route == "all_branches"

    def test_test_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/test.yaml")
        assert cfg.extraction.n_patients == 50
        assert cfg.training.model_size == "tiny"
        assert cfg.training.num_epochs == 3

    def test_tiny_training_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/training/tiny.yaml")
        assert cfg.training.model_size == "tiny"
        assert cfg.training.bf16 is False

    def test_small_training_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/training/small.yaml")
        assert cfg.training.model_size == "small"

    def test_medium_training_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/training/medium.yaml")
        assert cfg.training.model_size == "medium"
        assert cfg.training.gradient_accumulation_steps == 2

    def test_large_training_yaml_loads(self):
        cfg = PipelineConfig.from_yaml("config/training/large.yaml")
        assert cfg.training.model_size == "large"
        assert cfg.training.gradient_accumulation_steps == 4

    def test_config_roundtrip(self, tmp_path):
        cfg = PipelineConfig()
        path = tmp_path / "test.yaml"
        cfg.to_yaml(path)
        loaded = PipelineConfig.from_yaml(path)
        assert loaded.training.model_size == cfg.training.model_size
        assert loaded.extraction.bin_mode == cfg.extraction.bin_mode

    def test_default_yaml_has_all_sections(self):
        cfg = PipelineConfig.from_yaml("config/default.yaml")
        assert cfg.data is not None
        assert cfg.extraction is not None
        assert cfg.split is not None
        assert cfg.labels is not None
        assert cfg.ecg is not None
        assert cfg.time_tokens is not None
        assert cfg.tokenization is not None
        assert cfg.training is not None
        assert cfg.inference is not None
        assert cfg.evaluation is not None

    def test_default_yaml_horizons(self):
        cfg = PipelineConfig.from_yaml("config/default.yaml")
        assert cfg.evaluation.horizons == [8, 24, 48]

    def test_test_yaml_fewer_horizons(self):
        cfg = PipelineConfig.from_yaml("config/test.yaml")
        assert cfg.evaluation.horizons == [8, 24]


class TestPipelineConfigPaths:
    def test_processed_dir(self):
        cfg = PipelineConfig()
        assert cfg.processed_dir == Path("./data/processed")

    def test_metadata_dir(self):
        cfg = PipelineConfig()
        assert cfg.metadata_dir == Path("./data/processed/metadata")

    def test_tokenized_dir(self):
        cfg = PipelineConfig()
        assert cfg.tokenized_dir == Path("./data/processed/tokenized")

    def test_checkpoints_dir(self):
        cfg = PipelineConfig()
        assert cfg.checkpoints_dir == Path("./data/processed/checkpoints")

    def test_results_dir(self):
        cfg = PipelineConfig()
        assert cfg.results_dir == Path("./data/processed/results")


# ──────────────────────── CLI tests ──────────────────────────


class TestCLI:
    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ProtoECG" in result.output

    def test_extract_help(self):
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--n-patients" in result.output

    def test_tokenize_help(self):
        result = runner.invoke(app, ["tokenize", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--size" in result.output
        assert "--route" in result.output

    def test_evaluate_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    def test_ablation_help(self):
        result = runner.invoke(app, ["ablation", "--help"])
        assert result.exit_code == 0
        assert "--routes" in result.output
        assert "--sizes" in result.output

    def test_run_all_help(self):
        result = runner.invoke(app, ["run-all", "--help"])
        assert result.exit_code == 0

    def test_load_config_default(self):
        cfg = _load_config(None)
        assert isinstance(cfg, PipelineConfig)

    def test_load_config_from_yaml(self):
        cfg = _load_config(Path("config/test.yaml"))
        assert cfg.extraction.n_patients == 50

    def test_load_config_nonexistent_returns_default(self, tmp_path):
        cfg = _load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, PipelineConfig)
        assert cfg.extraction.n_patients is None  # default

    def test_tokenize_requires_extract_first(self):
        result = runner.invoke(app, ["tokenize"])
        # Should fail because sequences.parquet doesn't exist yet
        assert result.exit_code == 1

    def test_train_requires_tokenize_first(self):
        result = runner.invoke(app, ["train", "--size", "tiny", "--route", "no_ecg"])
        # Should fail because tokenized data doesn't exist yet
        assert result.exit_code != 0

    def test_evaluate_command_runs(self):
        result = runner.invoke(app, ["evaluate", "--backend", "native"])
        assert result.exit_code == 0

    def test_run_all_command_runs(self):
        result = runner.invoke(app, ["run-all"])
        assert result.exit_code == 0


# ──────────────────────── Ablation tests ──────────────────────────


class TestAblationResult:
    def test_defaults(self):
        r = AblationResult(route="no_ecg", model_size="tiny")
        assert r.route == "no_ecg"
        assert r.model_size == "tiny"
        assert r.success is False
        assert r.metrics == {}
        assert r.model_path is None


class TestAblationRunner:
    def test_init_defaults(self):
        cfg = PipelineConfig()
        runner = AblationRunner(cfg)
        assert runner.routes == ["no_ecg", "fusion_class", "all_branches"]
        assert runner.sizes == ["tiny"]

    def test_init_custom_routes(self):
        cfg = PipelineConfig()
        runner = AblationRunner(cfg, routes=["no_ecg", "fusion_class"], sizes=["small", "medium"])
        assert runner.routes == ["no_ecg", "fusion_class"]
        assert runner.sizes == ["small", "medium"]

    def test_run_produces_results(self, tmp_path):
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        runner = AblationRunner(cfg, routes=["no_ecg", "fusion_class"], sizes=["tiny"])
        results = runner.run()
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].route == "no_ecg"
        assert results[1].route == "fusion_class"

    def test_run_creates_report(self, tmp_path):
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        runner = AblationRunner(cfg, routes=["no_ecg"], sizes=["tiny"])
        runner.run()
        report_path = tmp_path / "results" / "ablation" / "ablation_summary.json"
        assert report_path.exists()

    def test_compare_routes_empty_without_baseline(self):
        cfg = PipelineConfig()
        runner = AblationRunner(cfg)
        # No results yet
        comparison = runner.compare_routes()
        assert comparison == {}

    def test_compare_routes_with_baseline(self, tmp_path):
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        runner = AblationRunner(cfg, routes=["no_ecg", "fusion_class"], sizes=["tiny"])
        runner.run()
        comparison = runner.compare_routes()
        assert "fusion_class" in comparison

    def test_model_paths_created(self, tmp_path):
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        runner = AblationRunner(cfg, routes=["no_ecg"], sizes=["tiny"])
        results = runner.run()
        assert results[0].model_path is not None

    def test_three_routes_three_sizes(self, tmp_path):
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        runner = AblationRunner(
            cfg,
            routes=["no_ecg", "fusion_class", "all_branches"],
            sizes=["tiny", "small"],
        )
        results = runner.run()
        assert len(results) == 6  # 3 routes x 2 sizes

    def test_ablation_cli_command(self, tmp_path):
        """Test ablation CLI command with config."""
        config_path = tmp_path / "cfg.yaml"
        cfg = PipelineConfig()
        cfg.data.output_dir = tmp_path
        cfg.to_yaml(config_path)
        result = runner.invoke(
            app,
            ["ablation", "--config", str(config_path), "--routes", "no_ecg", "--sizes", "tiny"],
        )
        assert result.exit_code == 0


# ──────────────────────── Integration tests ──────────────────────────


class TestConfigIntegration:
    def test_yaml_files_are_valid_yaml(self):
        yaml_files = [
            "config/default.yaml",
            "config/test.yaml",
            "config/training/tiny.yaml",
            "config/training/small.yaml",
            "config/training/medium.yaml",
            "config/training/large.yaml",
        ]
        for path in yaml_files:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{path} is not a valid YAML dict"

    def test_all_training_configs_load_as_pipeline_config(self):
        sizes = ["tiny", "small", "medium", "large"]
        for size in sizes:
            cfg = PipelineConfig.from_yaml(f"config/training/{size}.yaml")
            assert cfg.training.model_size == size
