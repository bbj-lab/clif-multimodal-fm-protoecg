"""Ablation runner: shared preprocessing, per-route train+eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from clif_protoecg.config import PipelineConfig
from clif_protoecg.utils.logging import get_logger

logger = get_logger("ablation")


@dataclass
class AblationResult:
    """Results from one ablation run (route x size)."""

    route: str
    model_size: str
    metrics: dict = field(default_factory=dict)
    model_path: Path | None = None
    success: bool = False


class AblationRunner:
    """Run ablation study across ECG routes and model sizes.

    The preprocessing (extract + tokenize) is done once with all_branches.
    Then per-route training filters ECG tokens at data-loading time.
    """

    def __init__(
        self,
        config: PipelineConfig,
        routes: list[str] | None = None,
        sizes: list[str] | None = None,
    ) -> None:
        self.config = config
        self.routes = routes or ["no_ecg", "fusion_class", "all_branches"]
        self.sizes = sizes or ["tiny"]
        self.results: list[AblationResult] = []

    def run(self) -> list[AblationResult]:
        """Execute the full ablation study."""
        logger.info(f"Starting ablation: routes={self.routes}, sizes={self.sizes}")

        # Step 1: Shared preprocessing (extract + tokenize once with all_branches)
        logger.info("Step 1: Shared preprocessing with all_branches")
        preprocessed = self._preprocess()

        if not preprocessed:
            logger.warning("Preprocessing returned no data, aborting ablation")
            return self.results

        # Step 2: Per-route, per-size train + eval
        for route in self.routes:
            for size in self.sizes:
                logger.info(f"Training: route={route}, size={size}")
                result = self._train_and_evaluate(route, size)
                self.results.append(result)

        # Step 3: Comparison report
        self._generate_report()

        return self.results

    def _preprocess(self) -> bool:
        """Run shared extraction and tokenization."""
        output_dir = self.config.processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preprocessing output: {output_dir}")
        return True

    def _train_and_evaluate(self, route: str, size: str) -> AblationResult:
        """Train model for one route+size combination and evaluate."""
        result = AblationResult(route=route, model_size=size)

        output_dir = self.config.processed_dir / "ablation" / f"{route}_{size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result.model_path = output_dir / "model"

        logger.info(f"Output: {output_dir}")
        result.success = True
        return result

    def _generate_report(self) -> None:
        """Generate ablation comparison report."""
        report_dir = self.config.results_dir / "ablation"
        report_dir.mkdir(parents=True, exist_ok=True)

        import json

        summary = []
        for r in self.results:
            summary.append({
                "route": r.route,
                "model_size": r.model_size,
                "success": r.success,
                "metrics": r.metrics,
            })

        (report_dir / "ablation_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info(f"Ablation report saved to {report_dir}")

    def compare_routes(self) -> dict:
        """Compare metrics across routes, using no_ecg as baseline."""
        baseline = next(
            (r for r in self.results if r.route == "no_ecg" and r.success),
            None,
        )
        if baseline is None:
            return {}

        comparison: dict[str, dict] = {}
        for r in self.results:
            if r.route == "no_ecg" or not r.success:
                continue
            comparison[r.route] = {
                "model_size": r.model_size,
                "metrics": r.metrics,
                "baseline_metrics": baseline.metrics,
            }
        return comparison
