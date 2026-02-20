"""Evaluation stage orchestrator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from clif_protoecg.core.constants import HORIZON_TO_CLOCK_TOKENS
from clif_protoecg.core.stage import BaseStage, StageResult
from clif_protoecg.evaluation.inference.base import GenerationConfig
from clif_protoecg.evaluation.inference.factory import create_inference_engine
from clif_protoecg.evaluation.inference.base import truncate_at_clock_limit
from clif_protoecg.evaluation.metrics.aggregator import (
    PredictionResult,
    aggregate_metrics,
)
from clif_protoecg.evaluation.prediction import (
    create_midnight_windows,
    extract_predictions_with_censoring,
    get_ground_truth,
)
from clif_protoecg.tokenization.vocabulary import Vocabulary
from clif_protoecg.utils.logging import get_logger

logger = get_logger("evaluation")


@dataclass
class _Task:
    """Metadata for one generation task (one window)."""

    patient_idx: int
    window_id: int
    context_ids: list[int]
    context_end_time: Any
    hospitalization_id: str


class EvaluationStage(BaseStage):
    @property
    def name(self) -> str:
        return "evaluation"

    def run(
        self,
        input_path: Path,
        output_path: Path,
        vocab: Vocabulary | None = None,
        test_data: list[dict] | None = None,
        label_token_ids: dict[str, int] | None = None,
        horizons: list[int] | None = None,
        n_samples: int = 8,
        max_new_tokens: int = 500,
        backend: str = "native",
        device: str = "cpu",
        **kwargs: Any,
    ) -> StageResult:
        start = time.time()
        horizons = horizons or [8, 24, 48]

        if test_data is None or label_token_ids is None:
            return StageResult(
                success=False,
                output_path=output_path,
                duration_seconds=time.time() - start,
                metrics={"error": "test_data and label_token_ids required"},
            )

        if vocab is None:
            vocab = Vocabulary.load(input_path.parent)

        # Read model's max context length from config
        import json as _json

        model_config_path = input_path / "config.json"
        if model_config_path.exists():
            model_max_len = _json.loads(model_config_path.read_text()).get(
                "max_position_embeddings", 8192
            )
        else:
            model_max_len = 8192
        # Reserve room for generation
        max_context_len = model_max_len - max_new_tokens

        engine = create_inference_engine(backend, input_path, device=device,
                                         max_model_len=model_max_len, **kwargs)

        # Identify special tokens
        clock_ids = {v for k, v in vocab.token_to_id.items() if k.startswith("CLOCK//")}
        death_ids = {v for k, v in vocab.token_to_id.items() if k == "DISCH//Expired"}
        discharge_ids = {
            v
            for k, v in vocab.token_to_id.items()
            if k.startswith("DISCH//") and k != "DISCH//Expired"
        }
        stop_ids = death_ids | discharge_ids | {vocab.eos_id}

        midnight_id = vocab.token_to_id.get("CLOCK//00:00")
        if midnight_id is None:
            return StageResult(
                success=False,
                output_path=output_path,
                duration_seconds=time.time() - start,
                metrics={"error": "CLOCK//00:00 not in vocabulary"},
            )

        gen_config = GenerationConfig(
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
        )

        # Build flat task list (one per unique window)
        tasks: list[_Task] = []
        patients_with_windows = 0
        n_truncated = 0
        for p_idx, patient in enumerate(test_data):
            windows = create_midnight_windows(
                patient["token_ids"],
                patient["timestamps"],
                midnight_id,
                hospitalization_id=patient.get("hospitalization_id", ""),
            )
            if windows:
                patients_with_windows += 1
            for window in windows:
                ctx = window.context_ids
                if len(ctx) > max_context_len:
                    ctx = ctx[-max_context_len:]  # keep most recent tokens
                    n_truncated += 1
                tasks.append(
                    _Task(
                        patient_idx=p_idx,
                        window_id=window.window_id,
                        context_ids=ctx,
                        context_end_time=window.context_end_time,
                        hospitalization_id=window.hospitalization_id,
                    )
                )

        if n_truncated:
            logger.info(
                f"Truncated {n_truncated} contexts to {max_context_len} tokens "
                f"(model max={model_max_len}, reserved {max_new_tokens} for generation)"
            )

        total_windows = len(tasks)
        total_eval_tasks = total_windows * len(horizons)
        logger.info(
            f"Evaluation plan: {len(test_data)} patients, "
            f"{total_windows} windows, {len(horizons)} horizons, "
            f"{total_eval_tasks} evaluation tasks ({n_samples} samples each)"
        )

        # Generate samples — one call per window, reuse across horizons
        use_batched = hasattr(engine, "generate_batch")

        if use_batched:
            # Batch all windows into one engine call
            logger.info(f"Batched generation ({backend}): {total_windows} windows x {n_samples} samples")
            all_contexts = [t.context_ids for t in tasks]
            all_samples = engine.generate_batch(
                contexts=all_contexts,
                config=gen_config,
                stop_token_ids=stop_ids,
            )
            gen_elapsed = time.time() - start
            logger.info(f"Generation complete in {gen_elapsed:.0f}s")
        else:
            # Native backend: sequential with KV cache reuse
            all_samples: list[list[list[int]]] = []
            pbar = tqdm(tasks, desc="Generating", unit="window", mininterval=0.5)
            for task in pbar:
                samples = engine.generate(
                    context_ids=task.context_ids,
                    config=gen_config,
                    stop_token_ids=stop_ids,
                )
                all_samples.append(samples)
            pbar.close()

        # Score predictions — truncate per horizon and extract labels
        logger.info("Scoring predictions across horizons...")
        all_results: list[PredictionResult] = []

        pbar = tqdm(total=total_eval_tasks, desc="Scoring", unit="task", mininterval=0.5)
        for task_idx, task in enumerate(tasks):
            samples = all_samples[task_idx]
            patient = test_data[task.patient_idx]

            for horizon in horizons:
                max_clocks = HORIZON_TO_CLOCK_TOKENS.get(horizon)

                # Truncate samples at clock limit for this horizon
                if clock_ids and max_clocks is not None:
                    truncated = [
                        truncate_at_clock_limit(s, clock_ids, max_clocks)
                        for s in samples
                    ]
                else:
                    truncated = samples

                predictions = extract_predictions_with_censoring(
                    truncated, label_token_ids, death_ids, discharge_ids
                )

                ground_truth = get_ground_truth(
                    patient["token_ids"],
                    patient["timestamps"],
                    task.context_end_time,
                    horizon,
                    label_token_ids,
                )

                for label_name in label_token_ids:
                    all_results.append(
                        PredictionResult(
                            patient_id=task.hospitalization_id,
                            window_id=task.window_id,
                            label=label_name,
                            horizon_hours=horizon,
                            predicted_prob=predictions[label_name],
                            actual_positive=ground_truth[label_name],
                        )
                    )

                pbar.update(1)

        pbar.close()
        elapsed = time.time() - start
        logger.info(
            f"Evaluation complete: {total_eval_tasks} tasks in {elapsed:.0f}s "
            f"({total_eval_tasks / elapsed:.1f} tasks/s), "
            f"{patients_with_windows}/{len(test_data)} patients had windows"
        )

        logger.info("Aggregating metrics...")
        metrics_list = aggregate_metrics(all_results)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_dict = {}
        for m in metrics_list:
            key = f"{m.label}_{m.horizon_hours}h"
            metrics_dict[key] = {
                "auroc": m.auroc,
                "auprc": m.auprc,
                "brier": m.brier,
                "ece": m.ece,
                "n_positive": m.n_positive,
                "n_total": m.n_total,
            }

        (output_path / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        total_elapsed = time.time() - start
        logger.info(
            f"Evaluation complete in {total_elapsed:.0f}s, "
            f"{len(metrics_list)} label-horizon metrics saved to {output_path / 'metrics.json'}"
        )

        return StageResult(
            success=True,
            output_path=output_path,
            duration_seconds=total_elapsed,
            metrics=metrics_dict,
        )
