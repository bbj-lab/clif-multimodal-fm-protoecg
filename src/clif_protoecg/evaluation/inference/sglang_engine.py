"""SGLang inference engine wrapper."""

from __future__ import annotations

from clif_protoecg.evaluation.inference.base import (
    GenerationConfig,
    InferenceEngine,
    truncate_at_clock_limit,
)


class SGLangInferenceEngine(InferenceEngine):
    """High-throughput inference using SGLang.

    Requires sglang to be installed and a GPU available.
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        **kwargs,
    ) -> None:
        try:
            import sglang as sgl
        except ImportError as e:
            raise ImportError(
                "SGLang is not installed. Install with: pip install sglang"
            ) from e

        self.engine = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            dtype="bfloat16",
            context_length=max_model_len,
            skip_tokenizer_init=True,
            **kwargs,
        )

    def generate(
        self,
        context_ids: list[int],
        config: GenerationConfig,
        stop_token_ids: set[int] | None = None,
        clock_token_ids: set[int] | None = None,
        max_clock_tokens: int | None = None,
    ) -> list[list[int]]:
        sampling_params = {
            "n": config.n_samples,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        if stop_token_ids:
            sampling_params["stop_token_ids"] = list(stop_token_ids)

        outputs = self.engine.generate(
            input_ids=context_ids,
            sampling_params=sampling_params,
        )

        # SGLang returns a list of dicts (one per sample) for a single prompt
        if not isinstance(outputs, list):
            outputs = [outputs]

        samples = []
        for out in outputs:
            token_ids = list(out["output_ids"])
            if clock_token_ids and max_clock_tokens is not None:
                token_ids = truncate_at_clock_limit(
                    token_ids, clock_token_ids, max_clock_tokens
                )
            samples.append(token_ids)

        return samples

    def generate_batch(
        self,
        contexts: list[list[int]],
        config: GenerationConfig,
        stop_token_ids: set[int] | None = None,
        clock_token_ids: set[int] | None = None,
        max_clock_tokens: int | None = None,
    ) -> list[list[list[int]]]:
        """Generate samples for multiple contexts in parallel."""
        sampling_params = {
            "n": config.n_samples,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        if stop_token_ids:
            sampling_params["stop_token_ids"] = list(stop_token_ids)

        outputs = self.engine.generate(
            input_ids=contexts,
            sampling_params=sampling_params,
        )

        # With multiple prompts and n>1, SGLang returns a flat list of dicts.
        # Each prompt produces n consecutive outputs.
        n = config.n_samples
        results = []
        for i in range(len(contexts)):
            context_samples = []
            for j in range(n):
                out = outputs[i * n + j]
                token_ids = list(out["output_ids"])
                if clock_token_ids and max_clock_tokens is not None:
                    token_ids = truncate_at_clock_limit(
                        token_ids, clock_token_ids, max_clock_tokens
                    )
                context_samples.append(token_ids)
            results.append(context_samples)

        return results
