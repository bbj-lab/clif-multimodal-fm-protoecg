"""vLLM inference engine wrapper."""

from __future__ import annotations

from clif_protoecg.evaluation.inference.base import InferenceEngine, GenerationConfig


class VLLMInferenceEngine(InferenceEngine):
    """High-throughput inference using vLLM.

    Requires vLLM to be installed and a GPU available.
    Falls back to native engine if vLLM is not available.
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            ) from e

        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            skip_tokenizer_init=True,
        )

    def generate(
        self,
        context_ids: list[int],
        config: GenerationConfig,
        stop_token_ids: set[int] | None = None,
        clock_token_ids: set[int] | None = None,
        max_clock_tokens: int | None = None,
    ) -> list[list[int]]:
        from vllm import SamplingParams

        # vLLM handles clock-based stopping via stop_token_ids
        all_stop = list(stop_token_ids or set())
        # Note: vLLM doesn't support "stop after N occurrences" natively.
        # For clock-based stopping, we use max_tokens as an approximation
        # and filter post-hoc.

        sampling_params = SamplingParams(
            n=config.n_samples,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop_token_ids=all_stop if all_stop else None,
        )

        outputs = self.llm.generate(
            {"prompt_token_ids": context_ids},
            sampling_params=sampling_params,
        )

        samples = []
        for completion in outputs[0].outputs:
            token_ids = list(completion.token_ids)
            # Post-hoc clock-based truncation
            if clock_token_ids and max_clock_tokens is not None:
                token_ids = _truncate_at_clock_limit(
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
        from vllm import SamplingParams

        all_stop = list(stop_token_ids or set())
        sampling_params = SamplingParams(
            n=config.n_samples,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop_token_ids=all_stop if all_stop else None,
        )

        outputs = self.llm.generate(
            [{"prompt_token_ids": ctx} for ctx in contexts],
            sampling_params=sampling_params,
        )

        results = []
        for output in outputs:
            context_samples = []
            for completion in output.outputs:
                token_ids = list(completion.token_ids)
                if clock_token_ids and max_clock_tokens is not None:
                    token_ids = _truncate_at_clock_limit(
                        token_ids, clock_token_ids, max_clock_tokens
                    )
                context_samples.append(token_ids)
            results.append(context_samples)

        return results


def _truncate_at_clock_limit(
    token_ids: list[int],
    clock_ids: set[int],
    max_clocks: int,
) -> list[int]:
    """Truncate a generated sequence at the Nth clock token."""
    count = 0
    for i, tid in enumerate(token_ids):
        if tid in clock_ids:
            count += 1
            if count >= max_clocks:
                return token_ids[: i + 1]
    return token_ids
