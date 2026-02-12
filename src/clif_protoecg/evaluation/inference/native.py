"""Native PyTorch inference with KV cache reuse."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from clif_protoecg.evaluation.inference.base import InferenceEngine, GenerationConfig


class NativeInferenceEngine(InferenceEngine):
    """PyTorch inference engine with prefix KV cache reuse across MC samples."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        context_ids: list[int],
        config: GenerationConfig,
        stop_token_ids: set[int] | None = None,
        clock_token_ids: set[int] | None = None,
        max_clock_tokens: int | None = None,
    ) -> list[list[int]]:
        stop_token_ids = stop_token_ids or set()
        clock_token_ids = clock_token_ids or set()

        # Compute prefix cache once
        ctx = torch.tensor([context_ids], device=self.device, dtype=torch.long)
        with torch.no_grad():
            outputs = self.model(ctx, use_cache=True)
        prefix_cache = outputs.past_key_values
        first_logits = outputs.logits[:, -1, :]

        all_samples: list[list[int]] = []
        for _ in range(config.n_samples):
            generated = self._sample_one(
                prefix_cache,
                first_logits,
                config,
                stop_token_ids,
                clock_token_ids,
                max_clock_tokens,
            )
            all_samples.append(generated)

        return all_samples

    def _sample_one(
        self,
        prefix_cache,
        first_logits: torch.Tensor,
        config: GenerationConfig,
        stop_ids: set[int],
        clock_ids: set[int],
        max_clocks: int | None,
    ) -> list[int]:
        generated: list[int] = []
        cache = _clone_cache(prefix_cache)
        logits = first_logits.clone()
        clock_count = 0

        for _ in range(config.max_new_tokens):
            next_token = _sample_token(logits, config.temperature, config.top_k, config.top_p)
            generated.append(next_token)

            if next_token in stop_ids:
                break
            if next_token in clock_ids:
                clock_count += 1
                if max_clocks is not None and clock_count >= max_clocks:
                    break

            with torch.no_grad():
                inp = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
                outputs = self.model(inp, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        return generated


def _clone_cache(cache) -> object:
    """Deep clone KV cache (supports both tuple and DynamicCache)."""
    if isinstance(cache, tuple):
        return tuple(
            tuple(t.clone() for t in layer)
            for layer in cache
        )
    # DynamicCache from transformers - use copy.deepcopy
    return copy.deepcopy(cache)


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample a single token from logits with temperature, top-k, and top-p."""
    logits = logits.squeeze(0)
    if temperature > 0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
