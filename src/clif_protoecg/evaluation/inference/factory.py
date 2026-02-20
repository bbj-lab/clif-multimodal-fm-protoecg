"""Factory for creating inference engines."""

from __future__ import annotations

from pathlib import Path

from clif_protoecg.evaluation.inference.base import InferenceEngine


def create_inference_engine(
    backend: str,
    model_path: str | Path,
    device: str = "cpu",
    **kwargs,
) -> InferenceEngine:
    """Create an inference engine by backend name.

    Args:
        backend: "native", "vllm", or "sglang".
        model_path: Path to HF model checkpoint.
        device: Torch device for native backend.
        **kwargs: Extra args forwarded to the engine.

    Returns:
        An InferenceEngine instance.
    """
    if backend == "native":
        from clif_protoecg.evaluation.inference.native import NativeInferenceEngine
        from clif_protoecg.training.model import load_model

        model = load_model(Path(model_path))
        return NativeInferenceEngine(model, device=device)

    elif backend == "vllm":
        from clif_protoecg.evaluation.inference.vllm_engine import VLLMInferenceEngine

        vllm_kwargs = {k: v for k, v in kwargs.items()
                       if k in ("max_model_len", "gpu_memory_utilization", "tensor_parallel_size")}
        return VLLMInferenceEngine(model_path=str(model_path), **vllm_kwargs)

    elif backend == "sglang":
        from clif_protoecg.evaluation.inference.sglang_engine import SGLangInferenceEngine

        known = ("max_model_len", "gpu_memory_utilization", "tensor_parallel_size")
        sglang_kwargs = {k: v for k, v in kwargs.items() if k in known}
        # Pass remaining kwargs through (e.g. disable_cuda_graph, attention_backend)
        extra = {k: v for k, v in kwargs.items() if k not in known}
        sglang_kwargs.update(extra)
        return SGLangInferenceEngine(model_path=str(model_path), **sglang_kwargs)

    else:
        raise ValueError(f"Unknown inference backend: {backend!r}. Use 'native', 'vllm', or 'sglang'.")
