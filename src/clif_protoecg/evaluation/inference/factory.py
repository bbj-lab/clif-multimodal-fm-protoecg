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
        backend: "native" or "vllm".
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

        return VLLMInferenceEngine(model_path=str(model_path), **kwargs)

    else:
        raise ValueError(f"Unknown inference backend: {backend!r}. Use 'native' or 'vllm'.")
