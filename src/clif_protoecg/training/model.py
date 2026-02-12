"""MedicalLM: LlamaForCausalLM wrapper with size presets."""

from __future__ import annotations

from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

SIZE_PRESETS: dict[str, dict] = {
    "tiny":   {"d_model": 128,  "n_layers": 4,  "n_heads": 4,  "n_kv_heads": 2,  "d_ff": 512,  "max_pos": 8192},
    "small":  {"d_model": 256,  "n_layers": 8,  "n_heads": 8,  "n_kv_heads": 4,  "d_ff": 1024, "max_pos": 8192},
    "medium": {"d_model": 512,  "n_layers": 12, "n_heads": 16, "n_kv_heads": 4,  "d_ff": 2048, "max_pos": 16384},
    "large":  {"d_model": 768,  "n_layers": 12, "n_heads": 12, "n_kv_heads": 4,  "d_ff": 3072, "max_pos": 32768},
}


def create_model(
    vocab_size: int,
    size_name: str = "tiny",
    max_position_embeddings: int | None = None,
) -> LlamaForCausalLM:
    """Create a LlamaForCausalLM model from a size preset."""
    preset = SIZE_PRESETS[size_name]
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=preset["d_model"],
        intermediate_size=preset["d_ff"],
        num_hidden_layers=preset["n_layers"],
        num_attention_heads=preset["n_heads"],
        num_key_value_heads=preset["n_kv_heads"],
        max_position_embeddings=max_position_embeddings or preset["max_pos"],
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        use_cache=True,
    )
    return LlamaForCausalLM(config)


def save_model(model: LlamaForCausalLM, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)


def load_model(path: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(path)
