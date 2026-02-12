# Step 7: Inference

## Goal

Generate predictions for clinical outcomes using the trained causal language model. The model generates continuations from midnight-aligned prediction windows, and we extract predicted probabilities for label tokens from the generated sequences.

## Inference Strategy

### Monte Carlo Sampling

The primary inference approach is Monte Carlo sampling: for each prediction context, generate **N independent samples** and compute the proportion containing each label token.

```
P(label | context) = (# samples containing label token) / N
```

Key parameters:
- `n_samples`: Number of stochastic samples per context (default: 8-32)
- `max_new_tokens`: Maximum tokens to generate per sample (default: 500-2000)
- `temperature`: Sampling temperature (default: 1.0)
- `top_k`: Top-k sampling (default: 50)
- `top_p`: Nucleus sampling threshold (default: 0.95)

### Prediction Windows

Predictions are generated from **midnight-aligned windows**, reflecting clinical workflow patterns (predictions made during night shift review):

```python
from datetime import datetime, timedelta


def create_midnight_windows(
    token_ids: list[int],
    timestamps: list[datetime],
    vocab: dict[str, int],
    max_days: int = 7,
) -> list[dict]:
    """Create prediction windows starting at each midnight.

    Each window uses all tokens up to midnight as context
    and generates forward to evaluate predictions.
    """
    windows = []
    midnight_token_id = vocab.get("CLOCK//00:00")

    for i, (tid, ts) in enumerate(zip(token_ids, timestamps)):
        if tid == midnight_token_id and len(windows) < max_days:
            windows.append({
                "context_ids": token_ids[:i + 1],
                "context_end_time": ts,
                "window_id": len(windows),
            })

    return windows
```

### Horizon-Based Stopping

Generation stops when any of these conditions is met:

1. **EOS token generated** -- natural sequence end
2. **Death token generated** (e.g., `DISCH//Expired`)
3. **Discharge token generated** (e.g., `DISCH//Home`)
4. **Clock token count reached** -- corresponds to the prediction horizon (4-hour clock intervals):
   - 8-hour horizon: stop after 2 clock tokens
   - 24-hour horizon: stop after 6 clock tokens
   - 48-hour horizon: stop after 12 clock tokens

```python
HORIZON_TO_CLOCK_TOKENS = {
    8: 2,    # 2 x 4h = 8 hours
    24: 6,   # 6 x 4h = 24 hours
    48: 12,  # 12 x 4h = 48 hours
}


def generate_with_stopping(
    model,
    context_ids: list[int],
    n_samples: int,
    max_new_tokens: int,
    horizon_hours: int,
    clock_token_ids: set[int],
    death_token_ids: set[int],
    discharge_token_ids: set[int],
    eos_token_id: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> list[list[int]]:
    """Generate N samples with horizon-based stopping."""
    max_clocks = HORIZON_TO_CLOCK_TOKENS[horizon_hours]
    all_samples = []

    for _ in range(n_samples):
        generated = []
        clock_count = 0

        # Initialize with context
        input_ids = context_ids.copy()

        for step in range(max_new_tokens):
            # Get next token prediction
            next_token = sample_next_token(
                model, input_ids, temperature, top_k, top_p
            )
            generated.append(next_token)
            input_ids.append(next_token)

            # Check stopping conditions
            if next_token == eos_token_id:
                break
            if next_token in death_token_ids:
                break
            if next_token in discharge_token_ids:
                break
            if next_token in clock_token_ids:
                clock_count += 1
                if clock_count >= max_clocks:
                    break

        all_samples.append(generated)

    return all_samples
```

## Efficient Inference with KV Cache

### Prefix Caching

For Monte Carlo sampling, compute the prefix (context) KV cache once and reuse it for all N samples:

```python
import torch


class EfficientSampler:
    """Generate multiple samples efficiently using prefix caching."""

    def __init__(self, model, device, max_seq_len=8192):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len

    def compute_prefix_cache(self, context_ids: torch.Tensor):
        """Process context once and cache KV states."""
        with torch.no_grad():
            outputs = self.model(
                context_ids.unsqueeze(0).to(self.device),
                use_cache=True,
            )
        return outputs.past_key_values, outputs.logits[:, -1, :]

    def sample_with_prefix(
        self,
        prefix_cache,
        first_logits,
        n_samples: int,
        max_new_tokens: int,
        stop_tokens: set[int],
        clock_token_ids: set[int],
        max_clock_tokens: int,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Generate N samples reusing the same prefix cache."""
        all_samples = []

        for _ in range(n_samples):
            generated = []
            cache = _clone_cache(prefix_cache)
            logits = first_logits.clone()
            clock_count = 0

            for step in range(max_new_tokens):
                # Sample from logits
                next_token = _sample(logits, temperature)
                generated.append(next_token)

                # Check stopping
                if next_token in stop_tokens:
                    break
                if next_token in clock_token_ids:
                    clock_count += 1
                    if clock_count >= max_clock_tokens:
                        break

                # Forward with cache
                with torch.no_grad():
                    outputs = self.model(
                        torch.tensor([[next_token]], device=self.device),
                        past_key_values=cache,
                        use_cache=True,
                    )
                cache = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

            all_samples.append(generated)

        return all_samples
```

## vLLM Backend (Recommended for Scale)

For large-scale inference, use vLLM's continuous batching:

```python
from vllm import LLM, SamplingParams


class VLLMInference:
    """High-throughput inference using vLLM.

    Advantages:
    - Continuous batching across multiple contexts
    - PagedAttention for efficient memory
    - Higher throughput than native PyTorch
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=8192,
        )

    def generate_batch(
        self,
        contexts: list[list[int]],
        n_samples: int = 8,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_token_ids: list[int] | None = None,
    ) -> list[list[list[int]]]:
        """Generate samples for multiple contexts in parallel.

        Returns: list of (n_samples lists of generated tokens) per context.
        """
        sampling_params = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )

        # vLLM handles batching and scheduling internally
        outputs = self.llm.generate(
            prompt_token_ids=contexts,
            sampling_params=sampling_params,
        )

        results = []
        for output in outputs:
            context_samples = [
                list(completion.token_ids)
                for completion in output.outputs
            ]
            results.append(context_samples)

        return results
```

### Native vs vLLM Comparison

| Feature | Native PyTorch | vLLM |
|---------|---------------|------|
| Throughput | ~200-500 tok/s | ~2000-5000 tok/s |
| Memory efficiency | Manual KV cache | PagedAttention |
| Batching | Manual | Continuous (automatic) |
| Setup complexity | Low | Requires vLLM install |
| Logit access | Full access | Limited |
| Model format | Any | HF format required |

## Extracting Predictions

### From Generated Samples

```python
def extract_predictions(
    generated_samples: list[list[int]],
    label_token_ids: dict[str, int],
    vocab: dict[str, int],
) -> dict[str, float]:
    """Extract predicted probabilities from Monte Carlo samples.

    P(label) = proportion of samples containing the label token.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    n_samples = len(generated_samples)

    predictions = {}
    for label_name, label_id in label_token_ids.items():
        count = sum(
            1 for sample in generated_samples
            if label_id in sample
        )
        predictions[label_name] = count / n_samples

    return predictions
```

### Censoring for Death/Discharge

Samples that terminate in death or discharge require special handling:

```python
def extract_predictions_with_censoring(
    samples: list[list[int]],
    label_token_ids: dict[str, int],
    death_token_ids: set[int],
    discharge_token_ids: set[int],
) -> dict[str, float]:
    """Extract predictions with proper censoring.

    For mortality prediction:
    - Samples ending in death: positive
    - Samples ending in discharge: negative
    - Samples reaching horizon: ambiguous (proportional)

    For other labels:
    - Check if label appears before death/discharge/horizon
    """
    predictions = {}
    n_samples = len(samples)

    for label_name, label_id in label_token_ids.items():
        if "mortality" in label_name:
            # Mortality: death token = positive
            count = sum(
                1 for s in samples
                if any(t in death_token_ids for t in s)
            )
        else:
            # Other labels: check occurrence
            count = sum(1 for s in samples if label_id in s)

        predictions[label_name] = count / n_samples

    return predictions
```

## Logit Accumulation (Alternative to Monte Carlo)

For deterministic predictions, accumulate logits directly:

```python
def logit_accumulation(
    model,
    context_ids: list[int],
    label_token_ids: dict[str, int],
    max_steps: int = 500,
    method: str = "complement",  # "complement", "sum", "max"
) -> dict[str, float]:
    """Accumulate label probabilities from logits without sampling.

    Methods:
    - complement: P = 1 - prod(1 - p_t) over time steps
    - sum: P = sum(p_t) / T
    - max: P = max(p_t) over time steps
    """
    probs = {name: [] for name in label_token_ids}

    input_ids = context_ids.copy()
    for step in range(max_steps):
        logits = model_forward(model, input_ids)
        softmax = torch.softmax(logits[-1], dim=-1)

        for name, tid in label_token_ids.items():
            probs[name].append(softmax[tid].item())

        # Greedy next token for advancing
        next_token = logits[-1].argmax().item()
        input_ids.append(next_token)

    # Aggregate
    results = {}
    for name, p_list in probs.items():
        if method == "complement":
            results[name] = 1.0 - np.prod([1.0 - p for p in p_list])
        elif method == "sum":
            results[name] = np.mean(p_list)
        elif method == "max":
            results[name] = max(p_list)

    return results
```

## Dependencies

- Input: Trained model checkpoint (HF format)
- Input: Tokenized test sequences with vocabulary
- Output: Per-window, per-label predicted probabilities
- Feeds into: Step 8 (Evaluation)
