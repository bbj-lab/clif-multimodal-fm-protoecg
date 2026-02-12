# Step 8: Evaluation

## Goal

Evaluate model predictions using window-based AUROCs across multiple prediction horizons. Compare performance across ECG ablation routes (`no_ecg`, `fusion_class`, `all_branches`) to quantify the contribution of ECG prototype integration.

## Evaluation Framework

### Prediction Windows

Evaluation uses **midnight-aligned prediction windows** that mirror clinical decision-making:

```
Day 1:  [admission] ... events ... [midnight 1] ← Window 0 context
Day 2:  ... events ...              [midnight 2] ← Window 1 context
Day 3:  ... events ...              [midnight 3] ← Window 2 context
```

At each midnight, the model generates predictions for the next 8, 24, and 48 hours.

### Multi-Horizon Evaluation

| Horizon | Clock Tokens (4h intervals) | Clinical Use Case |
|---------|-------------------------------|-------------------|
| 8 hours | 2 | Shift-level predictions |
| 24 hours | 6 | Daily planning |
| 48 hours | 12 | Discharge planning |

### Ground Truth Extraction

For each prediction window, determine whether each label actually occurred within the horizon:

```python
from datetime import datetime, timedelta


def get_ground_truth(
    token_ids: list[int],
    timestamps: list[datetime],
    context_end_time: datetime,
    horizon_hours: int,
    label_token_ids: dict[str, int],
) -> dict[str, bool]:
    """Extract ground truth labels within the prediction horizon.

    Args:
        token_ids: Full sequence of token IDs
        timestamps: Corresponding timestamps for each token
        context_end_time: Midnight timestamp (end of context)
        horizon_hours: Prediction horizon in hours
        label_token_ids: Map of label names to token IDs

    Returns:
        Dict of label_name -> whether label occurred in horizon
    """
    horizon_end = context_end_time + timedelta(hours=horizon_hours)

    ground_truth = {name: False for name in label_token_ids}

    for tid, ts in zip(token_ids, timestamps):
        if ts <= context_end_time:
            continue  # Before prediction window
        if ts > horizon_end:
            break  # Past horizon

        for label_name, label_id in label_token_ids.items():
            if tid == label_id:
                ground_truth[label_name] = True

    return ground_truth
```

## Metrics

### Primary: AUROC

Area Under the Receiver Operating Characteristic curve. Measures discrimination ability -- how well the model separates positive from negative cases.

```python
from sklearn.metrics import roc_auc_score
import numpy as np


def compute_auroc(
    predictions: list[float],
    labels: list[bool],
) -> float | None:
    """Compute AUROC, returning None for degenerate cases."""
    # Need both positive and negative examples
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    return roc_auc_score(labels, predictions)
```

### Additional Metrics

#### AUPRC (Area Under Precision-Recall Curve)

Better for imbalanced labels (most clinical outcomes are rare):

```python
from sklearn.metrics import average_precision_score


def compute_auprc(predictions: list[float], labels: list[bool]) -> float | None:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    return average_precision_score(labels, predictions)
```

#### Brier Score

Mean squared error between predicted probability and actual outcome. Lower is better:

```python
def compute_brier(predictions: list[float], labels: list[bool]) -> float:
    return np.mean([(p - float(y)) ** 2 for p, y in zip(predictions, labels)])
```

#### Expected Calibration Error (ECE)

Measures how well predicted probabilities match observed frequencies:

```python
def compute_ece(
    predictions: list[float],
    labels: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= p < hi) for p in predictions]
        count = sum(mask)

        if count == 0:
            continue

        bin_preds = [p for p, m in zip(predictions, mask) if m]
        bin_labels = [y for y, m in zip(labels, mask) if m]

        avg_pred = np.mean(bin_preds)
        avg_label = np.mean(bin_labels)

        ece += abs(avg_pred - avg_label) * (count / len(predictions))

    return ece
```

## Evaluation Pipeline

### Per-Label, Per-Horizon Aggregation

```python
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PredictionResult:
    patient_id: str
    window_id: int
    label: str
    horizon_hours: int
    predicted_prob: float
    actual_positive: bool


@dataclass
class LabelMetrics:
    label: str
    horizon_hours: int
    auroc: float | None
    auprc: float | None
    brier: float
    ece: float
    n_positive: int
    n_negative: int
    n_total: int


def aggregate_metrics(
    results: list[PredictionResult],
) -> list[LabelMetrics]:
    """Compute metrics per label per horizon."""
    # Group by (label, horizon)
    groups = defaultdict(list)
    for r in results:
        groups[(r.label, r.horizon_hours)].append(r)

    metrics = []
    for (label, horizon), group in groups.items():
        preds = [r.predicted_prob for r in group]
        labels = [r.actual_positive for r in group]

        metrics.append(LabelMetrics(
            label=label,
            horizon_hours=horizon,
            auroc=compute_auroc(preds, labels),
            auprc=compute_auprc(preds, labels),
            brier=compute_brier(preds, labels),
            ece=compute_ece(preds, labels),
            n_positive=sum(labels),
            n_negative=len(labels) - sum(labels),
            n_total=len(labels),
        ))

    return metrics
```

### Full Evaluation Loop

```python
def evaluate_model(
    model,
    test_dataset,
    vocab: dict[str, int],
    label_token_ids: dict[str, int],
    horizons: list[int] = [8, 24, 48],
    n_samples: int = 8,
    max_new_tokens: int = 500,
    max_days: int = 7,
) -> list[PredictionResult]:
    """Run full evaluation on test set.

    For each test patient:
    1. Find midnight prediction windows
    2. Generate Monte Carlo samples at each window
    3. Extract predicted probabilities
    4. Compare with ground truth
    """
    results = []

    # Identify special tokens
    clock_ids = {v for k, v in vocab.items() if k.startswith("CLOCK//")}
    death_ids = {v for k, v in vocab.items() if k in ("DISCH//Expired",)}
    discharge_ids = {v for k, v in vocab.items()
                     if k.startswith("DISCH//") and k != "DISCH//Expired"}
    eos_id = vocab["[EOS]"]

    for patient in test_dataset:
        # Create midnight windows
        windows = create_midnight_windows(
            patient["token_ids"],
            patient["timestamps"],
            vocab,
            max_days=max_days,
        )

        for window in windows:
            for horizon in horizons:
                # Generate samples
                samples = generate_with_stopping(
                    model=model,
                    context_ids=window["context_ids"],
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                    horizon_hours=horizon,
                    clock_token_ids=clock_ids,
                    death_token_ids=death_ids,
                    discharge_token_ids=discharge_ids,
                    eos_token_id=eos_id,
                )

                # Extract predictions
                predictions = extract_predictions_with_censoring(
                    samples, label_token_ids, death_ids, discharge_ids
                )

                # Get ground truth
                ground_truth = get_ground_truth(
                    patient["token_ids"],
                    patient["timestamps"],
                    window["context_end_time"],
                    horizon,
                    label_token_ids,
                )

                # Record results
                for label_name in label_token_ids:
                    results.append(PredictionResult(
                        patient_id=patient["patient_id"],
                        window_id=window["window_id"],
                        label=label_name,
                        horizon_hours=horizon,
                        predicted_prob=predictions[label_name],
                        actual_positive=ground_truth[label_name],
                    ))

    return results
```

## Ablation Comparison

### Cross-Route Analysis

```python
@dataclass
class AblationResult:
    route: str
    model_size: str
    metrics: list[LabelMetrics]


def compare_ablations(
    results: list[AblationResult],
) -> dict:
    """Compare metrics across ECG routes.

    Generates a comparison table showing AUROC improvement
    from ECG integration.
    """
    comparison = {}

    # Find baseline (no_ecg)
    baseline = next(r for r in results if r.route == "no_ecg")
    baseline_aurocs = {
        (m.label, m.horizon_hours): m.auroc
        for m in baseline.metrics if m.auroc is not None
    }

    for result in results:
        if result.route == "no_ecg":
            continue

        improvements = {}
        for metric in result.metrics:
            key = (metric.label, metric.horizon_hours)
            baseline_auroc = baseline_aurocs.get(key)
            if baseline_auroc and metric.auroc:
                delta = metric.auroc - baseline_auroc
                improvements[key] = {
                    "auroc": metric.auroc,
                    "baseline_auroc": baseline_auroc,
                    "delta": delta,
                    "relative_improvement": delta / baseline_auroc * 100,
                }

        comparison[result.route] = improvements

    return comparison
```

### Summary Table Format

```
ECG Route Comparison (48-hour horizon)
+------------------+----------+---------------+----------------+---------+
| Label            | no_ecg   | fusion_class  | all_branches   | Best    |
+------------------+----------+---------------+----------------+---------+
| mortality        | 0.892    | 0.905 (+0.013)| 0.901 (+0.009) | fusion  |
| icu_admission    | 0.815    | 0.823 (+0.008)| 0.827 (+0.012) | all     |
| mech_vent        | 0.778    | 0.795 (+0.017)| 0.791 (+0.013) | fusion  |
| vasopressor      | 0.801    | 0.812 (+0.011)| 0.815 (+0.014) | all     |
| ...              | ...      | ...           | ...            | ...     |
+------------------+----------+---------------+----------------+---------+
| Macro Average    | 0.821    | 0.834 (+0.013)| 0.833 (+0.012) | fusion  |
+------------------+----------+---------------+----------------+---------+
```

## Statistical Significance

### Bootstrap Confidence Intervals

```python
def bootstrap_auroc(
    predictions: list[float],
    labels: list[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap confidence interval.

    Returns (auroc, lower_ci, upper_ci).
    """
    aurocs = []
    n = len(predictions)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_preds = [predictions[i] for i in idx]
        boot_labels = [labels[i] for i in idx]

        auroc = compute_auroc(boot_preds, boot_labels)
        if auroc is not None:
            aurocs.append(auroc)

    if not aurocs:
        return 0.0, 0.0, 0.0

    alpha = (1 - confidence) / 2
    return (
        np.median(aurocs),
        np.percentile(aurocs, alpha * 100),
        np.percentile(aurocs, (1 - alpha) * 100),
    )
```

## Visualization

### Publication-Quality Plots

```python
import matplotlib.pyplot as plt


def plot_auroc_comparison(
    ablation_results: list[AblationResult],
    horizon: int = 48,
    output_path: str = "figures/auroc_comparison.pdf",
):
    """Bar chart comparing AUROCs across routes for each label."""
    labels = sorted(set(
        m.label for r in ablation_results for m in r.metrics
        if m.horizon_hours == horizon and m.auroc is not None
    ))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.25

    for i, result in enumerate(ablation_results):
        aurocs = []
        for label in labels:
            metric = next(
                (m for m in result.metrics
                 if m.label == label and m.horizon_hours == horizon),
                None
            )
            aurocs.append(metric.auroc if metric and metric.auroc else 0)

        ax.bar(x + i * width, aurocs, width, label=result.route)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Label Prediction Performance ({horizon}h Horizon)")
    ax.legend()
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
```

## Output Files

```
evaluation/
  results/
    {route}_{size}_{horizon}h_predictions.parquet
    {route}_{size}_{horizon}h_metrics.json
  comparison/
    ablation_comparison.json
    ablation_summary.csv
  figures/
    auroc_comparison.pdf
    calibration_curves.pdf
    per_label_horizons.pdf
```

## Dependencies

- Input: Step 7 inference outputs (predicted probabilities)
- Input: Test set ground truth labels
- Output: Metric tables, comparison reports, visualizations
