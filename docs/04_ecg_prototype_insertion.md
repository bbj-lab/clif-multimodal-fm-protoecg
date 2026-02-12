# Step 4: ECG Prototype Insertion

## Goal

Insert ECG prototype tokens into the event sequence at the time each ECG was recorded. ECG prototypes come from the ProtoECGNet foundation model and encode cardiac rhythm, morphology, and global features as discrete tokens.

## Input

- MEDS-like sequences with labels inserted (from Step 3)
- ECG prototype CSV: `./data/ecg_prototypes_with_shifted_dates.csv`

## ECG Prototype Data Format

The CSV file contains one row per ECG recording with columns:

| Column | Description |
|--------|-------------|
| `subject_id` | MIMIC subject ID (int64) — must be mapped to CLIF `patient_id` |
| `hadm_id` | MIMIC admission ID (int64) — must be mapped to CLIF `hospitalization_id` |
| `shifted_ecg_time` | Timestamp of the ECG recording (already shifted to match MIMIC-IV date offsets) |
| `top1class` / `top1score` | Top-1 predicted ECG classification and confidence |
| `top2class` / `top2score` | Top-2 prediction |
| `top3class` / `top3score` | Top-3 prediction |
| `1d_prototype_num` | Rhythm (1D) branch prototype ID |
| `1d_similarity` | Similarity score to 1D prototype (0-1) |
| `2d_partial_prototype_num` | Morphology (2D partial) branch prototype ID |
| `2d_partial_similarity` | Similarity to 2D morphology prototype |
| `2d_global_prototype_num` | Global (2D) branch prototype ID |
| `2d_global_similarity` | Similarity to 2D global prototype |

### ProtoECGNet Branch Details

| Branch | Prototype Count | Purpose |
|--------|----------------|---------|
| RHYTHM_1D | 16 classes | 1D rhythm analysis (e.g., AFIB, NSR, VT) |
| MORPH_2D | 52 classes | 2D morphology (ST changes, Q waves, etc.) |
| GLOBAL_2D | 3 classes | 2D global features (normal, borderline, abnormal) |

## Three ECG Routes (Ablation Study)

The model supports three ECG integration strategies for ablation comparison:

### Route 1: `no_ecg` (Baseline)

No ECG tokens are inserted. The model uses only clinical EHR data. This serves as the baseline for measuring ECG contribution.

### Route 2: `fusion_class` (3 tokens per ECG)

Insert 3 tokens per ECG event:

```
ECG//Class/{top1class}                    # Classification label
ECG//Prototype/1D/{1d_prototype_num}      # Fused prototype ID (1D branch)
ECG//Similarity/1D                        # Similarity score (quantized to decile)
```

Example:
```
... DT//30-60 ECG//Class/AFIB ECG//Prototype/1D/7 ECG//Similarity/1D Q_8 DT//60-120 ...
```

### Route 3: `all_branches` (up to 7 tokens per ECG)

Insert up to 7 tokens capturing all three branches:

```
ECG//Class/{top1class}                              # Classification
ECG//Prototype/1D/{1d_prototype_num}                # 1D rhythm prototype
ECG//Similarity/1D                                  # 1D similarity (quantized)
ECG//Prototype/2D_Morphology/{2d_partial_proto}     # 2D morphology prototype
ECG//Similarity/2D_Morphology                       # 2D morphology similarity
ECG//Prototype/2D_Global/{2d_global_proto}          # 2D global prototype
ECG//Similarity/2D_Global                           # 2D global similarity
```

## Implementation

### ECG Prototype Tokenizer

```python
from enum import Enum
from dataclasses import dataclass
import polars as pl
import numpy as np


class ECGRoute(Enum):
    NO_ECG = "no_ecg"
    FUSION_CLASS = "fusion_class"
    ALL_BRANCHES = "all_branches"


@dataclass
class ECGPrototypeTokenizer:
    """Tokenize ECG prototype data into MEDS events."""

    route: ECGRoute
    similarity_quantile_edges: dict[str, list[float]] | None = None

    def load_prototypes(
        self,
        csv_path: str,
        id_mapping: dict[int, str] | None = None,
    ) -> pl.DataFrame:
        """Load and validate ECG prototype CSV.

        Args:
            csv_path: Path to ecg_prototypes_with_shifted_dates.csv
            id_mapping: Maps MIMIC hadm_id (int) → CLIF hospitalization_id (str).
                        Required because the ECG CSV uses MIMIC IDs while
                        the rest of the pipeline uses CLIF IDs.
        """
        df = pl.read_csv(csv_path)

        # Parse shifted_ecg_time → ecg_time
        df = df.with_columns(
            pl.col("shifted_ecg_time").str.to_datetime().alias("ecg_time")
        )

        # Map MIMIC hadm_id → CLIF hospitalization_id
        if id_mapping is not None:
            mapping_df = pl.DataFrame({
                "hadm_id": list(id_mapping.keys()),
                "hospitalization_id": list(id_mapping.values()),
            })
            df = df.join(mapping_df, on="hadm_id", how="inner")

        return df

    def compute_similarity_quantiles(
        self, df: pl.DataFrame, n_bins: int = 10
    ) -> dict[str, list[float]]:
        """Compute quantile bin edges for similarity scores per branch."""
        branches = {
            "1D": "1d_similarity",
            "2D_Morphology": "2d_partial_similarity",
            "2D_Global": "2d_global_similarity",
        }

        edges = {}
        percentiles = np.linspace(0, 100, n_bins + 1)

        for branch_name, col in branches.items():
            if col in df.columns:
                values = df[col].drop_nulls().to_numpy()
                if len(values) >= n_bins:
                    edges[branch_name] = np.percentile(values, percentiles).tolist()

        self.similarity_quantile_edges = edges
        return edges

    def get_similarity_bin(self, value: float, branch: str) -> int:
        """Map a similarity score to its quantile bin (0-9)."""
        if self.similarity_quantile_edges is None:
            raise ValueError("Call compute_similarity_quantiles first")

        edges = self.similarity_quantile_edges.get(branch)
        if edges is None:
            return 5  # Default to middle bin

        for i in range(1, len(edges)):
            if value <= edges[i]:
                return i - 1
        return len(edges) - 2  # Last bin

    def tokenize_ecg(self, row: dict) -> list[dict]:
        """Convert a single ECG row to MEDS events.

        Returns list of event dicts to insert at ecg_time.
        """
        ecg_time = row["ecg_time"]
        events = []

        if self.route == ECGRoute.NO_ECG:
            return []

        # Classification token (always included for fusion_class and all_branches)
        top_class = row.get("top1class", "UNKNOWN")
        events.append({
            "time": ecg_time,
            "code": f"ECG//Class/{top_class}",
            "value": row.get("top1score"),
            "value_cat": None,
        })

        if self.route in (ECGRoute.FUSION_CLASS, ECGRoute.ALL_BRANCHES):
            # 1D branch (always included)
            proto_1d = row.get("1d_prototype_num")
            sim_1d = row.get("1d_similarity")
            if proto_1d is not None:
                events.append({
                    "time": ecg_time,
                    "code": f"ECG//Prototype/1D/{int(proto_1d)}",
                    "value": None,
                    "value_cat": None,
                })
            if sim_1d is not None:
                events.append({
                    "time": ecg_time,
                    "code": "ECG//Similarity/1D",
                    "value": sim_1d,
                    "value_cat": None,
                })

        if self.route == ECGRoute.ALL_BRANCHES:
            # 2D Morphology branch
            proto_2d = row.get("2d_partial_prototype_num")
            sim_2d = row.get("2d_partial_similarity")
            if proto_2d is not None:
                events.append({
                    "time": ecg_time,
                    "code": f"ECG//Prototype/2D_Morphology/{int(proto_2d)}",
                    "value": None,
                    "value_cat": None,
                })
            if sim_2d is not None:
                events.append({
                    "time": ecg_time,
                    "code": "ECG//Similarity/2D_Morphology",
                    "value": sim_2d,
                    "value_cat": None,
                })

            # 2D Global branch
            proto_g = row.get("2d_global_prototype_num")
            sim_g = row.get("2d_global_similarity")
            if proto_g is not None:
                events.append({
                    "time": ecg_time,
                    "code": f"ECG//Prototype/2D_Global/{int(proto_g)}",
                    "value": None,
                    "value_cat": None,
                })
            if sim_g is not None:
                events.append({
                    "time": ecg_time,
                    "code": "ECG//Similarity/2D_Global",
                    "value": sim_g,
                    "value_cat": None,
                })

        return events
```

### Injection into Event Sequences

```python
def inject_ecg_prototypes(
    events: list[dict],
    ecg_events: list[dict],
) -> list[dict]:
    """Insert ECG events into the clinical event sequence.

    ECG events are merged chronologically with existing events.
    At the same timestamp, ECG tokens appear after clinical events
    but before labels.
    """
    all_events = events + ecg_events
    all_events.sort(key=lambda e: (e["time"], _sort_priority(e["code"])))
    return all_events


def _sort_priority(code: str) -> int:
    """Sort priority: clinical=2, ECG=2.5, labels=3, discharge=4."""
    if code.startswith("DEMO//") or code.startswith("ADMIT//"):
        return 0
    if code.startswith("ECG//"):
        return 2.5  # After clinical events, before labels
    if code.startswith("LABEL//"):
        return 3
    if code.startswith("DISCH//"):
        return 4
    return 2
```

### ECG Token Filter (for Route-Based Training)

Tokenize once with `all_branches` and filter at training time:

```python
class ECGTokenFilter:
    """Filter ECG tokens based on route at training time.

    Tokenize once with all_branches, then filter per-route:
    - no_ecg: Remove all ECG// tokens
    - fusion_class: Keep Class + 1D prototype/similarity only
    - all_branches: Keep everything

    This avoids re-tokenizing for each ablation configuration.
    """

    ROUTE_ALLOWED_PREFIXES = {
        "no_ecg": set(),  # Remove all ECG tokens
        "fusion_class": {"ECG//Class", "ECG//Prototype/1D", "ECG//Similarity/1D"},
        "all_branches": None,  # Keep all (no filtering)
    }

    def __init__(self, vocab: dict[str, int], route: str):
        self.route = route
        allowed = self.ROUTE_ALLOWED_PREFIXES.get(route)
        if allowed is None:
            self.filtered_ids = frozenset()  # Keep all
        else:
            self.filtered_ids = frozenset(
                tid for token, tid in vocab.items()
                if token.startswith("ECG//") and not any(
                    token.startswith(prefix) for prefix in allowed
                )
            )

    def filter(self, token_ids: list[int]) -> list[int]:
        if not self.filtered_ids:
            return token_ids
        return [t for t in token_ids if t not in self.filtered_ids]
```

## ECG Filtering to Admission Windows

ECGs outside the hospitalization window should be excluded. The reference repo uses a 7-day lookback before admission:

```python
def filter_ecgs_to_window(
    ecg_df: pl.DataFrame,
    hosp_df: pl.DataFrame,
    lookback_days: int = 7,
) -> pl.DataFrame:
    """Keep only ECGs within [admission - lookback, discharge]."""
    joined = ecg_df.join(
        hosp_df.select("hospitalization_id", "admission_dttm", "discharge_dttm"),
        on="hospitalization_id",
    )

    return joined.filter(
        (pl.col("ecg_time") >= pl.col("admission_dttm") - pl.duration(days=lookback_days)) &
        (pl.col("ecg_time") <= pl.col("discharge_dttm"))
    )
```

## Output

Event sequences with ECG prototype tokens inserted chronologically. The number of additional tokens per ECG depends on the route:

| Route | Tokens per ECG | Total added (est.) |
|-------|---------------|-------------------|
| no_ecg | 0 | 0 |
| fusion_class | 3 | ~3x ECG count |
| all_branches | up to 7 | ~7x ECG count |

## Dependencies

- Input: Step 3 labeled event sequences
- Input: `./data/ecg_prototypes_with_shifted_dates.csv`
- Output: Feeds into Step 5 (Clock and Gap Tokens)
