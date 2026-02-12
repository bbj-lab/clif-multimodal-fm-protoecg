# Step 3: Label Insertion

## Goal

Insert label tokens at their **first occurrence** timestamp within each hospitalization's event sequence. Labels become part of the causal language modeling training signal -- the model learns to predict them autoregressively, and they serve as evaluation targets during inference.

## Input

- MEDS-like event sequences from Step 1 (one sequence per hospitalization)
- Label definitions from Step 2 (with timestamps and conditions)
- Optional: MIMIC-derived tables for Category 2 labels

## Insertion Rules

### First-Occurrence Only

Each label is inserted **at most once** per hospitalization, at the timestamp of the event that triggers the label. If the same condition occurs multiple times (e.g., multiple AKI episodes), only the first is labeled.

```python
def insert_labels(events: list[dict], labels: list[dict]) -> list[dict]:
    """Insert label tokens into the event sequence.

    Args:
        events: Chronologically sorted list of event dicts
        labels: List of label dicts with {code, time}

    Returns:
        Events with labels inserted at their timestamps
    """
    # De-duplicate labels: keep only first occurrence per label code
    seen_labels = set()
    unique_labels = []
    for label in sorted(labels, key=lambda x: x["time"]):
        if label["code"] not in seen_labels:
            seen_labels.add(label["code"])
            unique_labels.append(label)

    # Merge labels into events
    all_events = events + [
        {"time": l["time"], "code": l["code"], "value": None, "value_cat": None}
        for l in unique_labels
    ]

    # Stable sort preserves relative order for same-timestamp events
    # Labels should appear AFTER clinical events at the same timestamp
    # so the model learns: see clinical data → predict label
    all_events.sort(key=lambda e: (e["time"], _sort_priority(e["code"])))

    return all_events
```

### Sort Priority for Same-Timestamp Events

When a label and clinical event share the same timestamp, the label should appear **after** the triggering clinical event:

```python
def _sort_priority(code: str) -> int:
    """Assign sort priority for events at the same timestamp.

    Lower values sort first. For events at the same time:
    1. Demographics/admission events first
    2. Clinical events (vitals, labs, meds) next
    3. Labels last (they are the consequence of clinical events)
    4. Discharge events at the very end
    """
    if code.startswith("DEMO//") or code.startswith("ADMIT//"):
        return 0
    if code.startswith("LABEL//"):
        return 3
    if code.startswith("DISCH//"):
        return 4
    return 2  # All clinical events
```

### Discharge-Time Labels

Some labels (mortality, discharge disposition) are known only at discharge. These are placed at `discharge_dttm`:

```
... clinical events ... LABEL//mortality DISCH//Expired [EOS]
```

### Prospective vs Retrospective Label Timing

Following the reference repository's handling of ICD codes:

- **Prospective labels** (default): Labels appear at the time the clinician would first know the outcome. For labs, this is result time. For mortality, discharge time. For vasopressor, the admin start time.
- **Billing codes** (ICD, DRG): These are retrospective and placed at `discharge_dttm + 1 minute` to separate them from clinical events.

```python
# Labels that are inherently prospective (known at event time)
PROSPECTIVE_LABELS = {
    "LABEL//icu_admission",     # Known when transfer happens
    "LABEL//mech_vent",         # Known when intubation happens
    "LABEL//vasopressor",       # Known when infusion starts
    "LABEL//crrt",              # Known when CRRT starts
    "LABEL//dnr",               # Known when code status changes
    "LABEL//nippv",
    "LABEL//hfnc",
    "LABEL//ecmo",
    "LABEL//resp_escalation",
    "LABEL//prone",
    "LABEL//prolonged_icu_7d",  # Known when threshold crossed
    "LABEL//prolonged_icu_14d",
}

# Labels only known at discharge
DISCHARGE_LABELS = {
    "LABEL//mortality",
    "LABEL//discharge_home",
    "LABEL//discharge_snf",
    "LABEL//discharge_ltach",
    "LABEL//discharge_hospice",
    "LABEL//discharge_acute",
    "LABEL//readmit_30d",       # Known post-discharge
}
```

## Computing Labels from CLIF Data

### Category 1: CLIF-Only Labels

```python
import polars as pl

def compute_clif_labels(
    hospitalization_id: str,
    hosp_df: pl.DataFrame,
    adt_df: pl.DataFrame,
    resp_df: pl.DataFrame,
    meds_cont_df: pl.DataFrame,
    crrt_df: pl.DataFrame,
    code_status_df: pl.DataFrame,
    ecmo_df: pl.DataFrame,
    position_df: pl.DataFrame,
) -> list[dict]:
    """Compute all CLIF-only labels for a hospitalization.

    Returns list of {"code": "LABEL//...", "time": datetime} dicts.
    """
    labels = []

    # Filter all tables to this hospitalization
    hosp = hosp_df.filter(pl.col("hospitalization_id") == hospitalization_id)
    adt = adt_df.filter(pl.col("hospitalization_id") == hospitalization_id)
    resp = resp_df.filter(pl.col("hospitalization_id") == hospitalization_id)
    # ... etc.

    # 1. Mortality
    if hosp[0, "discharge_category"] == "Expired":
        labels.append({
            "code": "LABEL//mortality",
            "time": hosp[0, "discharge_dttm"],
        })

    # 2. ICU admission (first ICU transfer, excluding admission ICU)
    icu_transfers = adt.filter(
        pl.col("location_category") == "icu"
    ).sort("in_dttm")
    if len(icu_transfers) > 0:
        # Check if first ADT record is already ICU
        first_adt = adt.sort("in_dttm").head(1)
        if first_adt[0, "location_category"] != "icu":
            labels.append({
                "code": "LABEL//icu_admission",
                "time": icu_transfers[0, "in_dttm"],
            })

    # 3. Mechanical ventilation
    imv = resp.filter(pl.col("device_category") == "IMV").sort("recorded_dttm")
    if len(imv) > 0:
        labels.append({
            "code": "LABEL//mech_vent",
            "time": imv[0, "recorded_dttm"],
        })

    # ... (similar pattern for all other labels)

    return labels
```

### Category 2: Labels Requiring Additional Data

For labels that need MIMIC-derived tables (sepsis3, AKI), query the PostgreSQL database:

```python
async def compute_mimic_labels(
    db,
    hospitalization_id: str,
    hadm_id: int,  # Mapped from hospitalization_id
) -> list[dict]:
    """Compute labels requiring MIMIC-derived tables."""
    labels = []

    # Sepsis-3
    sepsis = await db.fetch_one(
        "SELECT sepsis3_onset FROM mimiciv_derived.sepsis3 WHERE hadm_id = $1",
        hadm_id,
    )
    if sepsis and sepsis["sepsis3_onset"]:
        labels.append({
            "code": "LABEL//sepsis3",
            "time": sepsis["sepsis3_onset"],
        })

    # AKI-KDIGO stages
    aki = await db.fetch_all(
        "SELECT stage, onset_time FROM mimiciv_derived.kdigo_stages WHERE hadm_id = $1 ORDER BY onset_time",
        hadm_id,
    )
    seen_stages = set()
    for row in aki:
        stage = row["stage"]
        if stage > 0 and stage not in seen_stages:
            seen_stages.add(stage)
            labels.append({
                "code": f"LABEL//aki_kdigo_{stage}",
                "time": row["onset_time"],
            })

    return labels
```

## Handling Partially-Available Labels

Some labels can be computed from CLIF data but with reduced accuracy:

```python
# Readmission: computable from CLIF hospitalization table
def compute_readmission_label(
    patient_id: str,
    hospitalization_id: str,
    hosp_df: pl.DataFrame,
) -> list[dict]:
    """Check for 30-day readmission using CLIF data only."""
    patient_hosps = hosp_df.filter(
        pl.col("patient_id") == patient_id
    ).sort("admission_dttm")

    # Find this hospitalization's index
    current = patient_hosps.filter(
        pl.col("hospitalization_id") == hospitalization_id
    )
    if len(current) == 0:
        return []

    discharge_time = current[0, "discharge_dttm"]

    # Check for subsequent admission within 30 days
    future = patient_hosps.filter(
        (pl.col("admission_dttm") > discharge_time) &
        (pl.col("admission_dttm") <= discharge_time + timedelta(days=30))
    )

    if len(future) > 0:
        return [{"code": "LABEL//readmit_30d", "time": discharge_time}]
    return []
```

## Output

Modified event sequences with labels interspersed at their occurrence timestamps. The schema remains the same as Step 1 output, with additional `LABEL//*` codes in the event stream.

## Dependencies

- Input: Step 1 MEDS sequences
- Input: Step 2 label definitions
- Input: CLIF parquet files + optional MIMIC PostgreSQL
- Output: Feeds into Step 4 (ECG Prototype Insertion) and Step 5 (Clock/Gap Tokens)
