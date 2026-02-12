# Step 1: CLIF to MEDS-like Format

## Goal

Convert 15 CLIF v2.1.0 parquet tables into a unified, chronologically-ordered event sequence per hospitalization. Each hospitalization produces one sequence of `(timestamp, code, value)` tuples that will later be tokenized for the causal language model.

This step mirrors the reference repository's `stages/extract.py` and `stages/meds_export.py` but sources from CLIF rather than raw MIMIC-IV.

## CLI Interface

```bash
# Full extraction
protoecg-pipeline extract --data-dir ./data --output-dir ./data/processed

# Test run on 50 patients
protoecg-pipeline extract --data-dir ./data --output-dir ./data/processed --n-patients 50

# Pure quantile binning (20 ventiles)
protoecg-pipeline extract --data-dir ./data --output-dir ./data/processed \
    --bin-mode quantile --n-bins 20

# Clinical bins only (no quantile fallback or padding)
protoecg-pipeline extract --data-dir ./data --output-dir ./data/processed \
    --bin-mode clinical_only
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-patients` | `None` (all) | Limit to N patients for test runs |
| `--bin-mode` | `hybrid` | `hybrid` (clinical where available + quantile fallback), `clinical_only` (clinical bins only), or `quantile` (equal-frequency everywhere) |
| `--n-bins` | `10` | Number of quantile bins for `quantile` mode or fallback bins in `hybrid` mode (e.g., 10=deciles, 20=ventiles) |
| `--min-events` | `1000` | Minimum event count for a code to be included (profiling threshold) |
| `--min-admissions` | `100` | Minimum admission count for a code to be included |
| `--seed` | `42` | Random seed for patient sampling in test mode |

## Input

All files in `./data/`:

| File | Key Columns | Timestamp Column(s) |
|------|-------------|-------------------|
| `clif_patient.parquet` | patient_id, race_category, sex_category, birth_date, death_dttm | Static |
| `clif_hospitalization.parquet` | patient_id, hospitalization_id, admission_dttm, discharge_dttm, age_at_admission, discharge_category | admission_dttm, discharge_dttm |
| `clif_adt.parquet` | hospitalization_id, patient_id, in_dttm, out_dttm, location_category, location_type | in_dttm |
| `clif_vitals.parquet` | hospitalization_id, recorded_dttm, vital_category, vital_value | recorded_dttm |
| `clif_labs.parquet` | hospitalization_id, lab_order_dttm, lab_collect_dttm, lab_result_dttm, lab_category, lab_value_numeric | lab_order_dttm, lab_result_dttm |
| `clif_medication_admin_continuous.parquet` | hospitalization_id, admin_dttm, med_category, med_group, med_dose, med_dose_unit, mar_action_category | admin_dttm |
| `clif_medication_admin_intermittent.parquet` | (same as continuous) | admin_dttm |
| `clif_respiratory_support.parquet` | hospitalization_id, recorded_dttm, device_category, mode_category, fio2_set, peep_set, etc. | recorded_dttm |
| `clif_patient_assessments.parquet` | hospitalization_id, recorded_dttm, assessment_category, numerical_value, categorical_value | recorded_dttm |
| `clif_code_status.parquet` | patient_id, start_dttm, code_status_category | start_dttm |
| `clif_position.parquet` | hospitalization_id, recorded_dttm, position_category | recorded_dttm |
| `clif_hospital_diagnosis.parquet` | hospitalization_id, diagnosis_code, diagnosis_code_format, diagnosis_primary, poa_present | Static (admission-level) |
| `clif_patient_procedures.parquet` | hospitalization_id, procedure_code, procedure_code_format, procedure_billed_dttm | procedure_billed_dttm |
| `clif_crrt_therapy.parquet` | hospitalization_id, recorded_dttm, crrt_mode_category, blood_flow_rate, ultrafiltration_out | recorded_dttm |
| `clif_ecmo_mcs.parquet` | hospitalization_id, recorded_dttm, device_category, mcs_group, flow | recorded_dttm |

## Output

Per-split parquet files with schema:

```
hospitalization_id: str
patient_id: str
events: list[dict]   # chronologically sorted
```

Each event dict:
```python
{
    "time": datetime,       # UTC timestamp
    "code": str,            # e.g. "VITAL//hr", "LAB_ORDER//troponin_t"
    "value": float | None,  # numeric value where applicable
    "value_cat": str | None # categorical value where applicable
}
```

### Token Naming Convention

Following the reference repository, use `//` as the namespace separator:
```
DEMO//AGE_30-39      (not DEMO|age)
VITAL//hr            (not VITAL|hr)
LAB_RESULT//troponin_t   (not LAB|troponin_t)
ECG//Class/AFIB      (not ECG|Class/AFIB)
```

## Event Mapping Rules

### 1. Demographics (at `admission_dttm`, sort priority 0)

Static events anchoring the sequence start. Following the reference repo's bucketing:

```python
AGE_BUCKETS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

def age_bucket(age: int) -> str:
    if age < 18: return "18-29"  # clamp
    if age >= 80: return "80+"
    decade = (age // 10) * 10
    return f"{decade}-{decade+9}"
```

Events emitted at `admission_dttm`:
```
DEMO//AGE_{bucket}           # e.g. DEMO//AGE_60-69
DEMO//SEX_{sex_category}     # e.g. DEMO//SEX_Male
DEMO//RACE_{race_category}   # e.g. DEMO//RACE_White
ADMIT//TYPE_{admission_type_category}  # e.g. ADMIT//TYPE_elective
```

### 2. Admission Diagnoses (at `admission_dttm`, sort priority 1)

From `clif_hospital_diagnosis` where `poa_present == 1`:
```
ICD//{diagnosis_code}    # e.g. ICD//E11.9, ICD//I50.9
```

Only `diagnosis_primary == 1` diagnoses emit at admission. Secondary diagnoses with `poa_present == 1` follow.

### 3. ADT Events (at `in_dttm`, sort priority 2)

From `clif_adt`:
```
ADT//{location_category}    # e.g. ADT//icu, ADT//ward, ADT//ed
```

### 4. Vital Signs (at `recorded_dttm`, sort priority 2)

From `clif_vitals`:
```
VITAL//{vital_category}     # e.g. VITAL//hr, VITAL//sbp, VITAL//spo2
```

Followed by a value quantile token (see Value Discretization below):
```
VITAL//hr Q_7               # heart rate in the 8th decile
```

**`vital_category` values** in CLIF: `hr`, `sbp`, `dbp`, `mbp`, `resp_rate`, `spo2`, `temp_c`, `height_cm`, `weight_kg`

### 5. Labs — Multi-Step Events

Labs use the 3-timestamp model from the user's design:

**Step 1 — Lab Order** (at `lab_order_dttm`, or `lab_collect_dttm` if order is null):
```
LAB_ORDER//{lab_category}   # e.g. LAB_ORDER//troponin_t
```

**Step 2 — Lab Result** (at `lab_result_dttm`):
```
LAB_RESULT//{lab_category}  # e.g. LAB_RESULT//troponin_t
```
Followed by value quantile token:
```
LAB_RESULT//troponin_t Q_9  # result value in the 10th decile (very high)
```

If `lab_result_dttm` is null, emit a single event at `lab_collect_dttm` with the value.

**Key `lab_category` values**: `albumin`, `alkaline_phosphatase`, `alt`, `anion_gap`, `ast`, `base_excess`, `bicarbonate`, `bilirubin_total`, `bilirubin_direct`, `bnp`, `bun`, `calcium`, `chloride`, `creatinine`, `crp`, `d_dimer`, `ferritin`, `fibrinogen`, `glucose`, `hba1c`, `hemoglobin`, `inr`, `lactate`, `ldh`, `lipase`, `magnesium`, `pco2`, `ph`, `phosphate`, `platelets`, `po2`, `potassium`, `procalcitonin`, `ptt`, `sodium`, `troponin_i`, `troponin_t`, `wbc`

### 6. Medications — Continuous (at `admin_dttm`, sort priority 2)

From `clif_medication_admin_continuous`:

```
MED_CONT//{med_category}        # start or rate change
MED_CONT_STOP//{med_category}   # stop/pause/hold
```

Logic:
- `mar_action_category` in (`"stop"`, `"paused"`, `"held"`) OR `med_dose == 0` → emit `MED_CONT_STOP//`
- Otherwise → emit `MED_CONT//` followed by dose quantile token

Key `med_group` values from CLIF data: `vasoactives`, `sedation`, `anticoagulation`, `analgesics`

### 7. Medications — Intermittent (at `admin_dttm`, sort priority 2)

From `clif_medication_admin_intermittent`:
```
MED_BOLUS//{med_category}   # e.g. MED_BOLUS//acetaminophen
```
Followed by dose quantile token.

### 8. Respiratory Support (at `recorded_dttm`, sort priority 2)

From `clif_respiratory_support`:
```
RESP//{device_category}     # e.g. RESP//IMV, RESP//NIPPV, RESP//High Flow NC
```

Followed by parameter tokens (only when non-null):
```
RESP_PARAM//fio2 Q_6
RESP_PARAM//peep Q_3
RESP_PARAM//tidal_volume Q_5
RESP_PARAM//resp_rate_set Q_4
RESP_PARAM//pressure_support Q_2
```

**`device_category` values**: `IMV`, `NIPPV`, `CPAP`, `High Flow NC`, `Nasal Cannula`, `Face Mask`, `Room Air`, `Tracheostomy Collar`

### 9. Patient Assessments (at `recorded_dttm`, sort priority 2)

From `clif_patient_assessments`:
```
ASSESS//{assessment_category}    # e.g. ASSESS//gcs_total, ASSESS//rass
```
Followed by value token. For integer-valued assessments (GCS, RASS, Braden), keep as-is rather than binning.

**Key `assessment_category` values**: `gcs_eye`, `gcs_verbal`, `gcs_motor`, `gcs_total`, `braden_total`, `rass`, `cam_icu`, `cam_inattention`, `sbt_fail_reason`, `pain_score`

### 10. Code Status (at `start_dttm`, sort priority 2)

From `clif_code_status`:
```
CODE_STATUS//{code_status_category}    # e.g. CODE_STATUS//Full, CODE_STATUS//DNR
```

### 11. Position (at `recorded_dttm`, sort priority 2)

From `clif_position`:
```
POSITION//{position_category}    # e.g. POSITION//prone, POSITION//not_prone
```

### 12. CRRT (at `recorded_dttm`, sort priority 2)

From `clif_crrt_therapy`:
```
CRRT//{crrt_mode_category}       # e.g. CRRT//cvvh, CRRT//cvvhd
CRRT_PARAM//blood_flow Q_5
CRRT_PARAM//uf_out Q_3
```

### 13. ECMO/MCS (at `recorded_dttm`, sort priority 2)

From `clif_ecmo_mcs`:
```
ECMO//{device_category}          # e.g. ECMO//ECMO
ECMO_PARAM//flow Q_4
```

### 14. Procedures (at `procedure_billed_dttm`, sort priority 2)

From `clif_patient_procedures`:
```
PROC//{procedure_code}    # e.g. PROC//0BH17EZ
```

### 15. Discharge Events (at `discharge_dttm`, sort priority 4)

A general discharge token followed by disposition:
```
DISCH//GENERAL                          # always emitted at discharge
DISCH//{discharge_category}             # e.g. DISCH//Expired, DISCH//Home
```

### 16. Billing Codes — Retrospective (at `discharge_dttm + 1 min`, sort priority 5)

Following the reference repo's ICD leakage handling, billing codes are placed 1 minute after discharge to prevent information leakage in prospective evaluation:

```
ICD//{diagnosis_code}    # All diagnoses (not just POA)
DRG//{drg_code}          # If available
```

## Sort Priority

When multiple events share a timestamp, sort by priority (lower first):

| Priority | Event Types | Rationale |
|----------|-------------|-----------|
| 0 | `DEMO//`, `ADMIT//` | Sequence anchor |
| 1 | `CLOCK//` | Temporal anchor |
| 2 | `VITAL//`, `LAB_*//`, `MED_*//`, `RESP//`, `ASSESS//`, `ADT//`, `CODE_STATUS//`, `POSITION//`, `CRRT//`, `ECMO//`, `PROC//` | Clinical events |
| 2.5 | `ECG//` | ECG tokens (inserted in Step 4) |
| 3 | `LABEL//` | Labels (inserted in Step 3) |
| 4 | `DISCH//` | Discharge events |
| 5 | `ICD//` (retrospective) | Billing codes |

## Value Discretization

Three binning modes are available via `--bin-mode`:

### Mode 1: `hybrid` (Default)

Uses clinically meaningful bin boundaries where they are defined, and falls back to quantile bins for all other codes. This is the recommended approach because it preserves clinical interpretability for well-understood measurements while still discretizing everything else.

**Clinically meaningful bin boundaries** for vital signs and common labs:

```python
CLINICAL_BINS = {
    # Vitals (7-8 boundaries each, yielding 6-7 bins)
    "VITAL//hr":        [0, 50, 60, 80, 100, 120, 150, 200],
    "VITAL//sbp":       [0, 70, 90, 110, 130, 150, 180, 250],
    "VITAL//dbp":       [0, 40, 50, 60, 70, 80, 100, 140],
    "VITAL//mbp":       [0, 50, 60, 65, 75, 90, 110, 150],
    "VITAL//resp_rate": [0, 8, 12, 16, 20, 25, 30, 40],
    "VITAL//spo2":      [0, 70, 80, 88, 92, 95, 98, 100],
    "VITAL//temp_c":    [33, 35, 36, 36.5, 37.5, 38, 39, 42],

    # Common critical labs (9-10 boundaries each)
    "LAB_RESULT//potassium": [0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 10],
    "LAB_RESULT//sodium":    [0, 120, 125, 130, 135, 140, 145, 150, 155, 180],
    "LAB_RESULT//glucose":   [0, 40, 60, 70, 100, 140, 200, 300, 500, 1000],
    "LAB_RESULT//creatinine":[0, 0.4, 0.7, 1.0, 1.3, 1.8, 2.5, 4.0, 8.0, 20],
    "LAB_RESULT//lactate":   [0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10, 20],
    "LAB_RESULT//hemoglobin":[0, 5, 7, 8, 10, 12, 14, 16, 20],
    "LAB_RESULT//platelets": [0, 20, 50, 100, 150, 250, 400, 600, 1000],
    "LAB_RESULT//wbc":       [0, 1, 4, 7, 11, 15, 20, 30, 50],
    "LAB_RESULT//troponin_t":[0, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0, 5.0, 50],
    "LAB_RESULT//inr":       [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10],
    "LAB_RESULT//ph":        [6.8, 7.0, 7.15, 7.25, 7.35, 7.45, 7.55, 7.65],
    "LAB_RESULT//pco2":      [0, 20, 30, 35, 40, 45, 50, 60, 80, 120],
    "LAB_RESULT//bicarbonate":[0, 10, 15, 18, 22, 26, 30, 35, 50],
}
```

**Behavior**:
- Codes in `CLINICAL_BINS`: Use the defined boundaries as-is. If fewer than `--n-bins` boundaries, pad with quantile-based sub-bins within the clinical ranges to reach the minimum.
- Codes **not** in `CLINICAL_BINS`: Fall back to quantile binning with `--n-bins` bins (default 10).

### Mode 2: `clinical_only`

Uses **only** the clinically meaningful bin boundaries above. Codes without a defined entry in `CLINICAL_BINS` get **no value discretization** — they emit the code token alone without a quantile token. This mode is useful for interpretability experiments where you want bin boundaries to always correspond to known clinical thresholds.

```bash
protoecg-pipeline extract --bin-mode clinical_only
```

### Mode 3: `quantile`

Pure equal-frequency binning computed on the **training set only**. Ignores all clinical bin definitions. Every numeric code gets the same number of bins (controlled by `--n-bins`).

```python
import numpy as np

def compute_quantile_bins(
    values: list[float],
    n_bins: int = 10,
) -> list[float]:
    """Compute quantile bin edges.

    Args:
        values: Training set values for a specific code
        n_bins: Number of bins (10=deciles, 20=ventiles, 5=quintiles)

    Returns:
        List of n_bins + 1 boundary values
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # De-duplicate edges (can happen with discrete values)
    return np.unique(edges).tolist()
```

### Bin Token Format

Bins are labeled `Q_0` through `Q_{n-1}`:
```
VITAL//hr Q_7    # heart rate in the 8th decile (hybrid/quantile mode)
VITAL//hr Q_3    # heart rate 80-100 bpm (clinical_only mode, Q_3 maps to 3rd clinical range)
```

The bin-to-range mapping is stored in `metadata/bin_edges.json` for all modes.

### Minimum Bin Count

In `hybrid` and `quantile` modes, all codes with numeric values use **at least 10 bins**. If clinical bins define fewer than 10 boundaries for a given code, pad with quantile-based sub-bins within the clinical ranges. In `clinical_only` mode, bin count is determined solely by the defined clinical boundaries.

## Code Profiling (Frequency Filtering)

Following the reference repo's `stages/profile.py`, only include codes that meet minimum frequency thresholds in the training set:

```python
PROFILING_THRESHOLDS = {
    "labs":        {"min_events": 1000, "min_admissions": 100},
    "vitals":      {"min_events": 500,  "min_admissions": 50},
    "medications": {"min_events": 200,  "min_admissions": 50},
    "assessments": {"min_events": 200,  "min_admissions": 50},
    "procedures":  {"min_events": 100,  "min_admissions": 20},
    "diagnoses":   {"min_events": 50,   "min_admissions": 10},
}
```

A `CLINICAL_WHITELIST` overrides threshold-based filtering for clinically important but rare codes:
```python
CLINICAL_WHITELIST = {
    "LAB_RESULT//troponin_t", "LAB_RESULT//troponin_i", "LAB_RESULT//bnp",
    "LAB_RESULT//procalcitonin", "LAB_RESULT//lactate", "LAB_RESULT//d_dimer",
    "RESP//IMV", "CRRT//cvvh", "ECMO//ECMO",
    "CODE_STATUS//DNR", "POSITION//prone",
}
```

## Algorithm

```python
import polars as pl
from pathlib import Path

def build_all_sequences(
    data_dir: Path,
    output_dir: Path,
    n_patients: int | None = None,
    bin_mode: str = "hybrid",
    n_bins: int = 10,
    seed: int = 42,
):
    """Main extraction pipeline.

    Args:
        data_dir: Directory containing CLIF parquet files
        output_dir: Where to write processed sequences
        n_patients: If set, randomly sample this many patients (for test runs)
        bin_mode: "hybrid", "clinical_only", or "quantile"
        n_bins: Number of bins for quantile/fallback bins
        seed: Random seed for patient sampling
    """
    # 1. Load all CLIF tables
    tables = load_clif_tables(data_dir)

    # 2. Get patient list
    all_patients = tables["hospitalization"]["patient_id"].unique().to_list()
    if n_patients is not None:
        rng = np.random.default_rng(seed)
        all_patients = rng.choice(all_patients, size=min(n_patients, len(all_patients)), replace=False).tolist()

    # 3. Get hospitalization list for selected patients
    hosps = tables["hospitalization"].filter(
        pl.col("patient_id").is_in(all_patients)
    )

    # 4. Split patients (train/val/test by anchor year)
    splits = load_patient_splits(all_patients)  # from MIMIC postgres

    # 5. Profile codes on training set
    train_hosps = hosps.filter(pl.col("patient_id").is_in(splits["train"]))
    code_profile = profile_codes(tables, train_hosps)

    # 6. Compute bin edges on training set
    if bin_mode == "hybrid":
        bin_edges = CLINICAL_BINS.copy()
        # Pad clinical bins to meet minimum bin count
        bin_edges = pad_clinical_bins(bin_edges, tables, train_hosps, min_bins=max(n_bins, 10))
        # Fill remaining codes with quantile bins
        remaining = compute_remaining_quantile_bins(tables, train_hosps, code_profile, n_bins=max(n_bins, 10))
        bin_edges.update(remaining)
    elif bin_mode == "clinical_only":
        bin_edges = CLINICAL_BINS.copy()  # No fallback, no padding
    else:  # quantile
        bin_edges = compute_all_quantile_bins(tables, train_hosps, code_profile, n_bins=n_bins)

    # 7. Extract sequences for all hospitalizations
    for split_name, patient_ids in splits.items():
        split_hosps = hosps.filter(pl.col("patient_id").is_in(patient_ids))
        sequences = []

        for row in split_hosps.iter_rows(named=True):
            events = build_hospitalization_sequence(
                hospitalization_id=row["hospitalization_id"],
                patient_id=row["patient_id"],
                tables=tables,
                code_profile=code_profile,
                bin_edges=bin_edges,
            )
            sequences.append({
                "hospitalization_id": row["hospitalization_id"],
                "patient_id": row["patient_id"],
                "events": events,
            })

        # Write to parquet
        write_sequences(sequences, output_dir / split_name / "meds_sequences.parquet")

    # 8. Save metadata
    save_metadata(output_dir / "metadata", bin_edges, code_profile, splits)


def build_hospitalization_sequence(
    hospitalization_id: str,
    patient_id: str,
    tables: dict,
    code_profile: set,
    bin_edges: dict,
) -> list[dict]:
    """Build the event sequence for a single hospitalization."""
    events = []

    hosp = tables["hospitalization"].filter(
        pl.col("hospitalization_id") == hospitalization_id
    ).row(0, named=True)
    patient = tables["patient"].filter(
        pl.col("patient_id") == patient_id
    ).row(0, named=True)

    admission_dttm = hosp["admission_dttm"]
    discharge_dttm = hosp["discharge_dttm"]

    # Demographics
    events += extract_demographics(hosp, patient, admission_dttm)

    # POA diagnoses
    events += extract_poa_diagnoses(tables["hospital_diagnosis"], hospitalization_id, admission_dttm)

    # Clinical events (all filtered to hospitalization)
    events += extract_adt(tables["adt"], hospitalization_id)
    events += extract_vitals(tables["vitals"], hospitalization_id, code_profile, bin_edges)
    events += extract_labs(tables["labs"], hospitalization_id, code_profile, bin_edges)
    events += extract_meds_continuous(tables["medication_admin_continuous"], hospitalization_id, code_profile, bin_edges)
    events += extract_meds_intermittent(tables["medication_admin_intermittent"], hospitalization_id, code_profile, bin_edges)
    events += extract_respiratory(tables["respiratory_support"], hospitalization_id, bin_edges)
    events += extract_assessments(tables["patient_assessments"], hospitalization_id, code_profile)
    events += extract_code_status(tables["code_status"], patient_id, admission_dttm, discharge_dttm)
    events += extract_position(tables["position"], hospitalization_id)
    events += extract_crrt(tables["crrt_therapy"], hospitalization_id, bin_edges)
    events += extract_ecmo(tables["ecmo_mcs"], hospitalization_id, bin_edges)
    events += extract_procedures(tables["patient_procedures"], hospitalization_id, code_profile)

    # Discharge events
    events += extract_discharge(hosp)

    # Retrospective billing codes (discharge + 1 min)
    events += extract_billing_codes(tables["hospital_diagnosis"], hospitalization_id, discharge_dttm)

    # Sort chronologically with priority-based tie-breaking
    events.sort(key=lambda e: (e["time"], sort_priority(e["code"])))

    # Clip to admission window
    events = [e for e in events if admission_dttm <= e["time"]]

    return events
```

## Train/Test Split

Patient-level split using MIMIC anchor year from the postgres DB:

```sql
SELECT subject_id, anchor_year_group
FROM mimiciv_hosp.patients;
```

Split logic (following reference repo's `stages/split.py`):
- **Train**: anchor_year_group in ("2008 - 2010", "2011 - 2013", "2014 - 2016") → ~87%
- **Test**: anchor_year_group in ("2017 - 2019") → ~13%
- **Val**: Optional — last 2 years of train split OR random 10% of train

Map MIMIC `subject_id` → CLIF `patient_id`. All hospitalizations for a given patient go into the same split.

```python
async def load_patient_splits(patient_ids: list[str], db_url: str) -> dict[str, list[str]]:
    """Load patient splits from MIMIC postgres DB."""
    # ... query mimiciv_hosp.patients for anchor_year_group
    # Map subject_id to patient_id
    # Assign to train/test based on anchor_year_group
```

## Output Files

```
data/processed/
├── train/
│   └── meds_sequences.parquet
├── val/
│   └── meds_sequences.parquet
├── test/
│   └── meds_sequences.parquet
└── metadata/
    ├── bin_edges.json            # discretization boundaries per code
    ├── code_profile.json         # included codes with frequency stats
    ├── patient_splits.json       # patient_id → split mapping
    └── extraction_config.json    # n_patients, bin_mode, n_bins, etc.
```

## Dependencies

- Input: Raw CLIF parquet files in `./data/`
- Input: MIMIC postgres DB (for patient-level split via anchor_year_group)
- Output: Feeds into Step 2 (Label Definitions) and Step 3 (Label Insertion)
