# Step 2: Label Definitions

## Goal

Define prediction labels for the foundation model. Labels are binary events inserted at their first occurrence during a hospitalization. The model learns to predict them autoregressively, and they serve as evaluation targets during inference.

Three categories:
1. **CLIF-only**: Computable entirely from CLIF v2.1.0 tables
2. **CLIF-partial**: Computable from CLIF with some caveats (may miss edge cases)
3. **Requires MIMIC**: Needs raw MIMIC-IV or MIMIC-IV derived tables

Following the reference repository's `labels/` package, labels are registered via a `@label` decorator into a `LabelRegistry` singleton.

## Discharge Token Design

A **general discharge token** (`DISCH//GENERAL`) is always emitted at `discharge_dttm` before the specific disposition token. This gives the model a unified "discharge is happening" signal regardless of destination:

```
... events ... DISCH//GENERAL DISCH//Home [EOS]
... events ... DISCH//GENERAL DISCH//Expired [EOS]
```

## Billing Code Tokens

ICD codes, procedure codes, and DRG codes are inserted as retrospective tokens at `discharge_dttm + 1 minute` to avoid information leakage during prospective evaluation:

```
... DISCH//GENERAL DISCH//Home ICD//E11.9 ICD//I50.9 PROC//0BH17EZ [EOS]
```

---

## Category 1: CLIF-Only Labels

### Mortality & Discharge

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| Hospital mortality | `LABEL//mortality` | clif_hospitalization | `discharge_category == "Expired"` | discharge_dttm |
| Mortality or hospice | `LABEL//mortality_or_hospice` | clif_hospitalization | `discharge_category in ("Expired", "Hospice")` | discharge_dttm |
| Discharge home | `LABEL//discharge_home` | clif_hospitalization | `discharge_category == "Home"` | discharge_dttm |
| Discharge SNF | `LABEL//discharge_snf` | clif_hospitalization | `discharge_category like "Skilled Nursing%"` | discharge_dttm |
| Discharge LTACH | `LABEL//discharge_ltach` | clif_hospitalization | `discharge_category == "LTACH"` | discharge_dttm |
| Discharge hospice | `LABEL//discharge_hospice` | clif_hospitalization | `discharge_category == "Hospice"` | discharge_dttm |
| Discharge acute transfer | `LABEL//discharge_acute` | clif_hospitalization | `discharge_category == "Acute Care Hospital"` | discharge_dttm |

### ICU & Location

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| ICU admission | `LABEL//icu_admission` | clif_adt | `location_category == "icu"` (not if first ADT already ICU) | First `in_dttm` for ICU |
| Prolonged ICU (>7d) | `LABEL//prolonged_icu_7d` | clif_adt | Sum of ICU intervals ≥ 168 hours | When threshold crossed |
| Prolonged ICU (>14d) | `LABEL//prolonged_icu_14d` | clif_adt | Sum of ICU intervals ≥ 336 hours | When threshold crossed |
| Stepdown admission | `LABEL//stepdown_admission` | clif_adt | `location_category == "stepdown"` | First `in_dttm` for stepdown |

### Respiratory

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| Mechanical ventilation | `LABEL//mech_vent` | clif_respiratory_support | `device_category == "IMV"` | First IMV `recorded_dttm` |
| New intubation | `LABEL//new_intubation` | clif_respiratory_support | `device_category == "IMV"` AND `tracheostomy == False` | First such `recorded_dttm` |
| NIPPV | `LABEL//nippv` | clif_respiratory_support | `device_category == "NIPPV"` | First `recorded_dttm` |
| HFNC | `LABEL//hfnc` | clif_respiratory_support | `device_category == "High Flow NC"` | First `recorded_dttm` |
| Rapid resp escalation | `LABEL//resp_escalation` | clif_respiratory_support | ≥2 device severity level increase within 24h | Escalation time |
| High FiO2 (>0.6) | `LABEL//high_fio2` | clif_respiratory_support | `fio2_set > 0.6` | First such `recorded_dttm` |
| Hypoxemia | `LABEL//hypoxemia` | clif_vitals | `vital_category == "spo2"` AND `vital_value < 88` | First occurrence |
| Prone positioning | `LABEL//prone` | clif_position | `position_category == "prone"` | First `recorded_dttm` |
| Reintubation 48h | `LABEL//reintubation_48h` | clif_respiratory_support | IMV → non-IMV → IMV within 48h | Second IMV onset |

```python
# Respiratory device severity for escalation detection
RESP_SEVERITY = {
    "Room Air": 0, "Nasal Cannula": 1, "Face Mask": 1,
    "High Flow NC": 2, "CPAP": 2, "NIPPV": 3, "IMV": 4,
}
```

### Cardiovascular

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| Vasopressor start | `LABEL//vasopressor` | clif_medication_admin_continuous | `med_group == "vasoactives"` AND not stop | First `admin_dttm` |
| Hypotension (MAP<65) | `LABEL//hypotension` | clif_vitals | `vital_category == "mbp"` AND `vital_value < 65` | First occurrence |
| Tachycardia (HR>120) | `LABEL//tachycardia` | clif_vitals | `vital_category == "hr"` AND `vital_value > 120` | First occurrence |
| Bradycardia (HR<50) | `LABEL//bradycardia` | clif_vitals | `vital_category == "hr"` AND `vital_value < 50` | First occurrence |
| ECMO initiation | `LABEL//ecmo` | clif_ecmo_mcs | `mcs_group == "ECMO"` | First `recorded_dttm` |

### Renal

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| CRRT initiation | `LABEL//crrt` | clif_crrt_therapy | Any CRRT record | First `recorded_dttm` |

### Code Status

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| DNR/CMO transition | `LABEL//dnr` | clif_code_status | Transition from "Full" to DNR/DNAR/CMO | First such `start_dttm` |

### Lab-Based Critical Values

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| Hyperkalemia | `LABEL//hyperkalemia` | clif_labs | `lab_category == "potassium"` AND `lab_value_numeric > 6.0` | First `lab_result_dttm` |
| Severe hyperkalemia | `LABEL//severe_hyperkalemia` | clif_labs | `lab_category == "potassium"` AND `lab_value_numeric > 6.5` | First `lab_result_dttm` |
| Hypoglycemia | `LABEL//hypoglycemia` | clif_labs | `lab_category == "glucose"` AND `lab_value_numeric < 50` | First `lab_result_dttm` |
| Severe anemia | `LABEL//severe_anemia` | clif_labs | `lab_category == "hemoglobin"` AND `lab_value_numeric < 7` | First `lab_result_dttm` |
| Thrombocytopenia | `LABEL//thrombocytopenia` | clif_labs | `lab_category == "platelets"` AND `lab_value_numeric < 50000` | First `lab_result_dttm` |
| Lactate elevation | `LABEL//lactate_elevated` | clif_labs | `lab_category == "lactate"` AND `lab_value_numeric > 2.0` | First `lab_result_dttm` |
| Severe lactate | `LABEL//lactate_severe` | clif_labs | `lab_category == "lactate"` AND `lab_value_numeric > 4.0` | First `lab_result_dttm` |
| Coagulopathy | `LABEL//coagulopathy` | clif_labs | `lab_category == "inr"` AND `lab_value_numeric > 2.0` | First `lab_result_dttm` |
| Acidosis | `LABEL//acidosis` | clif_labs | `lab_category == "ph"` AND `lab_value_numeric < 7.25` | First `lab_result_dttm` |
| Acute creatinine rise | `LABEL//creatinine_rise` | clif_labs | `lab_category == "creatinine"` AND value ≥ 1.5x first value | First qualifying `lab_result_dttm` |
| Elevated troponin | `LABEL//troponin_elevated` | clif_labs | `lab_category in ("troponin_t", "troponin_i")` AND above ULN | First `lab_result_dttm` |
| Elevated BNP | `LABEL//bnp_elevated` | clif_labs | `lab_category == "bnp"` AND `lab_value_numeric > 300` | First `lab_result_dttm` |

### Assessment-Based

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| GCS deterioration (<8) | `LABEL//gcs_low` | clif_patient_assessments | `assessment_category == "gcs_total"` AND `numerical_value < 8` | First occurrence |
| Delirium (CAM-ICU+) | `LABEL//delirium` | clif_patient_assessments | `assessment_category == "cam_icu"` AND positive result | First occurrence |
| Pressure injury risk | `LABEL//braden_low` | clif_patient_assessments | `assessment_category == "braden_total"` AND `numerical_value <= 12` | First occurrence |

---

## Category 2: CLIF-Partial Labels

These are computable from CLIF data but with caveats:

### Readmission

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| 30-day readmission | `LABEL//readmit_30d` | clif_hospitalization | Same `patient_id`, new admission within 30d of discharge | discharge_dttm |
| 7-day readmission | `LABEL//readmit_7d` | clif_hospitalization | Same `patient_id`, new admission within 7d | discharge_dttm |

**Caveat**: Cannot distinguish planned vs unplanned readmissions without additional data.

### Abnormal Troponin (ECG-linked)

| Label | Token | Source | Condition | Timestamp |
|-------|-------|--------|-----------|-----------|
| Abnormal troponin 30d | `LABEL//abnormal_troponin_30d` | clif_labs + ECG CSV | Troponin above ULN within 30d of ECG | ECG time |

---

## Category 3: Requires MIMIC Raw/Derived Data

### From `mimiciv_derived` tables

| Label | Token | MIMIC Source | Description |
|-------|-------|-------------|-------------|
| Sepsis-3 | `LABEL//sepsis3` | `mimiciv_derived.sepsis3` | Suspected infection + SOFA ≥ 2 |
| AKI Stage 1 | `LABEL//aki_kdigo_1` | `mimiciv_derived.kdigo_stages` | KDIGO stage 1 |
| AKI Stage 2 | `LABEL//aki_kdigo_2` | `mimiciv_derived.kdigo_stages` | KDIGO stage 2 |
| AKI Stage 3 | `LABEL//aki_kdigo_3` | `mimiciv_derived.kdigo_stages` | KDIGO stage 3 |

### From `mimiciv_hosp` / `mimiciv_icu` tables

| Label | Token | MIMIC Source | Description |
|-------|-------|-------------|-------------|
| Cardiac arrest | `LABEL//cardiac_arrest` | procedureevents / diagnoses_icd | CPR procedure or cardiac arrest ICD code |
| Transfusion | `LABEL//transfusion` | inputevents | Blood product administration |
| Dialysis (any) | `LABEL//dialysis` | procedureevents | Hemodialysis or peritoneal dialysis |
| Central line | `LABEL//central_line` | procedureevents | Central venous catheter insertion |

### From clinical scoring

| Label | Token | MIMIC Source | Description |
|-------|-------|-------------|-------------|
| qSOFA ≥ 2 | `LABEL//qsofa_2` | chartevents + vitals | Altered mentation + RR≥22 + SBP≤100 (1h window) |

### Comorbidity Scores (Static, at admission)

| Label | Token | MIMIC Source | Description |
|-------|-------|-------------|-------------|
| Charlson ≥ 3 | `LABEL//charlson_high` | diagnoses_icd | Charlson Comorbidity Index ≥ 3 |
| Elixhauser ≥ 5 | `LABEL//elixhauser_high` | diagnoses_icd | Elixhauser count ≥ 5 |

### Disease Phenotypes (ICD-based, static)

| Label | Token | MIMIC Source | Description |
|-------|-------|-------------|-------------|
| CHF | `LABEL//phecode_chf` | diagnoses_icd | ICD-10 I50.* |
| COPD | `LABEL//phecode_copd` | diagnoses_icd | ICD-10 J44.* |
| Diabetes | `LABEL//phecode_diabetes` | diagnoses_icd | ICD-10 E10-E14.* |
| CKD | `LABEL//phecode_ckd` | diagnoses_icd | ICD-10 N18.* |
| Cancer | `LABEL//phecode_cancer` | diagnoses_icd | ICD-10 C00-C97.* |

---

## Label Implementation

### Registry Pattern (from reference repo)

```python
from dataclasses import dataclass
from typing import Callable
import polars as pl

_LABEL_REGISTRY: dict[str, "LabelDefinition"] = {}

@dataclass
class LabelDefinition:
    name: str
    token: str
    category: str  # "clif_only", "clif_partial", "mimic_required"
    description: str
    compute_fn: Callable
    source_tables: list[str]

def label(name: str, token: str, category: str, description: str, source_tables: list[str]):
    """Decorator to register a label computation function."""
    def decorator(fn):
        _LABEL_REGISTRY[name] = LabelDefinition(
            name=name, token=token, category=category,
            description=description, compute_fn=fn,
            source_tables=source_tables,
        )
        return fn
    return decorator

def get_registry() -> dict[str, LabelDefinition]:
    return _LABEL_REGISTRY.copy()
```

### Example Label Implementations

```python
@label(
    name="mortality",
    token="LABEL//mortality",
    category="clif_only",
    description="In-hospital mortality",
    source_tables=["clif_hospitalization"],
)
def compute_mortality(hosp_row: dict, **kwargs) -> dict | None:
    if hosp_row["discharge_category"] == "Expired":
        return {"code": "LABEL//mortality", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="vasopressor",
    token="LABEL//vasopressor",
    category="clif_only",
    description="Vasopressor initiation",
    source_tables=["clif_medication_admin_continuous"],
)
def compute_vasopressor(
    hosp_row: dict,
    meds_cont_df: pl.DataFrame,
    **kwargs,
) -> dict | None:
    vasoactives = meds_cont_df.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"]) &
        (pl.col("med_group") == "vasoactives") &
        (pl.col("mar_action_category") != "stop")
    ).sort("admin_dttm")

    if len(vasoactives) > 0:
        return {"code": "LABEL//vasopressor", "time": vasoactives[0, "admin_dttm"]}
    return None


@label(
    name="hyperkalemia",
    token="LABEL//hyperkalemia",
    category="clif_only",
    description="Potassium > 6.0 mEq/L",
    source_tables=["clif_labs"],
)
def compute_hyperkalemia(
    hosp_row: dict,
    labs_df: pl.DataFrame,
    **kwargs,
) -> dict | None:
    high_k = labs_df.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"]) &
        (pl.col("lab_category") == "potassium") &
        (pl.col("lab_value_numeric") > 6.0)
    ).sort("lab_result_dttm")

    if len(high_k) > 0:
        return {"code": "LABEL//hyperkalemia", "time": high_k[0, "lab_result_dttm"]}
    return None


@label(
    name="sepsis3",
    token="LABEL//sepsis3",
    category="mimic_required",
    description="Sepsis-3 onset (suspected infection + SOFA >= 2)",
    source_tables=["mimiciv_derived.sepsis3"],
)
async def compute_sepsis3(
    hosp_row: dict,
    db=None,
    **kwargs,
) -> dict | None:
    row = await db.fetch_one(
        "SELECT sepsis3_onset FROM mimiciv_derived.sepsis3 WHERE hadm_id = $1",
        int(hosp_row["hospitalization_id"]),
    )
    if row and row["sepsis3_onset"]:
        return {"code": "LABEL//sepsis3", "time": row["sepsis3_onset"]}
    return None
```

## Label Summary Table

### CLIF-Only (28 labels)

| # | Label | Token | Prevalence (est.) |
|---|-------|-------|--------------------|
| 1 | Hospital mortality | `LABEL//mortality` | ~10% |
| 2 | Mortality or hospice | `LABEL//mortality_or_hospice` | ~12% |
| 3-7 | Discharge disposition (5) | `LABEL//discharge_*` | varies |
| 8 | ICU admission | `LABEL//icu_admission` | ~35% |
| 9-10 | Prolonged ICU (7d, 14d) | `LABEL//prolonged_icu_*` | ~12%, ~5% |
| 11 | Stepdown admission | `LABEL//stepdown_admission` | ~15% |
| 12 | Mechanical ventilation | `LABEL//mech_vent` | ~20% |
| 13 | New intubation | `LABEL//new_intubation` | ~18% |
| 14 | NIPPV | `LABEL//nippv` | ~12% |
| 15 | HFNC | `LABEL//hfnc` | ~12% |
| 16 | Resp escalation | `LABEL//resp_escalation` | ~8% |
| 17 | High FiO2 | `LABEL//high_fio2` | ~10% |
| 18 | Hypoxemia | `LABEL//hypoxemia` | ~15% |
| 19 | Prone | `LABEL//prone` | ~3% |
| 20 | Reintubation 48h | `LABEL//reintubation_48h` | ~3% |
| 21 | Vasopressor | `LABEL//vasopressor` | ~18% |
| 22 | Hypotension | `LABEL//hypotension` | ~25% |
| 23 | Tachycardia | `LABEL//tachycardia` | ~30% |
| 24 | Bradycardia | `LABEL//bradycardia` | ~5% |
| 25 | ECMO | `LABEL//ecmo` | ~0.5% |
| 26 | CRRT | `LABEL//crrt` | ~4% |
| 27 | DNR transition | `LABEL//dnr` | ~8% |
| 28 | GCS < 8 | `LABEL//gcs_low` | ~8% |

### CLIF Lab-Based (12 labels)

| # | Label | Token | Threshold |
|---|-------|-------|-----------|
| 29 | Hyperkalemia | `LABEL//hyperkalemia` | K+ > 6.0 |
| 30 | Severe hyperkalemia | `LABEL//severe_hyperkalemia` | K+ > 6.5 |
| 31 | Hypoglycemia | `LABEL//hypoglycemia` | Glucose < 50 |
| 32 | Severe anemia | `LABEL//severe_anemia` | Hgb < 7 |
| 33 | Thrombocytopenia | `LABEL//thrombocytopenia` | Plt < 50K |
| 34 | Lactate elevated | `LABEL//lactate_elevated` | Lactate > 2.0 |
| 35 | Severe lactate | `LABEL//lactate_severe` | Lactate > 4.0 |
| 36 | Coagulopathy | `LABEL//coagulopathy` | INR > 2.0 |
| 37 | Acidosis | `LABEL//acidosis` | pH < 7.25 |
| 38 | Creatinine rise | `LABEL//creatinine_rise` | ≥1.5x baseline |
| 39 | Troponin elevated | `LABEL//troponin_elevated` | Above ULN |
| 40 | BNP elevated | `LABEL//bnp_elevated` | BNP > 300 |

### CLIF Assessment-Based (3 labels)

| # | Label | Token |
|---|-------|-------|
| 41 | Delirium | `LABEL//delirium` |
| 42 | Braden low | `LABEL//braden_low` |
| 43 | (reserved) | — |

### CLIF-Partial (3 labels)

| # | Label | Token |
|---|-------|-------|
| 44 | 30-day readmission | `LABEL//readmit_30d` |
| 45 | 7-day readmission | `LABEL//readmit_7d` |
| 46 | Abnormal troponin 30d | `LABEL//abnormal_troponin_30d` |

### MIMIC-Required (14 labels)

| # | Label | Token |
|---|-------|-------|
| 47 | Sepsis-3 | `LABEL//sepsis3` |
| 48-50 | AKI KDIGO 1/2/3 | `LABEL//aki_kdigo_*` |
| 51 | qSOFA ≥ 2 | `LABEL//qsofa_2` |
| 52 | Cardiac arrest | `LABEL//cardiac_arrest` |
| 53 | Transfusion | `LABEL//transfusion` |
| 54 | Dialysis | `LABEL//dialysis` |
| 55 | Central line | `LABEL//central_line` |
| 56 | Charlson ≥ 3 | `LABEL//charlson_high` |
| 57 | Elixhauser ≥ 5 | `LABEL//elixhauser_high` |
| 58-62 | Disease phenotypes (5) | `LABEL//phecode_*` |

**Total: ~62 labels** (43 CLIF-only, 3 partial, 3 assessment, 14 MIMIC)

## Output

```
data/processed/metadata/
├── label_definitions.json    # machine-readable label specs
├── label_registry.py         # label computation functions
└── label_prevalence.json     # actual prevalence in training data
```

## Dependencies

- Input: CLIF parquet files in `./data/`
- Input: MIMIC postgres DB (for Category 3 labels)
- Output: Feeds into Step 3 (Label Insertion)
