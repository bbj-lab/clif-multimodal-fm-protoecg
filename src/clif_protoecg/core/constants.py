"""Pipeline constants: clinical bins, sort priorities, age buckets, profiling thresholds."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Age Buckets
# ---------------------------------------------------------------------------
AGE_BUCKETS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def age_bucket(age: int) -> str:
    """Map numeric age to a decade bucket string."""
    if age < 30:
        return "18-29"
    if age >= 80:
        return "80+"
    decade = (age // 10) * 10
    return f"{decade}-{decade + 9}"


# ---------------------------------------------------------------------------
# Clinical Bin Boundaries (hybrid mode)
# ---------------------------------------------------------------------------
CLINICAL_BINS: dict[str, list[float]] = {
    # Vitals
    "VITAL//hr": [0, 50, 60, 80, 100, 120, 150, 200],
    "VITAL//sbp": [0, 70, 90, 110, 130, 150, 180, 250],
    "VITAL//dbp": [0, 40, 50, 60, 70, 80, 100, 140],
    "VITAL//mbp": [0, 50, 60, 65, 75, 90, 110, 150],
    "VITAL//resp_rate": [0, 8, 12, 16, 20, 25, 30, 40],
    "VITAL//spo2": [0, 70, 80, 88, 92, 95, 98, 100],
    "VITAL//temp_c": [33, 35, 36, 36.5, 37.5, 38, 39, 42],
    # Critical labs
    "LAB_RESULT//potassium": [0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 10],
    "LAB_RESULT//sodium": [0, 120, 125, 130, 135, 140, 145, 150, 155, 180],
    "LAB_RESULT//glucose": [0, 40, 60, 70, 100, 140, 200, 300, 500, 1000],
    "LAB_RESULT//creatinine": [0, 0.4, 0.7, 1.0, 1.3, 1.8, 2.5, 4.0, 8.0, 20],
    "LAB_RESULT//lactate": [0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10, 20],
    "LAB_RESULT//hemoglobin": [0, 5, 7, 8, 10, 12, 14, 16, 20],
    "LAB_RESULT//platelets": [0, 20, 50, 100, 150, 250, 400, 600, 1000],
    "LAB_RESULT//wbc": [0, 1, 4, 7, 11, 15, 20, 30, 50],
    "LAB_RESULT//troponin_t": [0, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0, 5.0, 50],
    "LAB_RESULT//inr": [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10],
    "LAB_RESULT//ph": [6.8, 7.0, 7.15, 7.25, 7.35, 7.45, 7.55, 7.65],
    "LAB_RESULT//pco2": [0, 20, 30, 35, 40, 45, 50, 60, 80, 120],
    "LAB_RESULT//bicarbonate": [0, 10, 15, 18, 22, 26, 30, 35, 50],
}

# ---------------------------------------------------------------------------
# Sort Priority (lower = earlier at same timestamp)
# ---------------------------------------------------------------------------
SORT_PRIORITY_MAP: dict[str, float] = {
    "DEMO//": 0,
    "ADMIT//": 0,
    "ICD_POA//": 1,
    "CLOCK//": 1,
    "ADT//": 2,
    "VITAL//": 2,
    "LAB_ORDER//": 2,
    "LAB_RESULT//": 2,
    "MED_CONT//": 2,
    "MED_CONT_STOP//": 2,
    "MED_BOLUS//": 2,
    "RESP//": 2,
    "RESP_PARAM//": 2,
    "ASSESS//": 2,
    "CODE_STATUS//": 2,
    "POSITION//": 2,
    "CRRT//": 2,
    "CRRT_PARAM//": 2,
    "ECMO//": 2,
    "ECMO_PARAM//": 2,
    "PROC//": 2,
    "DT//": 2,
    "ECG//": 2.5,
    "LABEL//": 3,
    "DISCH//": 4,
    "ICD//": 5,
    "DRG//": 5,
}


_sort_priority_cache: dict[str, float] = {}


def sort_priority(code: str) -> float:
    """Return sort priority for an event code (lower = earlier)."""
    # Extract prefix (everything up to and including "//")
    sep = code.find("//")
    prefix_key = code[: sep + 2] if sep != -1 else code

    cached = _sort_priority_cache.get(prefix_key)
    if cached is not None:
        return cached

    for prefix, priority in SORT_PRIORITY_MAP.items():
        if prefix_key == prefix:
            _sort_priority_cache[prefix_key] = priority
            return priority

    _sort_priority_cache[prefix_key] = 2.0
    return 2.0


# ---------------------------------------------------------------------------
# Code Profiling Thresholds
# ---------------------------------------------------------------------------
PROFILING_THRESHOLDS: dict[str, dict[str, int]] = {
    "labs": {"min_events": 1000, "min_admissions": 100},
    "vitals": {"min_events": 500, "min_admissions": 50},
    "medications": {"min_events": 200, "min_admissions": 50},
    "assessments": {"min_events": 200, "min_admissions": 50},
    "procedures": {"min_events": 100, "min_admissions": 20},
    "diagnoses": {"min_events": 50, "min_admissions": 10},
}

# Codes that bypass frequency filtering
CLINICAL_WHITELIST: frozenset[str] = frozenset({
    "LAB_RESULT//troponin_t",
    "LAB_RESULT//troponin_i",
    "LAB_RESULT//bnp",
    "LAB_RESULT//procalcitonin",
    "LAB_RESULT//lactate",
    "LAB_RESULT//d_dimer",
    "RESP//IMV",
    "CRRT//cvvh",
    "ECMO//ECMO",
    "CODE_STATUS//DNR",
    "POSITION//prone",
})

# ---------------------------------------------------------------------------
# Time Token Constants
# ---------------------------------------------------------------------------
CLOCK_TIMES = [0, 4, 8, 12, 16, 20]
CLOCK_INTERVAL_HOURS = 4

TIME_BUCKET_EDGES = [1, 5, 15, 30, 60, 120, 240, 480]

# Horizon mapping: hours -> clock token count
HORIZON_TO_CLOCK_TOKENS = {8: 2, 24: 6, 48: 12}

# ---------------------------------------------------------------------------
# Special Token Strings
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
CONT_TOKEN = "[CONT]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, CONT_TOKEN]

# Label token prefixes for loss weighting
LABEL_TOKEN_PREFIXES = ("LABEL//", "DISCH//", "ICD//", "PROC//", "DRG//")

# Respiratory device severity for escalation detection
RESP_SEVERITY: dict[str, int] = {
    "Room Air": 0,
    "Nasal Cannula": 1,
    "Face Mask": 1,
    "High Flow NC": 2,
    "CPAP": 2,
    "NIPPV": 3,
    "IMV": 4,
}
