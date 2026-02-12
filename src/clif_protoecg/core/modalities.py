"""Modality and ECG route enumerations."""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Clinical data modalities present in CLIF data."""

    # Core clinical
    DEMOGRAPHICS = "demographics"
    VITALS = "vitals"
    LABS = "labs"
    MEDICATIONS_CONTINUOUS = "medications_continuous"
    MEDICATIONS_INTERMITTENT = "medications_intermittent"
    RESPIRATORY = "respiratory"
    ASSESSMENTS = "assessments"
    ADT = "adt"
    CODE_STATUS = "code_status"
    POSITION = "position"
    CRRT = "crrt"
    ECMO = "ecmo"
    PROCEDURES = "procedures"
    DIAGNOSES = "diagnoses"
    DISCHARGE = "discharge"

    # ECG
    ECG = "ecg"

    # Temporal
    CLOCK = "clock"
    GAP = "gap"


class ECGRoute(str, Enum):
    """ECG ablation routes for training."""

    NO_ECG = "no_ecg"
    FUSION_CLASS = "fusion_class"
    ALL_BRANCHES = "all_branches"


# Modality groupings for convenience
CLINICAL_MODALITIES = frozenset(
    m for m in Modality if m not in (Modality.ECG, Modality.CLOCK, Modality.GAP)
)

TIME_MODALITIES = frozenset({Modality.CLOCK, Modality.GAP})
