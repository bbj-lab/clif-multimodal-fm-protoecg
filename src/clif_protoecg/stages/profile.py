"""Code frequency profiling to filter rare codes."""

from __future__ import annotations

import polars as pl

from clif_protoecg.core.artifacts import CodeProfile
from clif_protoecg.core.constants import CLINICAL_WHITELIST, PROFILING_THRESHOLDS
from clif_protoecg.data.loader import CLIFDataLoader
from clif_protoecg.utils.logging import get_logger

logger = get_logger("profile")


def _count_codes(
    df: pl.DataFrame,
    code_col: str,
    hosp_col: str = "hospitalization_id",
    prefix: str = "",
) -> pl.DataFrame:
    """Count events and unique hospitalizations per code value."""
    return (
        df.group_by(code_col)
        .agg(
            pl.len().alias("event_count"),
            pl.col(hosp_col).n_unique().alias("admission_count"),
        )
        .with_columns(
            (pl.lit(prefix) + pl.col(code_col).cast(pl.Utf8)).alias("code")
        )
        .select("code", "event_count", "admission_count")
    )


def profile_codes(
    loader: CLIFDataLoader,
    train_hosp_ids: list[str],
    thresholds: dict[str, dict[str, int]] | None = None,
) -> CodeProfile:
    """Profile code frequencies on the training set and apply thresholds.

    Returns a CodeProfile with included_codes (passing thresholds or whitelisted).
    """
    if thresholds is None:
        thresholds = PROFILING_THRESHOLDS

    all_stats: list[pl.DataFrame] = []

    # Profile each modality
    _SOURCES = [
        ("vitals", "vital_category", "VITAL//"),
        ("labs", "lab_category", "LAB_RESULT//"),
        ("medication_admin_continuous", "med_category", "MED_CONT//"),
        ("medication_admin_intermittent", "med_category", "MED_BOLUS//"),
        ("patient_assessments", "assessment_category", "ASSESS//"),
        ("patient_procedures", "procedure_code", "PROC//"),
        ("hospital_diagnosis", "diagnosis_code", "ICD//"),
    ]

    for table_name, code_col, prefix in _SOURCES:
        try:
            df = loader.load_for_hospitalizations(table_name, train_hosp_ids)
        except FileNotFoundError:
            continue
        if len(df) > 0:
            stats = _count_codes(df, code_col, prefix=prefix)
            all_stats.append(stats)
            logger.info(
                f"  {prefix.rstrip('/'):20s} {len(stats):>5} codes, "
                f"{len(df):>9,} events"
            )

    if not all_stats:
        return CodeProfile()

    combined = pl.concat(all_stats)

    # Build code_stats dict and apply thresholds
    code_stats: dict[str, dict[str, int]] = {}
    included: set[str] = set()

    # Track per-modality include/exclude for reporting
    modality_stats: dict[str, dict[str, int]] = {}
    for _, _, prefix in _SOURCES:
        modality_stats[prefix] = {"included": 0, "excluded": 0, "whitelisted": 0}

    for row in combined.iter_rows(named=True):
        code = row["code"]
        ec = row["event_count"]
        ac = row["admission_count"]
        code_stats[code] = {"event_count": ec, "admission_count": ac}

        # Find which modality bucket this code belongs to
        modality_key = next(
            (pfx for _, _, pfx in _SOURCES if code.startswith(pfx)),
            None,
        )

        # Determine modality threshold key
        thresh = _get_threshold(code, thresholds)
        if code in CLINICAL_WHITELIST:
            included.add(code)
            if modality_key:
                modality_stats[modality_key]["whitelisted"] += 1
        elif thresh is not None:
            if ec >= thresh["min_events"] and ac >= thresh["min_admissions"]:
                included.add(code)
                if modality_key:
                    modality_stats[modality_key]["included"] += 1
            else:
                if modality_key:
                    modality_stats[modality_key]["excluded"] += 1

    # Log per-modality results
    for _, _, prefix in _SOURCES:
        s = modality_stats[prefix]
        total = s["included"] + s["excluded"] + s["whitelisted"]
        if total == 0:
            continue
        parts = [f'{s["included"]} included']
        if s["whitelisted"]:
            parts.append(f'{s["whitelisted"]} whitelisted')
        parts.append(f'{s["excluded"]} excluded')
        logger.info(f"  {prefix.rstrip('/'):20s} {'/'.join(parts)} (of {total})")

    return CodeProfile(code_stats=code_stats, included_codes=included)


def _get_threshold(
    code: str, thresholds: dict[str, dict[str, int]]
) -> dict[str, int] | None:
    """Map a code to its profiling threshold category."""
    if code.startswith("VITAL//"):
        return thresholds.get("vitals")
    if code.startswith(("LAB_RESULT//", "LAB_ORDER//")):
        return thresholds.get("labs")
    if code.startswith(("MED_CONT//", "MED_BOLUS//", "MED_CONT_STOP//")):
        return thresholds.get("medications")
    if code.startswith("ASSESS//"):
        return thresholds.get("assessments")
    if code.startswith("PROC//"):
        return thresholds.get("procedures")
    if code.startswith("ICD//"):
        return thresholds.get("diagnoses")
    return None
