"""Assessment-based labels (3 labels)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


def _get_assess(hosp_row: dict, tables: dict) -> pl.DataFrame:
    assess = tables.get("clif_patient_assessments", pl.DataFrame())
    if len(assess) == 0:
        return assess
    return assess.filter(
        pl.col("hospitalization_id") == hosp_row["hospitalization_id"]
    )


@label(name="gcs_low", token="LABEL//gcs_low", category="clif_only",
       description="GCS total < 8", source_tables=["clif_patient_assessments"])
def compute_gcs_low(hosp_row: dict, tables: dict, **kw) -> dict | None:
    assess = _get_assess(hosp_row, tables)
    if len(assess) == 0:
        return None
    low = assess.filter(
        (pl.col("assessment_category") == "gcs_total")
        & (pl.col("numerical_value") < 8)
    ).sort("recorded_dttm")
    if len(low) > 0:
        return {"code": "LABEL//gcs_low", "time": low[0, "recorded_dttm"]}
    return None


@label(name="delirium", token="LABEL//delirium", category="clif_only",
       description="CAM-ICU positive", source_tables=["clif_patient_assessments"])
def compute_delirium(hosp_row: dict, tables: dict, **kw) -> dict | None:
    assess = _get_assess(hosp_row, tables)
    if len(assess) == 0:
        return None
    cam = assess.filter(pl.col("assessment_category") == "cam_icu")
    if len(cam) == 0:
        return None
    # Positive result: categorical_value contains "positive" or numerical_value == 1
    positive = cam.filter(
        (pl.col("categorical_value").str.to_lowercase().str.contains("positive"))
        | (pl.col("numerical_value") == 1)
    ).sort("recorded_dttm")
    if len(positive) > 0:
        return {"code": "LABEL//delirium", "time": positive[0, "recorded_dttm"]}
    return None


@label(name="braden_low", token="LABEL//braden_low", category="clif_only",
       description="Braden score <= 12 (pressure injury risk)",
       source_tables=["clif_patient_assessments"])
def compute_braden_low(hosp_row: dict, tables: dict, **kw) -> dict | None:
    assess = _get_assess(hosp_row, tables)
    if len(assess) == 0:
        return None
    low = assess.filter(
        (pl.col("assessment_category") == "braden_total")
        & (pl.col("numerical_value") <= 12)
    ).sort("recorded_dttm")
    if len(low) > 0:
        return {"code": "LABEL//braden_low", "time": low[0, "recorded_dttm"]}
    return None
