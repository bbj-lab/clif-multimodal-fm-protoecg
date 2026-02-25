"""Cardiovascular labels (5 labels)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


@label(
    name="vasopressor",
    token="LABEL//vasopressor",
    category="clif_only",
    description="Vasopressor initiation",
    source_tables=["clif_medication_admin_continuous"],
)
def compute_vasopressor(hosp_row: dict, tables: dict, **kw) -> dict | None:
    meds = tables.get("clif_medication_admin_continuous", pl.DataFrame())
    if len(meds) == 0:
        return None
    vaso = meds.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (
            pl.col("med_category").is_in(
                [
                    "angiotensin",
                    "dopamine",
                    "epinephrine",
                    "norepinephrine",
                    "phenylephrine",
                    "vasopressin",
                ]
            )
        )
        & (~pl.col("mar_action_category").is_in(["stop", "other"]))
    ).sort("admin_dttm")
    if len(vaso) > 0:
        return {"code": "LABEL//vasopressor", "time": vaso[0, "admin_dttm"]}
    return None


@label(
    name="hypotension",
    token="LABEL//hypotension",
    category="clif_only",
    description="MAP < 65 mmHg",
    source_tables=["clif_vitals"],
)
def compute_hypotension(hosp_row: dict, tables: dict, **kw) -> dict | None:
    vitals = tables.get("clif_vitals", pl.DataFrame())
    if len(vitals) == 0:
        return None
    low = vitals.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("vital_category") == "map")
        & (pl.col("vital_value") < 65)
    ).sort("recorded_dttm")
    if len(low) > 0:
        return {"code": "LABEL//hypotension", "time": low[0, "recorded_dttm"]}
    return None


@label(
    name="tachycardia",
    token="LABEL//tachycardia",
    category="clif_only",
    description="Heart rate > 120",
    source_tables=["clif_vitals"],
)
def compute_tachycardia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    vitals = tables.get("clif_vitals", pl.DataFrame())
    if len(vitals) == 0:
        return None
    tachy = vitals.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("vital_category") == "heart_rate")
        & (pl.col("vital_value") > 120)
    ).sort("recorded_dttm")
    if len(tachy) > 0:
        return {"code": "LABEL//tachycardia", "time": tachy[0, "recorded_dttm"]}
    return None


@label(
    name="bradycardia",
    token="LABEL//bradycardia",
    category="clif_only",
    description="Heart rate < 50",
    source_tables=["clif_vitals"],
)
def compute_bradycardia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    vitals = tables.get("clif_vitals", pl.DataFrame())
    if len(vitals) == 0:
        return None
    brady = vitals.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("vital_category") == "heart_rate")
        & (pl.col("vital_value") < 50)
    ).sort("recorded_dttm")
    if len(brady) > 0:
        return {"code": "LABEL//bradycardia", "time": brady[0, "recorded_dttm"]}
    return None


@label(
    name="ecmo",
    token="LABEL//ecmo",
    category="clif_only",
    description="ECMO initiation",
    source_tables=["clif_ecmo_mcs"],
)
def compute_ecmo(hosp_row: dict, tables: dict, **kw) -> dict | None:
    ecmo = tables.get("clif_ecmo_mcs", pl.DataFrame())
    if len(ecmo) == 0:
        return None
    rows = ecmo.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
    ).sort("recorded_dttm")
    if len(rows) > 0:
        return {"code": "LABEL//ecmo", "time": rows[0, "recorded_dttm"]}
    return None
