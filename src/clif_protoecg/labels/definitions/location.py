"""ICU and location-based labels (4 labels)."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from clif_protoecg.labels.base import label


def _get_adt(hosp_row: dict, tables: dict) -> pl.DataFrame:
    adt = tables.get("clif_adt", pl.DataFrame())
    if len(adt) == 0:
        return adt
    return adt.filter(
        pl.col("hospitalization_id") == hosp_row["hospitalization_id"]
    ).sort("in_dttm")


@label(
    name="icu_admission",
    token="LABEL//icu_admission",
    category="clif_only",
    description="ICU admission (not if admitted directly to ICU)",
    source_tables=["clif_adt"],
)
def compute_icu_admission(hosp_row: dict, tables: dict, **kw) -> dict | None:
    adt = _get_adt(hosp_row, tables)
    if len(adt) == 0:
        return None
    first_loc = adt[0, "location_category"]
    if first_loc == "icu":
        return None  # Already in ICU at admission
    icu = adt.filter(pl.col("location_category") == "icu")
    if len(icu) > 0:
        return {"code": "LABEL//icu_admission", "time": icu[0, "in_dttm"]}
    return None


@label(
    name="prolonged_icu_7d",
    token="LABEL//prolonged_icu_7d",
    category="clif_only",
    description="Prolonged ICU stay >= 7 days",
    source_tables=["clif_adt"],
)
def compute_prolonged_icu_7d(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _prolonged_icu(hosp_row, tables, hours=168)


@label(
    name="prolonged_icu_14d",
    token="LABEL//prolonged_icu_14d",
    category="clif_only",
    description="Prolonged ICU stay >= 14 days",
    source_tables=["clif_adt"],
)
def compute_prolonged_icu_14d(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _prolonged_icu(hosp_row, tables, hours=336)


def _prolonged_icu(hosp_row: dict, tables: dict, hours: int) -> dict | None:
    adt = _get_adt(hosp_row, tables)
    if len(adt) == 0:
        return None
    icu = adt.filter(pl.col("location_category") == "icu")
    if len(icu) == 0:
        return None
    total = timedelta()
    for row in icu.iter_rows(named=True):
        in_t = row["in_dttm"]
        out_t = row.get("out_dttm") or hosp_row.get("discharge_dttm", in_t)
        total += out_t - in_t
        if total >= timedelta(hours=hours):
            threshold_time = in_t + (timedelta(hours=hours) - (total - (out_t - in_t)))
            return {
                "code": f"LABEL//prolonged_icu_{hours // 24}d",
                "time": threshold_time,
            }
    return None


@label(
    name="stepdown_admission",
    token="LABEL//stepdown_admission",
    category="clif_only",
    description="Stepdown unit admission",
    source_tables=["clif_adt"],
)
def compute_stepdown(hosp_row: dict, tables: dict, **kw) -> dict | None:
    adt = _get_adt(hosp_row, tables)
    if len(adt) == 0:
        return None
    sd = adt.filter(pl.col("location_category") == "stepdown")
    if len(sd) > 0:
        return {"code": "LABEL//stepdown_admission", "time": sd[0, "in_dttm"]}
    return None
