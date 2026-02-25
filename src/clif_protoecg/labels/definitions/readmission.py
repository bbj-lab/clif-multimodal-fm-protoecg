"""Readmission and ECG-linked labels (3 labels, CLIF-partial)."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from clif_protoecg.labels.base import label


def _readmission(hosp_row: dict, tables: dict, days: int, code: str) -> dict | None:
    hosp = tables.get("clif_hospitalization", pl.DataFrame())
    if len(hosp) == 0:
        return None
    pid = hosp_row["patient_id"]
    discharge = hosp_row.get("discharge_dttm")
    if discharge is None:
        return None
    future = hosp.filter(
        (pl.col("patient_id") == pid)
        & (pl.col("admission_dttm") > discharge)
        & (pl.col("admission_dttm") <= discharge + timedelta(days=days))
    )
    if len(future) > 0:
        return {"code": code, "time": discharge}
    return None


@label(
    name="readmit_30d",
    token="LABEL//readmit_30d",
    category="clif_partial",
    description="30-day readmission",
    source_tables=["clif_hospitalization"],
)
def compute_readmit_30d(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _readmission(hosp_row, tables, 30, "LABEL//readmit_30d")


@label(
    name="readmit_7d",
    token="LABEL//readmit_7d",
    category="clif_partial",
    description="7-day readmission",
    source_tables=["clif_hospitalization"],
)
def compute_readmit_7d(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _readmission(hosp_row, tables, 7, "LABEL//readmit_7d")
