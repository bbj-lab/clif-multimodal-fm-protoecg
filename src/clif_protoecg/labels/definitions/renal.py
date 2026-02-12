"""Renal labels (2 labels)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


@label(name="crrt", token="LABEL//crrt", category="clif_only",
       description="CRRT initiation", source_tables=["clif_crrt_therapy"])
def compute_crrt(hosp_row: dict, tables: dict, **kw) -> dict | None:
    crrt = tables.get("clif_crrt_therapy", pl.DataFrame())
    if len(crrt) == 0:
        return None
    rows = crrt.filter(
        pl.col("hospitalization_id") == hosp_row["hospitalization_id"]
    ).sort("recorded_dttm")
    if len(rows) > 0:
        return {"code": "LABEL//crrt", "time": rows[0, "recorded_dttm"]}
    return None
