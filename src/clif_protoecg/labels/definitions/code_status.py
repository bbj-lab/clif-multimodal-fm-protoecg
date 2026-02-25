"""Code status transition label (1 label)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


@label(
    name="dnr",
    token="LABEL//dnr",
    category="clif_only",
    description="Transition to DNR/DNAR/CMO",
    source_tables=["clif_code_status"],
)
def compute_dnr(hosp_row: dict, tables: dict, **kw) -> dict | None:
    cs = tables.get("clif_code_status", pl.DataFrame())
    if len(cs) == 0:
        return None
    hid = hosp_row["hospitalization_id"]
    pid = hosp_row["patient_id"]

    # Filter by hospitalization_id if available, else patient_id
    if "hospitalization_id" in cs.columns:
        rows = cs.filter(pl.col("hospitalization_id") == hid).sort("start_dttm")
    else:
        admit = hosp_row.get("admission_dttm")
        discharge = hosp_row.get("discharge_dttm")
        rows = cs.filter(
            (pl.col("patient_id") == pid)
            & (pl.col("start_dttm") >= admit)
            & (pl.col("start_dttm") <= discharge)
        ).sort("start_dttm")

    if len(rows) < 1:
        return None

    # Look for transition from Full to DNR/DNAR/CMO
    dnr_categories = {"DNR", "DNAR", "UDNR", "DNR/DNI", "DNAR/DNI", "AND"}
    for row in rows.iter_rows(named=True):
        cat = row.get("code_status_category", "")
        if cat in dnr_categories:
            return {"code": "LABEL//dnr", "time": row["start_dttm"]}
    return None
