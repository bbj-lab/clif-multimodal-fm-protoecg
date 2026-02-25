"""Mortality and discharge disposition labels (7 labels)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


@label(
    name="mortality",
    token="LABEL//mortality",
    category="clif_only",
    description="In-hospital mortality",
    source_tables=["clif_hospitalization"],
)
def compute_mortality(hosp_row: dict, **kw) -> dict | None:
    if hosp_row.get("discharge_category") == "Expired":
        return {"code": "LABEL//mortality", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="mortality_or_hospice",
    token="LABEL//mortality_or_hospice",
    category="clif_only",
    description="In-hospital mortality or discharge to hospice",
    source_tables=["clif_hospitalization"],
)
def compute_mortality_or_hospice(hosp_row: dict, **kw) -> dict | None:
    if hosp_row.get("discharge_category") in ("Expired", "Hospice"):
        return {
            "code": "LABEL//mortality_or_hospice",
            "time": hosp_row["discharge_dttm"],
        }
    return None


@label(
    name="discharge_home",
    token="LABEL//discharge_home",
    category="clif_only",
    description="Discharge to home",
    source_tables=["clif_hospitalization"],
)
def compute_discharge_home(hosp_row: dict, **kw) -> dict | None:
    if hosp_row.get("discharge_category") == "Home":
        return {"code": "LABEL//discharge_home", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="discharge_snf",
    token="LABEL//discharge_snf",
    category="clif_only",
    description="Discharge to skilled nursing facility",
    source_tables=["clif_hospitalization"],
)
def compute_discharge_snf(hosp_row: dict, **kw) -> dict | None:
    cat = hosp_row.get("discharge_category") or ""
    if "Skilled Nursing" in cat:
        return {"code": "LABEL//discharge_snf", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="discharge_ltach",
    token="LABEL//discharge_ltach",
    category="clif_only",
    description="Discharge to LTACH",
    source_tables=["clif_hospitalization"],
)
def compute_discharge_ltach(hosp_row: dict, **kw) -> dict | None:
    cat = hosp_row.get("discharge_category") or ""
    if "LTACH" in cat:
        return {"code": "LABEL//discharge_ltach", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="discharge_hospice",
    token="LABEL//discharge_hospice",
    category="clif_only",
    description="Discharge to hospice",
    source_tables=["clif_hospitalization"],
)
def compute_discharge_hospice(hosp_row: dict, **kw) -> dict | None:
    if hosp_row.get("discharge_category") == "Hospice":
        return {"code": "LABEL//discharge_hospice", "time": hosp_row["discharge_dttm"]}
    return None


@label(
    name="discharge_acute",
    token="LABEL//discharge_acute",
    category="clif_only",
    description="Discharge to acute care hospital",
    source_tables=["clif_hospitalization"],
)
def compute_discharge_acute(hosp_row: dict, **kw) -> dict | None:
    if hosp_row.get("discharge_category") == "Acute Care Hospital":
        return {"code": "LABEL//discharge_acute", "time": hosp_row["discharge_dttm"]}
    return None
