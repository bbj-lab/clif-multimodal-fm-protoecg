"""Lab-based critical value labels (12 labels)."""

from __future__ import annotations

import polars as pl

from clif_protoecg.labels.base import label


def _first_lab_above(
    hosp_row: dict, tables: dict, lab_cat: str, threshold: float, code: str
) -> dict | None:
    labs = tables.get("clif_labs", pl.DataFrame())
    if len(labs) == 0:
        return None
    hit = labs.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("lab_category") == lab_cat)
        & (pl.col("lab_value_numeric") > threshold)
    ).sort("lab_result_dttm")
    if len(hit) > 0:
        return {"code": code, "time": hit[0, "lab_result_dttm"]}
    return None


def _first_lab_below(
    hosp_row: dict, tables: dict, lab_cat: str, threshold: float, code: str
) -> dict | None:
    labs = tables.get("clif_labs", pl.DataFrame())
    if len(labs) == 0:
        return None
    hit = labs.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("lab_category") == lab_cat)
        & (pl.col("lab_value_numeric") < threshold)
    ).sort("lab_result_dttm")
    if len(hit) > 0:
        return {"code": code, "time": hit[0, "lab_result_dttm"]}
    return None


@label(
    name="hyperkalemia",
    token="LABEL//hyperkalemia",
    category="clif_only",
    description="Potassium > 6.0",
    source_tables=["clif_labs"],
)
def compute_hyperkalemia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_above(hosp_row, tables, "potassium", 6.0, "LABEL//hyperkalemia")


@label(
    name="severe_hyperkalemia",
    token="LABEL//severe_hyperkalemia",
    category="clif_only",
    description="Potassium > 6.5",
    source_tables=["clif_labs"],
)
def compute_severe_hyperkalemia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_above(
        hosp_row, tables, "potassium", 6.5, "LABEL//severe_hyperkalemia"
    )


@label(
    name="hypoglycemia",
    token="LABEL//hypoglycemia",
    category="clif_only",
    description="Glucose < 50",
    source_tables=["clif_labs"],
)
def compute_hypoglycemia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_below(hosp_row, tables, "glucose", 50, "LABEL//hypoglycemia")


@label(
    name="severe_anemia",
    token="LABEL//severe_anemia",
    category="clif_only",
    description="Hemoglobin < 7",
    source_tables=["clif_labs"],
)
def compute_severe_anemia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_below(hosp_row, tables, "hemoglobin", 7, "LABEL//severe_anemia")


@label(
    name="thrombocytopenia",
    token="LABEL//thrombocytopenia",
    category="clif_only",
    description="Platelets < 50000",
    source_tables=["clif_labs"],
)
def compute_thrombocytopenia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_below(
        hosp_row, tables, "platelet_count", 50000, "LABEL//thrombocytopenia"
    )


@label(
    name="lactate_elevated",
    token="LABEL//lactate_elevated",
    category="clif_only",
    description="Lactate > 2.0",
    source_tables=["clif_labs"],
)
def compute_lactate_elevated(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_above(hosp_row, tables, "lactate", 2.0, "LABEL//lactate_elevated")


@label(
    name="lactate_severe",
    token="LABEL//lactate_severe",
    category="clif_only",
    description="Lactate > 4.0",
    source_tables=["clif_labs"],
)
def compute_lactate_severe(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_above(hosp_row, tables, "lactate", 4.0, "LABEL//lactate_severe")


@label(
    name="coagulopathy",
    token="LABEL//coagulopathy",
    category="clif_only",
    description="INR > 2.0",
    source_tables=["clif_labs"],
)
def compute_coagulopathy(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_above(hosp_row, tables, "inr", 2.0, "LABEL//coagulopathy")


@label(
    name="acidosis",
    token="LABEL//acidosis",
    category="clif_only",
    description="pH < 7.25",
    source_tables=["clif_labs"],
)
def compute_acidosis(hosp_row: dict, tables: dict, **kw) -> dict | None:
    return _first_lab_below(hosp_row, tables, "ph_arterial", 7.25, "LABEL//acidosis")


@label(
    name="creatinine_rise",
    token="LABEL//creatinine_rise",
    category="clif_only",
    description="Creatinine >= 1.5x baseline",
    source_tables=["clif_labs"],
)
def compute_creatinine_rise(hosp_row: dict, tables: dict, **kw) -> dict | None:
    labs = tables.get("clif_labs", pl.DataFrame())
    if len(labs) == 0:
        return None
    creat = labs.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("lab_category") == "creatinine")
        & (pl.col("lab_value_numeric").is_not_null())
    ).sort("lab_result_dttm")
    if len(creat) < 2:
        return None
    baseline = creat[0, "lab_value_numeric"]
    if baseline <= 0:
        return None
    threshold = baseline * 1.5
    elevated = creat.filter(pl.col("lab_value_numeric") >= threshold)
    if len(elevated) > 0:
        return {
            "code": "LABEL//creatinine_rise",
            "time": elevated[0, "lab_result_dttm"],
        }
    return None


@label(
    name="troponin_elevated",
    token="LABEL//troponin_elevated",
    category="clif_only",
    description="Troponin above upper limit of normal",
    source_tables=["clif_labs"],
)
def compute_troponin_elevated(hosp_row: dict, tables: dict, **kw) -> dict | None:
    labs = tables.get("clif_labs", pl.DataFrame())
    if len(labs) == 0:
        return None
    # ULN for troponin_t ~0.04 ng/mL, troponin_i ~0.04 ng/mL
    trop = labs.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("lab_category").is_in(["troponin_t", "troponin_i"]))
        & (pl.col("lab_value_numeric") > 0.04)
    ).sort("lab_result_dttm")
    if len(trop) > 0:
        return {"code": "LABEL//troponin_elevated", "time": trop[0, "lab_result_dttm"]}
    return None
