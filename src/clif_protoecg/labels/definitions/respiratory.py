"""Respiratory labels (9 labels)."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from clif_protoecg.labels.base import label
from clif_protoecg.core.constants import RESP_SEVERITY


def _get_resp(hosp_row: dict, tables: dict) -> pl.DataFrame:
    resp = tables.get("clif_respiratory_support", pl.DataFrame())
    if len(resp) == 0:
        return resp
    return resp.filter(
        pl.col("hospitalization_id") == hosp_row["hospitalization_id"]
    ).sort("recorded_dttm")


@label(
    name="mech_vent",
    token="LABEL//mech_vent_trach",
    category="clif_only",
    description="New intubation (tracheostomy)",
    source_tables=["clif_respiratory_support"],
)
def compute_mech_vent(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    imv = resp.filter(
        (pl.col("device_category") == "IMV") & (pl.col("tracheostomy") == 1)
    )
    if len(imv) > 0:
        return {"code": "LABEL//mech_vent", "time": imv[0, "recorded_dttm"]}
    return None


@label(
    name="new_intubation",
    token="LABEL//mech_vent_no_trach",
    category="clif_only",
    description="New intubation (non-tracheostomy)",
    source_tables=["clif_respiratory_support"],
)
def compute_new_intubation(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    imv = resp.filter(
        (pl.col("device_category") == "IMV") & (pl.col("tracheostomy") == 0)
    )
    if len(imv) > 0:
        return {"code": "LABEL//new_intubation", "time": imv[0, "recorded_dttm"]}
    return None


@label(
    name="nippv",
    token="LABEL//nippv",
    category="clif_only",
    description="Non-invasive positive pressure ventilation",
    source_tables=["clif_respiratory_support"],
)
def compute_nippv(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    nippv = resp.filter(pl.col("device_category") == "NIPPV")
    if len(nippv) > 0:
        return {"code": "LABEL//nippv", "time": nippv[0, "recorded_dttm"]}
    return None


@label(
    name="hfnc",
    token="LABEL//hfnc",
    category="clif_only",
    description="High flow nasal cannula",
    source_tables=["clif_respiratory_support"],
)
def compute_hfnc(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    hf = resp.filter(pl.col("device_category") == "High Flow NC")
    if len(hf) > 0:
        return {"code": "LABEL//hfnc", "time": hf[0, "recorded_dttm"]}
    return None


@label(
    name="resp_escalation",
    token="LABEL//resp_escalation",
    category="clif_only",
    description="Rapid respiratory escalation (>=2 severity levels in 24h)",
    source_tables=["clif_respiratory_support"],
)
def compute_resp_escalation(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    if len(resp) < 2:
        return None
    rows = resp.select("recorded_dttm", "device_category").to_dicts()
    for i in range(1, len(rows)):
        sev_prev = RESP_SEVERITY.get(rows[i - 1]["device_category"], 0)
        sev_curr = RESP_SEVERITY.get(rows[i]["device_category"], 0)
        dt = rows[i]["recorded_dttm"] - rows[i - 1]["recorded_dttm"]
        if sev_curr - sev_prev >= 2 and dt <= timedelta(hours=24):
            return {"code": "LABEL//resp_escalation", "time": rows[i]["recorded_dttm"]}
    return None


@label(
    name="high_fio2",
    token="LABEL//high_fio2",
    category="clif_only",
    description="FiO2 > 0.6",
    source_tables=["clif_respiratory_support"],
)
def compute_high_fio2(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    if "fio2_set" not in resp.columns:
        return None
    high = resp.filter(pl.col("fio2_set") > 0.6)
    if len(high) > 0:
        return {"code": "LABEL//high_fio2", "time": high[0, "recorded_dttm"]}
    return None


@label(
    name="hypoxemia",
    token="LABEL//hypoxemia",
    category="clif_only",
    description="SpO2 < 88%",
    source_tables=["clif_vitals"],
)
def compute_hypoxemia(hosp_row: dict, tables: dict, **kw) -> dict | None:
    vitals = tables.get("clif_vitals", pl.DataFrame())
    if len(vitals) == 0:
        return None
    low_spo2 = vitals.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("vital_category") == "spo2")
        & (pl.col("vital_value") < 88)
    ).sort("recorded_dttm")
    if len(low_spo2) > 0:
        return {"code": "LABEL//hypoxemia", "time": low_spo2[0, "recorded_dttm"]}
    return None


@label(
    name="prone",
    token="LABEL//prone",
    category="clif_only",
    description="Prone positioning",
    source_tables=["clif_position"],
)
def compute_prone(hosp_row: dict, tables: dict, **kw) -> dict | None:
    pos = tables.get("clif_position", pl.DataFrame())
    if len(pos) == 0:
        return None
    prone = pos.filter(
        (pl.col("hospitalization_id") == hosp_row["hospitalization_id"])
        & (pl.col("position_category") == "prone")
    ).sort("recorded_dttm")
    if len(prone) > 0:
        return {"code": "LABEL//prone", "time": prone[0, "recorded_dttm"]}
    return None


@label(
    name="reintubation_48h",
    token="LABEL//reintubation_48h",
    category="clif_only",
    description="Reintubation within 48h of extubation",
    source_tables=["clif_respiratory_support"],
)
def compute_reintubation(hosp_row: dict, tables: dict, **kw) -> dict | None:
    resp = _get_resp(hosp_row, tables)
    if len(resp) < 3:
        return None
    rows = resp.select("recorded_dttm", "device_category").to_dicts()
    # Find IMV -> non-IMV -> IMV pattern within 48h
    last_imv_end = None
    for i, r in enumerate(rows):
        if r["device_category"] == "IMV":
            if last_imv_end is not None:
                gap = r["recorded_dttm"] - last_imv_end
                if gap <= timedelta(hours=48):
                    return {
                        "code": "LABEL//reintubation_48h",
                        "time": r["recorded_dttm"],
                    }
            last_imv_end = None
        else:
            # Transition from IMV to non-IMV
            if i > 0 and rows[i - 1]["device_category"] == "IMV":
                last_imv_end = r["recorded_dttm"]
    return None
