"""CLIF -> MEDS event extraction: 16 event types."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from clif_protoecg.core.artifacts import BinEdges, CodeProfile
from clif_protoecg.core.constants import age_bucket, sort_priority
from clif_protoecg.data.loader import CLIFDataLoader


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hospitalization_sequence(
    hospitalization_id: str,
    patient_id: str,
    loader: CLIFDataLoader,
    code_profile: CodeProfile,
    bin_edges: BinEdges,
    hosp_row: dict | None = None,
    patient_row: dict | None = None,
) -> list[dict[str, Any]]:
    """Build the full event sequence for one hospitalization."""
    if hosp_row is None:
        hosp_df = loader.load("hospitalization")
        hosp_row = hosp_df.filter(
            pl.col("hospitalization_id") == hospitalization_id
        ).row(0, named=True)

    if patient_row is None:
        patient_df = loader.load("patient")
        patient_row = patient_df.filter(pl.col("patient_id") == patient_id).row(
            0, named=True
        )

    admit = hosp_row["admission_dttm"]
    discharge = hosp_row["discharge_dttm"]

    events: list[dict] = []

    # 1. Demographics
    events.extend(_extract_demographics(hosp_row, patient_row, admit))

    # 3-14. Clinical events
    events.extend(_extract_adt(loader, hospitalization_id))
    events.extend(_extract_vitals(loader, hospitalization_id, code_profile, bin_edges))
    events.extend(_extract_labs(loader, hospitalization_id, code_profile, bin_edges))
    events.extend(
        _extract_meds_continuous(loader, hospitalization_id, code_profile, bin_edges)
    )
    events.extend(
        _extract_meds_intermittent(loader, hospitalization_id, code_profile, bin_edges)
    )
    events.extend(_extract_respiratory(loader, hospitalization_id, bin_edges))
    events.extend(_extract_assessments(loader, hospitalization_id, code_profile))
    events.extend(
        _extract_code_status(loader, patient_id, hospitalization_id, admit, discharge)
    )
    events.extend(_extract_position(loader, hospitalization_id))
    events.extend(_extract_crrt(loader, hospitalization_id, bin_edges))
    events.extend(_extract_ecmo(loader, hospitalization_id, bin_edges))
    events.extend(_extract_procedures(loader, hospitalization_id, code_profile))

    # 15. Discharge
    events.extend(_extract_discharge(hosp_row))

    # 16. Retrospective billing codes
    events.extend(_extract_billing_codes(loader, hospitalization_id, discharge))

    # Sort: chronological, then by priority
    events.sort(key=lambda e: (e["time"], sort_priority(e["code"])))

    # Clip to admission window
    events = [e for e in events if e["time"] >= admit]

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    time: datetime, code: str, value: float | None = None, value_cat: str | None = None
) -> dict:
    return {"time": time, "code": code, "value": value, "value_cat": value_cat}


def _bin_token(code: str, value: float, bin_edges: BinEdges) -> dict | None:
    """Create a quantile bin event, or None if no edges for this code."""
    idx = bin_edges.get_bin(code, value)
    if idx is None:
        return None
    return _evt(None, f"Q_{idx}", value)  # time set by caller


def _code_included(code: str, profile: CodeProfile) -> bool:
    return code in profile.included_codes or not profile.included_codes


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _extract_demographics(hosp: dict, patient: dict, admit: datetime) -> list[dict]:
    events = []
    age = hosp.get("age_at_admission")
    if age is not None:
        events.append(_evt(admit, f"DEMO//AGE_{age_bucket(int(age))}"))

    sex = patient.get("sex_category")
    if sex:
        events.append(_evt(admit, f"DEMO//SEX_{sex}"))

    race = patient.get("race_category")
    if race:
        events.append(_evt(admit, f"DEMO//RACE_{race}"))

    return events


def _extract_adt(loader: CLIFDataLoader, hosp_id: str) -> list[dict]:
    rows = loader.load_for_hospitalizations("adt", [hosp_id]).sort("in_dttm")
    return [
        _evt(row["in_dttm"], f"ADT//{row['location_category']}")
        for row in rows.iter_rows(named=True)
    ]


def _extract_vitals(
    loader: CLIFDataLoader,
    hosp_id: str,
    profile: CodeProfile,
    bin_edges: BinEdges,
) -> list[dict]:
    vitals = loader.load_for_hospitalizations("vitals", [hosp_id])
    events = []
    for row in vitals.iter_rows(named=True):
        code = f"VITAL//{row['vital_category']}"
        if not _code_included(code, profile):
            continue
        t = row["recorded_dttm"]
        events.append(_evt(t, code, row["vital_value"]))
        bt = _bin_token(code, row["vital_value"], bin_edges)
        if bt:
            bt["time"] = t
            events.append(bt)
    return events


def _extract_labs(
    loader: CLIFDataLoader,
    hosp_id: str,
    profile: CodeProfile,
    bin_edges: BinEdges,
) -> list[dict]:
    labs = loader.load_for_hospitalizations("labs", [hosp_id])
    events = []
    for row in labs.iter_rows(named=True):
        cat = row["lab_category"]
        order_code = f"LAB_ORDER//{cat}"
        result_code = f"LAB_RESULT//{cat}"
        if not _code_included(result_code, profile):
            continue

        # Order event
        order_t = row.get("lab_order_dttm") or row.get("lab_collect_dttm")
        if order_t is not None:
            events.append(_evt(order_t, order_code))

        # Result event
        result_t = row.get("lab_result_dttm")
        val = row.get("lab_value_numeric")
        if result_t is not None and val is not None:
            events.append(_evt(result_t, result_code, val))
            bt = _bin_token(result_code, val, bin_edges)
            if bt:
                bt["time"] = result_t
                events.append(bt)
        elif row.get("lab_collect_dttm") is not None and val is not None:
            # Fallback: result at collect time
            ct = row["lab_collect_dttm"]
            events.append(_evt(ct, result_code, val))
            bt = _bin_token(result_code, val, bin_edges)
            if bt:
                bt["time"] = ct
                events.append(bt)
    return events


def _extract_meds_continuous(
    loader: CLIFDataLoader,
    hosp_id: str,
    profile: CodeProfile,
    bin_edges: BinEdges,
) -> list[dict]:
    meds = loader.load_for_hospitalizations("medication_admin_continuous", [hosp_id])
    events = []
    for row in meds.iter_rows(named=True):
        cat = row["med_category"]
        action = row.get("mar_action_category", "")
        dose = row.get("med_dose", 0) or 0
        t = row["admin_dttm"]

        if action in ("stop", "paused", "held") or dose == 0:
            code = f"MED_CONT_STOP//{cat}"
            events.append(_evt(t, code))
        else:
            code = f"MED_CONT//{cat}"
            if not _code_included(code, profile):
                continue
            events.append(_evt(t, code, dose))
            bt = _bin_token(code, dose, bin_edges)
            if bt:
                bt["time"] = t
                events.append(bt)
    return events


def _extract_meds_intermittent(
    loader: CLIFDataLoader,
    hosp_id: str,
    profile: CodeProfile,
    bin_edges: BinEdges,
) -> list[dict]:
    meds = loader.load_for_hospitalizations("medication_admin_intermittent", [hosp_id])
    events = []
    for row in meds.iter_rows(named=True):
        cat = row["med_category"]
        code = f"MED_BOLUS//{cat}"
        if not _code_included(code, profile):
            continue
        t = row["admin_dttm"]
        dose = row.get("med_dose")
        events.append(_evt(t, code, dose))
        if dose is not None:
            bt = _bin_token(code, dose, bin_edges)
            if bt:
                bt["time"] = t
                events.append(bt)
    return events


def _extract_respiratory(
    loader: CLIFDataLoader,
    hosp_id: str,
    bin_edges: BinEdges,
) -> list[dict]:
    resp = loader.load_for_hospitalizations("respiratory_support", [hosp_id])
    events = []
    for row in resp.iter_rows(named=True):
        t = row["recorded_dttm"]
        device = row.get("device_category", "unknown")
        events.append(_evt(t, f"RESP//{device}"))

        # Parameter tokens
        for param_col, param_name in [
            ("fio2_set", "fio2"),
            ("peep_set", "peep"),
            ("tidal_volume_set", "tidal_volume"),
            ("resp_rate_set", "resp_rate_set"),
            ("pressure_support_set", "pressure_support"),
        ]:
            val = row.get(param_col)
            if val is not None:
                code = f"RESP_PARAM//{param_name}"
                events.append(_evt(t, code, val))
                bt = _bin_token(code, val, bin_edges)
                if bt:
                    bt["time"] = t
                    events.append(bt)
    return events


def _extract_assessments(
    loader: CLIFDataLoader,
    hosp_id: str,
    profile: CodeProfile,
) -> list[dict]:
    assess = loader.load_for_hospitalizations("patient_assessments", [hosp_id])
    events = []
    for row in assess.iter_rows(named=True):
        cat = row["assessment_category"]
        code = f"ASSESS//{cat}"
        if not _code_included(code, profile):
            continue
        t = row["recorded_dttm"]
        val = row.get("numerical_value")
        val_cat = row.get("categorical_value")
        events.append(_evt(t, code, val, val_cat))
    return events


def _extract_code_status(
    loader: CLIFDataLoader,
    patient_id: str,
    hosp_id: str,
    admit: datetime,
    discharge: datetime,
) -> list[dict]:
    cs = loader.load("code_status")
    # Filter by hospitalization_id if available, else by patient_id + time window
    if "hospitalization_id" in cs.columns:
        rows = cs.filter(pl.col("hospitalization_id") == hosp_id)
    else:
        rows = cs.filter(
            (pl.col("patient_id") == patient_id)
            & (pl.col("start_dttm") >= admit)
            & (pl.col("start_dttm") <= discharge)
        )
    return [
        _evt(row["start_dttm"], f"CODE_STATUS//{row['code_status_category']}")
        for row in rows.iter_rows(named=True)
    ]


def _extract_position(loader: CLIFDataLoader, hosp_id: str) -> list[dict]:
    rows = loader.load_for_hospitalizations("position", [hosp_id])
    return [
        _evt(row["recorded_dttm"], f"POSITION//{row['position_category']}")
        for row in rows.iter_rows(named=True)
    ]


def _extract_crrt(
    loader: CLIFDataLoader, hosp_id: str, bin_edges: BinEdges
) -> list[dict]:
    rows = loader.load_for_hospitalizations("crrt_therapy", [hosp_id])
    events = []
    for row in rows.iter_rows(named=True):
        t = row["recorded_dttm"]
        mode = row.get("crrt_mode_category", "unknown")
        events.append(_evt(t, f"CRRT//{mode}"))
        for param_col, param_name in [
            ("blood_flow_rate", "blood_flow"),
            ("ultrafiltration_out", "uf_out"),
        ]:
            val = row.get(param_col)
            if val is not None:
                code = f"CRRT_PARAM//{param_name}"
                events.append(_evt(t, code, val))
                bt = _bin_token(code, val, bin_edges)
                if bt:
                    bt["time"] = t
                    events.append(bt)
    return events


def _extract_ecmo(
    loader: CLIFDataLoader, hosp_id: str, bin_edges: BinEdges
) -> list[dict]:
    rows = loader.load_for_hospitalizations("ecmo_mcs", [hosp_id])
    events = []
    for row in rows.iter_rows(named=True):
        t = row["recorded_dttm"]
        device = row.get("device_category", "ECMO")
        events.append(_evt(t, f"ECMO//{device}"))
        flow = row.get("flow")
        if flow is not None:
            code = "ECMO_PARAM//flow"
            events.append(_evt(t, code, flow))
            bt = _bin_token(code, flow, bin_edges)
            if bt:
                bt["time"] = t
                events.append(bt)
    return events


def _extract_procedures(
    loader: CLIFDataLoader, hosp_id: str, profile: CodeProfile
) -> list[dict]:
    rows = loader.load_for_hospitalizations("patient_procedures", [hosp_id])
    events = []
    for row in rows.iter_rows(named=True):
        code = f"PROC//{row['procedure_code']}"
        if not _code_included(code, profile):
            continue
        t = row.get("procedure_billed_dttm")
        if t is not None:
            events.append(_evt(t, code))
    return events


def _extract_discharge(hosp: dict) -> list[dict]:
    discharge = hosp.get("discharge_dttm")
    if discharge is None:
        return []
    events = [_evt(discharge, "DISCH//GENERAL")]
    cat = hosp.get("discharge_category")
    if cat:
        events.append(_evt(discharge, f"DISCH//{cat}"))
    return events


def _extract_billing_codes(
    loader: CLIFDataLoader, hosp_id: str, discharge: datetime | None
) -> list[dict]:
    if discharge is None:
        return []
    billing_time = discharge + timedelta(minutes=1)
    rows = loader.load_for_hospitalizations("hospital_diagnosis", [hosp_id])
    return [
        _evt(billing_time, f"ICD//{row['diagnosis_code']}")
        for row in rows.iter_rows(named=True)
    ]
