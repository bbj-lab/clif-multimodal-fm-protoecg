"""Shared fixtures: mock CLIF data for all tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with minimal CLIF parquet files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Base timestamps
    admit1 = datetime(2020, 3, 15, 8, 30)
    disch1 = datetime(2020, 3, 18, 14, 0)
    admit2 = datetime(2020, 6, 1, 12, 0)
    disch2 = datetime(2020, 6, 5, 10, 0)
    admit3 = datetime(2021, 1, 10, 6, 0)
    disch3 = datetime(2021, 1, 12, 18, 0)

    # -- clif_patient --
    pl.DataFrame({
        "patient_id": ["P001", "P002", "P003"],
        "race_category": ["White", "Black", "Asian"],
        "sex_category": ["Male", "Female", "Male"],
        "birth_date": [
            datetime(1955, 1, 1),
            datetime(1980, 6, 15),
            datetime(1942, 11, 20),
        ],
        "death_dttm": [None, None, disch3],
    }).write_parquet(data_dir / "clif_patient.parquet")

    # -- clif_hospitalization --
    pl.DataFrame({
        "patient_id": ["P001", "P002", "P003"],
        "hospitalization_id": ["H001", "H002", "H003"],
        "admission_dttm": [admit1, admit2, admit3],
        "discharge_dttm": [disch1, disch2, disch3],
        "age_at_admission": [65, 40, 78],
        "discharge_category": ["Home", "Home", "Expired"],
    }).write_parquet(data_dir / "clif_hospitalization.parquet")

    # -- clif_adt --
    pl.DataFrame({
        "hospitalization_id": ["H001", "H001", "H002", "H003", "H003"],
        "patient_id": ["P001", "P001", "P002", "P003", "P003"],
        "in_dttm": [
            admit1,
            admit1 + timedelta(hours=12),
            admit2,
            admit3,
            admit3 + timedelta(hours=6),
        ],
        "out_dttm": [
            admit1 + timedelta(hours=12),
            disch1,
            disch2,
            admit3 + timedelta(hours=6),
            disch3,
        ],
        "location_category": ["ward", "icu", "ward", "ed", "icu"],
    }).write_parquet(data_dir / "clif_adt.parquet")

    # -- clif_vitals --
    vitals_rows = []
    for hid, base in [("H001", admit1), ("H002", admit2), ("H003", admit3)]:
        for i in range(5):
            t = base + timedelta(hours=i * 4)
            vitals_rows.append({
                "hospitalization_id": hid,
                "recorded_dttm": t,
                "vital_category": "hr",
                "vital_value": 70.0 + i * 5,
            })
            vitals_rows.append({
                "hospitalization_id": hid,
                "recorded_dttm": t,
                "vital_category": "sbp",
                "vital_value": 120.0 + i * 3,
            })
    pl.DataFrame(vitals_rows).write_parquet(data_dir / "clif_vitals.parquet")

    # -- clif_labs --
    lab_rows = []
    for hid, base in [("H001", admit1), ("H002", admit2), ("H003", admit3)]:
        for i in range(3):
            order_t = base + timedelta(hours=i * 8)
            result_t = order_t + timedelta(hours=2)
            lab_rows.append({
                "hospitalization_id": hid,
                "lab_order_dttm": order_t,
                "lab_collect_dttm": order_t + timedelta(minutes=30),
                "lab_result_dttm": result_t,
                "lab_category": "potassium",
                "lab_value_numeric": 4.0 + i * 0.5,
            })
            lab_rows.append({
                "hospitalization_id": hid,
                "lab_order_dttm": order_t,
                "lab_collect_dttm": order_t + timedelta(minutes=30),
                "lab_result_dttm": result_t,
                "lab_category": "glucose",
                "lab_value_numeric": 100.0 + i * 20,
            })
    pl.DataFrame(lab_rows).write_parquet(data_dir / "clif_labs.parquet")

    # -- clif_medication_admin_continuous --
    pl.DataFrame({
        "hospitalization_id": ["H001", "H003"],
        "admin_dttm": [
            admit1 + timedelta(hours=6),
            admit3 + timedelta(hours=3),
        ],
        "med_category": ["norepinephrine", "norepinephrine"],
        "med_group": ["vasoactives", "vasoactives"],
        "med_dose": [0.1, 0.2],
        "med_dose_unit": ["mcg/kg/min", "mcg/kg/min"],
        "mar_action_category": ["new", "new"],
    }).write_parquet(data_dir / "clif_medication_admin_continuous.parquet")

    # -- clif_medication_admin_intermittent --
    pl.DataFrame({
        "hospitalization_id": ["H002"],
        "admin_dttm": [admit2 + timedelta(hours=2)],
        "med_category": ["acetaminophen"],
        "med_group": ["analgesics"],
        "med_dose": [1000.0],
        "med_dose_unit": ["mg"],
        "mar_action_category": ["given"],
    }).write_parquet(data_dir / "clif_medication_admin_intermittent.parquet")

    # -- clif_respiratory_support --
    pl.DataFrame({
        "hospitalization_id": ["H001", "H003"],
        "recorded_dttm": [
            admit1 + timedelta(hours=14),
            admit3 + timedelta(hours=8),
        ],
        "device_category": ["Nasal Cannula", "IMV"],
        "mode_category": [None, "AC/VC"],
        "fio2_set": [0.4, 0.8],
        "peep_set": [None, 10.0],
        "tidal_volume_set": [None, 450.0],
        "resp_rate_set": [None, 14.0],
        "pressure_support_set": [None, None],
    }).write_parquet(data_dir / "clif_respiratory_support.parquet")

    # -- clif_patient_assessments --
    pl.DataFrame({
        "hospitalization_id": ["H001", "H003"],
        "recorded_dttm": [
            admit1 + timedelta(hours=4),
            admit3 + timedelta(hours=2),
        ],
        "assessment_category": ["gcs_total", "gcs_total"],
        "numerical_value": [15.0, 6.0],
        "categorical_value": [None, None],
    }).write_parquet(data_dir / "clif_patient_assessments.parquet")

    # -- clif_code_status --
    pl.DataFrame({
        "patient_id": ["P003"],
        "hospitalization_id": ["H003"],
        "start_dttm": [admit3 + timedelta(hours=24)],
        "code_status_category": ["DNR"],
    }).write_parquet(data_dir / "clif_code_status.parquet")

    # -- clif_position --
    pl.DataFrame({
        "hospitalization_id": ["H003"],
        "recorded_dttm": [admit3 + timedelta(hours=10)],
        "position_category": ["prone"],
    }).write_parquet(data_dir / "clif_position.parquet")

    # -- clif_hospital_diagnosis --
    pl.DataFrame({
        "hospitalization_id": ["H001", "H001", "H002", "H003"],
        "diagnosis_code": ["I50.9", "E11.9", "J44.1", "I21.0"],
        "diagnosis_code_format": ["ICD-10", "ICD-10", "ICD-10", "ICD-10"],
        "diagnosis_primary": [1, 0, 1, 1],
        "poa_present": [1, 1, 0, 1],
    }).write_parquet(data_dir / "clif_hospital_diagnosis.parquet")

    # -- clif_patient_procedures --
    pl.DataFrame({
        "hospitalization_id": ["H001"],
        "procedure_code": ["0BH17EZ"],
        "procedure_code_format": ["ICD-10-PCS"],
        "procedure_billed_dttm": [disch1],
    }).write_parquet(data_dir / "clif_patient_procedures.parquet")

    # -- clif_crrt_therapy --
    pl.DataFrame({
        "hospitalization_id": ["H003"],
        "recorded_dttm": [admit3 + timedelta(hours=18)],
        "crrt_mode_category": ["cvvh"],
        "blood_flow_rate": [200.0],
        "ultrafiltration_out": [150.0],
    }).write_parquet(data_dir / "clif_crrt_therapy.parquet")

    # -- clif_ecmo_mcs --
    pl.DataFrame({
        "hospitalization_id": pl.Series([], dtype=pl.Utf8),
        "recorded_dttm": pl.Series([], dtype=pl.Datetime),
        "device_category": pl.Series([], dtype=pl.Utf8),
        "mcs_group": pl.Series([], dtype=pl.Utf8),
        "flow": pl.Series([], dtype=pl.Float64),
    }).write_parquet(data_dir / "clif_ecmo_mcs.parquet")

    return data_dir


@pytest.fixture()
def sample_events() -> list[dict]:
    """A small pre-built event list for testing insertion stages."""
    base = datetime(2020, 3, 15, 8, 30)
    return [
        {"time": base, "code": "DEMO//AGE_60-69", "value": None, "value_cat": None},
        {"time": base, "code": "DEMO//SEX_Male", "value": None, "value_cat": None},
        {"time": base + timedelta(hours=1), "code": "VITAL//hr", "value": 72.0, "value_cat": None},
        {"time": base + timedelta(hours=1), "code": "VITAL//sbp", "value": 130.0, "value_cat": None},
        {"time": base + timedelta(hours=4), "code": "LAB_RESULT//potassium", "value": 4.2, "value_cat": None},
        {"time": base + timedelta(hours=8), "code": "VITAL//hr", "value": 85.0, "value_cat": None},
        {"time": base + timedelta(days=3, hours=5, minutes=30), "code": "DISCH//GENERAL", "value": None, "value_cat": None},
        {"time": base + timedelta(days=3, hours=5, minutes=30), "code": "DISCH//Home", "value": None, "value_cat": None},
    ]
