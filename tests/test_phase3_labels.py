"""Tests for Phase 3: Label registry, definitions, evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

# Force registration of all labels
import clif_protoecg.labels.definitions  # noqa: F401

from clif_protoecg.labels.base import get_registry, LabelDefinition
from clif_protoecg.labels.evaluator import LabelEvaluator
from clif_protoecg.data.loader import CLIFDataLoader


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestLabelRegistry:
    def test_registry_not_empty(self):
        reg = get_registry()
        assert len(reg) > 0

    def test_all_labels_have_required_fields(self):
        for name, defn in get_registry().items():
            assert isinstance(defn, LabelDefinition)
            assert defn.name == name
            assert defn.token.startswith("LABEL//")
            assert defn.category in ("clif_only", "clif_partial", "mimic_required")
            assert callable(defn.compute_fn)

    def test_clif_only_count(self):
        reg = get_registry()
        clif_only = [d for d in reg.values() if d.category == "clif_only"]
        # Should have mortality(7) + location(4) + respiratory(9) +
        # cardiovascular(5) + renal(1) + lab_critical(12) + assessments(3) +
        # code_status(1) = 42
        assert len(clif_only) >= 40

    def test_mimic_required_are_stubs(self):
        reg = get_registry()
        mimic = [d for d in reg.values() if d.category == "mimic_required"]
        assert len(mimic) >= 14
        # All should return None
        for defn in mimic:
            result = defn.compute_fn(hosp_row={})
            assert result is None

    def test_unique_tokens(self):
        reg = get_registry()
        tokens = [d.token for d in reg.values()]
        assert len(tokens) == len(set(tokens)), "Duplicate label tokens found"


# ---------------------------------------------------------------------------
# Individual label computations
# ---------------------------------------------------------------------------


class TestMortalityLabels:
    def test_mortality_expired(self):
        from clif_protoecg.labels.definitions.mortality import compute_mortality
        t = datetime(2020, 3, 18, 14, 0)
        result = compute_mortality(hosp_row={
            "discharge_category": "Expired", "discharge_dttm": t
        })
        assert result is not None
        assert result["code"] == "LABEL//mortality"
        assert result["time"] == t

    def test_mortality_alive(self):
        from clif_protoecg.labels.definitions.mortality import compute_mortality
        result = compute_mortality(hosp_row={
            "discharge_category": "Home", "discharge_dttm": datetime.now()
        })
        assert result is None

    def test_discharge_home(self):
        from clif_protoecg.labels.definitions.mortality import compute_discharge_home
        t = datetime(2020, 3, 18, 14, 0)
        result = compute_discharge_home(hosp_row={
            "discharge_category": "Home", "discharge_dttm": t
        })
        assert result is not None
        assert result["code"] == "LABEL//discharge_home"


class TestLocationLabels:
    def test_icu_admission_from_ward(self):
        from clif_protoecg.labels.definitions.location import compute_icu_admission
        admit = datetime(2020, 1, 1)
        icu_time = admit + timedelta(hours=6)
        tables = {
            "clif_adt": pl.DataFrame({
                "hospitalization_id": ["H1", "H1"],
                "in_dttm": [admit, icu_time],
                "out_dttm": [icu_time, admit + timedelta(days=1)],
                "location_category": ["ward", "icu"],
            })
        }
        result = compute_icu_admission(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None
        assert result["time"] == icu_time

    def test_no_icu_if_admitted_to_icu(self):
        from clif_protoecg.labels.definitions.location import compute_icu_admission
        admit = datetime(2020, 1, 1)
        tables = {
            "clif_adt": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "in_dttm": [admit],
                "out_dttm": [admit + timedelta(days=1)],
                "location_category": ["icu"],
            })
        }
        result = compute_icu_admission(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is None


class TestRespiratoryLabels:
    def test_mech_vent(self):
        from clif_protoecg.labels.definitions.respiratory import compute_mech_vent
        t = datetime(2020, 1, 1, 10, 0)
        tables = {
            "clif_respiratory_support": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "recorded_dttm": [t],
                "device_category": ["IMV"],
            })
        }
        result = compute_mech_vent(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None
        assert result["code"] == "LABEL//mech_vent"

    def test_high_fio2(self):
        from clif_protoecg.labels.definitions.respiratory import compute_high_fio2
        t = datetime(2020, 1, 1, 10, 0)
        tables = {
            "clif_respiratory_support": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "recorded_dttm": [t],
                "device_category": ["IMV"],
                "fio2_set": [0.8],
            })
        }
        result = compute_high_fio2(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None


class TestCardiovascularLabels:
    def test_vasopressor(self):
        from clif_protoecg.labels.definitions.cardiovascular import compute_vasopressor
        t = datetime(2020, 1, 1, 10, 0)
        tables = {
            "clif_medication_admin_continuous": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "admin_dttm": [t],
                "med_group": ["vasoactives"],
                "mar_action_category": ["new"],
            })
        }
        result = compute_vasopressor(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None
        assert result["code"] == "LABEL//vasopressor"


class TestLabCriticalLabels:
    def test_hyperkalemia(self):
        from clif_protoecg.labels.definitions.lab_critical import compute_hyperkalemia
        t = datetime(2020, 1, 1, 10, 0)
        tables = {
            "clif_labs": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "lab_category": ["potassium"],
                "lab_value_numeric": [6.5],
                "lab_result_dttm": [t],
            })
        }
        result = compute_hyperkalemia(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None
        assert result["code"] == "LABEL//hyperkalemia"

    def test_no_hyperkalemia_normal(self):
        from clif_protoecg.labels.definitions.lab_critical import compute_hyperkalemia
        t = datetime(2020, 1, 1, 10, 0)
        tables = {
            "clif_labs": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "lab_category": ["potassium"],
                "lab_value_numeric": [4.0],
                "lab_result_dttm": [t],
            })
        }
        result = compute_hyperkalemia(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is None

    def test_creatinine_rise(self):
        from clif_protoecg.labels.definitions.lab_critical import compute_creatinine_rise
        t1 = datetime(2020, 1, 1, 10, 0)
        t2 = datetime(2020, 1, 2, 10, 0)
        tables = {
            "clif_labs": pl.DataFrame({
                "hospitalization_id": ["H1", "H1"],
                "lab_category": ["creatinine", "creatinine"],
                "lab_value_numeric": [1.0, 2.0],
                "lab_result_dttm": [t1, t2],
            })
        }
        result = compute_creatinine_rise(
            hosp_row={"hospitalization_id": "H1"}, tables=tables
        )
        assert result is not None
        assert result["time"] == t2


class TestCodeStatusLabels:
    def test_dnr_transition(self):
        from clif_protoecg.labels.definitions.code_status import compute_dnr
        t = datetime(2020, 1, 2, 10, 0)
        tables = {
            "clif_code_status": pl.DataFrame({
                "hospitalization_id": ["H1"],
                "patient_id": ["P1"],
                "start_dttm": [t],
                "code_status_category": ["DNR"],
            })
        }
        result = compute_dnr(
            hosp_row={
                "hospitalization_id": "H1",
                "patient_id": "P1",
                "admission_dttm": datetime(2020, 1, 1),
                "discharge_dttm": datetime(2020, 1, 5),
            },
            tables=tables,
        )
        assert result is not None
        assert result["code"] == "LABEL//dnr"


class TestReadmissionLabels:
    def test_readmit_30d(self):
        from clif_protoecg.labels.definitions.readmission import compute_readmit_30d
        d1 = datetime(2020, 1, 10)
        a2 = datetime(2020, 1, 25)
        tables = {
            "clif_hospitalization": pl.DataFrame({
                "patient_id": ["P1", "P1"],
                "hospitalization_id": ["H1", "H2"],
                "admission_dttm": [datetime(2020, 1, 1), a2],
                "discharge_dttm": [d1, datetime(2020, 2, 1)],
            })
        }
        result = compute_readmit_30d(
            hosp_row={"patient_id": "P1", "discharge_dttm": d1,
                       "hospitalization_id": "H1"},
            tables=tables,
        )
        assert result is not None
        assert result["code"] == "LABEL//readmit_30d"

    def test_no_readmit_if_too_far(self):
        from clif_protoecg.labels.definitions.readmission import compute_readmit_30d
        d1 = datetime(2020, 1, 10)
        a2 = datetime(2020, 3, 1)  # > 30 days
        tables = {
            "clif_hospitalization": pl.DataFrame({
                "patient_id": ["P1", "P1"],
                "hospitalization_id": ["H1", "H2"],
                "admission_dttm": [datetime(2020, 1, 1), a2],
                "discharge_dttm": [d1, datetime(2020, 3, 10)],
            })
        }
        result = compute_readmit_30d(
            hosp_row={"patient_id": "P1", "discharge_dttm": d1,
                       "hospitalization_id": "H1"},
            tables=tables,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestLabelEvaluator:
    def test_evaluate_hospitalization(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only", "clif_partial"])

        results = evaluator.evaluate_batch(["H003"])
        labels = results["H003"]
        # H003 is expired patient with ICU, IMV, CRRT, DNR, prone
        codes = [l["code"] for l in labels]
        assert "LABEL//mortality" in codes
        assert "LABEL//crrt" in codes
        assert "LABEL//dnr" in codes

    def test_home_discharge_no_mortality(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only"])

        results = evaluator.evaluate_batch(["H001"])
        labels = results["H001"]
        codes = [l["code"] for l in labels]
        assert "LABEL//mortality" not in codes
        assert "LABEL//discharge_home" in codes

    def test_icu_admission_label(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only"])

        results = evaluator.evaluate_batch(["H001"])
        labels = results["H001"]
        codes = [l["code"] for l in labels]
        # H001: ward -> icu, should have ICU admission label
        assert "LABEL//icu_admission" in codes

    def test_label_names_property(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only"])
        names = evaluator.label_names
        assert "mortality" in names
        assert "vasopressor" in names

    def test_mimic_labels_excluded_by_default(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader)
        names = evaluator.label_names
        assert "sepsis3" not in names

    def test_gcs_low_for_h003(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only"])
        results = evaluator.evaluate_batch(["H003"])
        codes = [l["code"] for l in results["H003"]]
        # H003 has GCS=6 which is < 8
        assert "LABEL//gcs_low" in codes

    def test_all_labels_have_time(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        evaluator = LabelEvaluator(loader, categories=["clif_only", "clif_partial"])
        results = evaluator.evaluate_batch(["H001", "H002", "H003"])
        for hid, labels in results.items():
            for label in labels:
                assert "time" in label, f"Label {label['code']} for {hid} missing time"
                assert "code" in label
                assert label["time"] is not None
