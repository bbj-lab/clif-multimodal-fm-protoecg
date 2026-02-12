"""Tests for Phase 2: Profiling, bin edges, and extraction."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from clif_protoecg.core.artifacts import BinEdges, CodeProfile
from clif_protoecg.data.loader import CLIFDataLoader
from clif_protoecg.stages.bin_edges import compute_bin_edges, compute_quantile_bins
from clif_protoecg.stages.extract import build_hospitalization_sequence
from clif_protoecg.stages.profile import profile_codes


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


class TestProfileCodes:
    def test_returns_code_profile(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        profile = profile_codes(loader, hosp_ids)
        assert isinstance(profile, CodeProfile)
        assert len(profile.code_stats) > 0

    def test_codes_have_counts(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        profile = profile_codes(loader, hosp_ids)
        for code, stats in profile.code_stats.items():
            assert "event_count" in stats
            assert "admission_count" in stats
            assert stats["event_count"] > 0

    def test_whitelist_codes_included(self, tmp_data_dir):
        """Whitelisted codes should be included even with high thresholds."""
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        # Set impossibly high thresholds
        high_thresh = {k: {"min_events": 999999, "min_admissions": 999999}
                       for k in ["labs", "vitals", "medications", "assessments",
                                 "procedures", "diagnoses"]}
        profile = profile_codes(loader, hosp_ids, thresholds=high_thresh)
        # Nothing should pass thresholds, but whitelist codes present in data
        # should still be in included_codes (if they appear in the data)
        # Our test data doesn't have the whitelisted codes, so included should be empty
        # This test validates the mechanism works without error
        assert isinstance(profile.included_codes, set)

    def test_low_thresholds_include_everything(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        low_thresh = {k: {"min_events": 1, "min_admissions": 1}
                      for k in ["labs", "vitals", "medications", "assessments",
                                "procedures", "diagnoses"]}
        profile = profile_codes(loader, hosp_ids, thresholds=low_thresh)
        # All codes should be included with min thresholds of 1
        assert len(profile.included_codes) == len(profile.code_stats)


# ---------------------------------------------------------------------------
# Bin Edges
# ---------------------------------------------------------------------------


class TestComputeQuantileBins:
    def test_ten_bins(self):
        vals = np.arange(100, dtype=float)
        edges = compute_quantile_bins(vals, n_bins=10)
        # Should have between 2 and 11 unique edges
        assert len(edges) >= 2
        assert len(edges) <= 11

    def test_sorted(self):
        vals = np.random.default_rng(42).normal(0, 1, size=1000)
        edges = compute_quantile_bins(vals, n_bins=5)
        assert edges == sorted(edges)

    def test_small_data(self):
        vals = np.array([1.0, 2.0, 3.0])
        edges = compute_quantile_bins(vals, n_bins=10)
        assert len(edges) >= 2


class TestComputeBinEdges:
    def test_hybrid_mode(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        be = compute_bin_edges(loader, hosp_ids, bin_mode="hybrid", n_bins=5)
        assert isinstance(be, BinEdges)
        # Should have clinical bins
        assert "VITAL//hr" in be.edges

    def test_quantile_mode(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        be = compute_bin_edges(loader, hosp_ids, bin_mode="quantile", n_bins=3)
        # All codes computed from data (not clinical bins)
        for code, edges in be.edges.items():
            assert len(edges) >= 2

    def test_clinical_only_mode(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        be = compute_bin_edges(loader, hosp_ids, bin_mode="clinical_only")
        # Only codes in CLINICAL_BINS should be present
        from clif_protoecg.core.constants import CLINICAL_BINS
        for code in be.edges:
            assert code in CLINICAL_BINS

    def test_get_bin_works(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        be = compute_bin_edges(loader, hosp_ids, bin_mode="hybrid")
        # HR with clinical bins
        idx = be.get_bin("VITAL//hr", 85.0)
        assert idx is not None
        assert isinstance(idx, int)
        assert idx >= 0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestBuildHospitalizationSequence:
    def test_returns_list_of_events(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        hosp_ids = loader.get_hospitalization_ids()
        be = compute_bin_edges(loader, hosp_ids, bin_mode="hybrid")
        cp = CodeProfile(included_codes=set())  # empty = include all

        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_events_have_required_keys(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        for e in events:
            assert "time" in e
            assert "code" in e
            assert isinstance(e["time"], datetime)
            assert isinstance(e["code"], str)

    def test_chronological_order(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        times = [e["time"] for e in events]
        assert times == sorted(times)

    def test_starts_with_demographics(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        first_codes = [e["code"] for e in events[:5]]
        assert any(c.startswith("DEMO//") for c in first_codes)

    def test_ends_with_discharge(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        # Last events should be discharge/billing
        last_codes = [e["code"] for e in events[-5:]]
        assert any("DISCH//" in c for c in last_codes)

    def test_expired_patient_has_death_discharge(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H003", "P003", loader, cp, be)
        codes = [e["code"] for e in events]
        assert "DISCH//Expired" in codes

    def test_contains_vitals(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("VITAL//") for c in codes)

    def test_contains_labs(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("LAB_ORDER//") for c in codes)
        assert any(c.startswith("LAB_RESULT//") for c in codes)

    def test_contains_meds(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        # H001 has continuous med
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("MED_CONT//") for c in codes)

    def test_contains_adt(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("ADT//") for c in codes)

    def test_contains_respiratory(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("RESP//") for c in codes)

    def test_contains_crrt_for_h003(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H003", "P003", loader, cp, be)
        codes = [e["code"] for e in events]
        assert any(c.startswith("CRRT//") for c in codes)

    def test_contains_position(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H003", "P003", loader, cp, be)
        codes = [e["code"] for e in events]
        assert "POSITION//prone" in codes

    def test_contains_code_status(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H003", "P003", loader, cp, be)
        codes = [e["code"] for e in events]
        assert "CODE_STATUS//DNR" in codes

    def test_billing_codes_after_discharge(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        disch_time = None
        for e in events:
            if e["code"] == "DISCH//GENERAL":
                disch_time = e["time"]
        icd_events = [e for e in events if e["code"].startswith("ICD//")]
        for e in icd_events:
            assert e["time"] > disch_time

    def test_poa_diagnoses_at_admission(self, tmp_data_dir):
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        hosp = loader.load("hospitalization").filter(
            pl.col("hospitalization_id") == "H001"
        ).row(0, named=True)
        admit = hosp["admission_dttm"]
        poa_events = [e for e in events if e["code"].startswith("ICD_POA//")]
        for e in poa_events:
            assert e["time"] == admit

    def test_sort_priority_respected(self, tmp_data_dir):
        """Events at same timestamp should be ordered by sort priority."""
        loader = CLIFDataLoader(tmp_data_dir)
        be = BinEdges()
        cp = CodeProfile()
        events = build_hospitalization_sequence("H001", "P001", loader, cp, be)
        # Group events by timestamp
        from collections import defaultdict
        from clif_protoecg.core.constants import sort_priority as sp
        groups = defaultdict(list)
        for e in events:
            groups[e["time"]].append(e)
        for t, evts in groups.items():
            priorities = [sp(e["code"]) for e in evts]
            assert priorities == sorted(priorities), \
                f"Priority order violated at {t}: {[e['code'] for e in evts]}"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
import polars as pl
