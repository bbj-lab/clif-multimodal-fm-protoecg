"""Tests for Phase 4: Label insertion, ECG prototypes, time tokens."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from clif_protoecg.core.constants import sort_priority
from clif_protoecg.core.modalities import ECGRoute
from clif_protoecg.stages.label_insertion import insert_labels
from clif_protoecg.stages.ecg_prototypes import (
    ECGPrototypeTokenizer,
    inject_ecg_prototypes,
    filter_ecgs_to_window,
)
from clif_protoecg.stages.time_tokens import (
    insert_clock_tokens,
    insert_gap_tokens,
    add_temporal_tokens,
    _get_gap_token,
)


# ---------------------------------------------------------------------------
# Label Insertion
# ---------------------------------------------------------------------------


class TestLabelInsertion:
    def test_labels_inserted(self, sample_events):
        labels = [
            {"code": "LABEL//mortality", "time": sample_events[-1]["time"]},
        ]
        result = insert_labels(sample_events, labels)
        codes = [e["code"] for e in result]
        assert "LABEL//mortality" in codes

    def test_label_after_clinical_at_same_time(self, sample_events):
        t = sample_events[2]["time"]  # vitals time
        labels = [{"code": "LABEL//tachycardia", "time": t}]
        result = insert_labels(sample_events, labels)
        # Find the label in the result
        for i, e in enumerate(result):
            if e["code"] == "LABEL//tachycardia":
                # All events before it at the same time should be clinical
                for j in range(i):
                    if result[j]["time"] == t:
                        assert sort_priority(result[j]["code"]) <= sort_priority("LABEL//tachycardia")
                break

    def test_dedup_labels(self, sample_events):
        t = sample_events[2]["time"]
        labels = [
            {"code": "LABEL//test", "time": t},
            {"code": "LABEL//test", "time": t + timedelta(hours=1)},
        ]
        result = insert_labels(sample_events, labels)
        label_count = sum(1 for e in result if e["code"] == "LABEL//test")
        assert label_count == 1

    def test_no_labels_unchanged(self, sample_events):
        result = insert_labels(sample_events, [])
        assert len(result) == len(sample_events)

    def test_chronological_order_preserved(self, sample_events):
        t = sample_events[4]["time"]
        labels = [{"code": "LABEL//hyperkalemia", "time": t}]
        result = insert_labels(sample_events, labels)
        times = [e["time"] for e in result]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# ECG Prototypes
# ---------------------------------------------------------------------------


class TestECGPrototypeTokenizer:
    def _make_row(self):
        return {
            "ecg_time": datetime(2020, 3, 15, 12, 0),
            "top1class": "AFIB",
            "top1score": 0.95,
            "1d_prototype_num": 7,
            "1d_similarity": 0.82,
            "2d_partial_prototype_num": 23,
            "2d_partial_similarity": 0.71,
            "2d_global_prototype_num": 1,
            "2d_global_similarity": 0.90,
        }

    def test_no_ecg_returns_empty(self):
        tok = ECGPrototypeTokenizer(route=ECGRoute.NO_ECG)
        assert tok.tokenize_ecg(self._make_row()) == []

    def test_fusion_class_returns_3_tokens(self):
        tok = ECGPrototypeTokenizer(route=ECGRoute.FUSION_CLASS)
        events = tok.tokenize_ecg(self._make_row())
        assert len(events) == 3
        codes = [e["code"] for e in events]
        assert codes[0] == "ECG//Class/AFIB"
        assert codes[1] == "ECG//Prototype/1D/7"
        assert codes[2] == "ECG//Similarity/1D"

    def test_all_branches_returns_7_tokens(self):
        tok = ECGPrototypeTokenizer(route=ECGRoute.ALL_BRANCHES)
        events = tok.tokenize_ecg(self._make_row())
        assert len(events) == 7
        codes = [e["code"] for e in events]
        assert "ECG//Class/AFIB" in codes
        assert "ECG//Prototype/2D_Morphology/23" in codes
        assert "ECG//Prototype/2D_Global/1" in codes

    def test_all_events_have_ecg_time(self):
        tok = ECGPrototypeTokenizer(route=ECGRoute.ALL_BRANCHES)
        row = self._make_row()
        events = tok.tokenize_ecg(row)
        for e in events:
            assert e["time"] == row["ecg_time"]

    def test_similarity_quantile_binning(self):
        import numpy as np
        tok = ECGPrototypeTokenizer(route=ECGRoute.ALL_BRANCHES)
        tok.similarity_quantile_edges = {
            "1D": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }
        assert tok.get_similarity_bin(0.1, "1D") == 0
        assert tok.get_similarity_bin(0.5, "1D") == 2
        assert tok.get_similarity_bin(0.9, "1D") == 4  # 5th bin (0.8-1.0)

    def test_inject_ecg_preserves_order(self, sample_events):
        ecg_time = sample_events[3]["time"]  # same as a vital
        ecg_events = [
            {"time": ecg_time, "code": "ECG//Class/SR", "value": None, "value_cat": None}
        ]
        result = inject_ecg_prototypes(sample_events, ecg_events)
        times = [e["time"] for e in result]
        assert times == sorted(times)

    def test_filter_ecgs_to_window(self):
        import polars as pl
        admit = datetime(2020, 3, 15, 8, 0)
        discharge = datetime(2020, 3, 18, 14, 0)
        hosp_df = pl.DataFrame({
            "hospitalization_id": ["H001"],
            "admission_dttm": [admit],
            "discharge_dttm": [discharge],
        })
        ecg_df = pl.DataFrame({
            "hospitalization_id": ["H001", "H001", "H001", "H001"],
            "ecg_time": [
                admit - timedelta(days=3),   # within 7d lookback -> keep
                admit - timedelta(days=10),  # outside lookback -> drop
                admit + timedelta(hours=12), # during admission -> keep
                discharge + timedelta(days=1),  # after discharge -> drop
            ],
            "top1class": ["SR", "SR", "AFIB", "SR"],
        })
        result = filter_ecgs_to_window(ecg_df, hosp_df, lookback_days=7)
        assert len(result) == 2
        assert set(result.columns) == set(ecg_df.columns)
        classes = result["top1class"].to_list()
        assert "SR" in classes  # the -3d one
        assert "AFIB" in classes  # the +12h one

    def test_filter_ecgs_to_window_custom_lookback(self):
        import polars as pl
        admit = datetime(2020, 3, 15, 8, 0)
        discharge = datetime(2020, 3, 18, 14, 0)
        hosp_df = pl.DataFrame({
            "hospitalization_id": ["H001"],
            "admission_dttm": [admit],
            "discharge_dttm": [discharge],
        })
        ecg_df = pl.DataFrame({
            "hospitalization_id": ["H001", "H001"],
            "ecg_time": [
                admit - timedelta(days=3),   # within 7d but outside 2d -> drop
                admit - timedelta(days=1),   # within 2d -> keep
            ],
            "top1class": ["SR", "AFIB"],
        })
        result = filter_ecgs_to_window(ecg_df, hosp_df, lookback_days=2)
        assert len(result) == 1
        assert result["top1class"][0] == "AFIB"

    def test_ecg_after_clinical_before_label(self, sample_events):
        t = sample_events[2]["time"]
        ecg_events = [
            {"time": t, "code": "ECG//Class/SR", "value": None, "value_cat": None}
        ]
        labels = [{"code": "LABEL//test", "time": t}]
        from clif_protoecg.stages.label_insertion import insert_labels
        with_labels = insert_labels(sample_events, labels)
        result = inject_ecg_prototypes(with_labels, ecg_events)
        # Find positions
        at_t = [e for e in result if e["time"] == t]
        codes_at_t = [e["code"] for e in at_t]
        if "LABEL//test" in codes_at_t and "ECG//Class/SR" in codes_at_t:
            ecg_idx = codes_at_t.index("ECG//Class/SR")
            label_idx = codes_at_t.index("LABEL//test")
            assert ecg_idx < label_idx


# ---------------------------------------------------------------------------
# Time Tokens
# ---------------------------------------------------------------------------


class TestGapToken:
    def test_zero_gap(self):
        assert _get_gap_token(0, [1, 5, 15, 30, 60]) is None

    def test_sub_minute_gap(self):
        assert _get_gap_token(0.5, [1, 5, 15, 30, 60]) is None

    def test_small_gap(self):
        assert _get_gap_token(3, [1, 5, 15, 30, 60]) == "DT//1-5"

    def test_large_gap(self):
        assert _get_gap_token(500, [1, 5, 15, 30, 60, 120, 240, 480]) == "DT//480+"

    def test_exact_boundary(self):
        assert _get_gap_token(60, [1, 5, 15, 30, 60, 120]) == "DT//30-60"


class TestClockTokens:
    def test_clock_tokens_generated(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = insert_clock_tokens(sample_events, admit, discharge)
        clock_codes = [e["code"] for e in result if e["code"].startswith("CLOCK//")]
        assert len(clock_codes) > 0

    def test_clock_format(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = insert_clock_tokens(sample_events, admit, discharge)
        for e in result:
            if e["code"].startswith("CLOCK//"):
                # Should match CLOCK//HH:00
                parts = e["code"].split("//")[1]
                assert ":" in parts
                h, m = parts.split(":")
                assert int(h) in [0, 4, 8, 12, 16, 20]
                assert m == "00"

    def test_clock_within_window(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = insert_clock_tokens(sample_events, admit, discharge)
        for e in result:
            if e["code"].startswith("CLOCK//"):
                assert admit <= e["time"] <= discharge

    def test_clock_before_clinical_events(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = insert_clock_tokens(sample_events, admit, discharge)
        for i, e in enumerate(result):
            if e["code"].startswith("CLOCK//"):
                # Check priority at same timestamp
                for j in range(i + 1, len(result)):
                    if result[j]["time"] != e["time"]:
                        break
                    assert sort_priority(result[j]["code"]) >= sort_priority(e["code"])


class TestGapTokens:
    def test_gap_tokens_inserted(self, sample_events):
        result = insert_gap_tokens(sample_events)
        gap_codes = [e["code"] for e in result if e["code"].startswith("DT//")]
        assert len(gap_codes) > 0

    def test_no_gap_before_first_event(self, sample_events):
        result = insert_gap_tokens(sample_events)
        assert not result[0]["code"].startswith("DT//")

    def test_gap_count(self, sample_events):
        result = insert_gap_tokens(sample_events)
        gap_count = sum(1 for e in result if e["code"].startswith("DT//"))
        # Only pairs with >=1 min gap get a DT token; 0-min pairs are skipped
        expected = sum(
            1 for i in range(1, len(sample_events))
            if (sample_events[i]["time"] - sample_events[i - 1]["time"]).total_seconds() >= 60
        )
        assert gap_count == expected

    def test_no_zero_gap_tokens(self, sample_events):
        result = insert_gap_tokens(sample_events)
        for e in result:
            if e["code"].startswith("DT//"):
                assert e["code"] != "DT//0", "Should not insert DT//0 tokens"

    def test_empty_events(self):
        assert insert_gap_tokens([]) == []


class TestAddTemporalTokens:
    def test_combined(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = add_temporal_tokens(sample_events, admit, discharge)
        codes = [e["code"] for e in result]
        has_clock = any(c.startswith("CLOCK//") for c in codes)
        has_gap = any(c.startswith("DT//") for c in codes)
        assert has_clock
        assert has_gap

    def test_chronological(self, sample_events):
        admit = sample_events[0]["time"]
        discharge = sample_events[-1]["time"]
        result = add_temporal_tokens(sample_events, admit, discharge)
        times = [e["time"] for e in result]
        assert times == sorted(times)
