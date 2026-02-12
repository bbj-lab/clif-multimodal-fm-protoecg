"""Tests for Phase 5: Vocabulary, tokenizers, tokenization stage."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from clif_protoecg.tokenization.vocabulary import (
    Vocabulary,
    VocabularyBuilder,
    QuantileBins,
)
from clif_protoecg.tokenization.modality_tokenizers.clinical import ClinicalTokenizer
from clif_protoecg.tokenization.modality_tokenizers.ecg import ECGTokenizer
from clif_protoecg.tokenization.modality_tokenizers.time import TimeTokenizer
from clif_protoecg.tokenization.stage import (
    tokenize_events,
    build_vocabulary_from_sequences,
    tokenize_and_write,
)


# ---------------------------------------------------------------------------
# QuantileBins
# ---------------------------------------------------------------------------


class TestQuantileBins:
    def test_get_bin_basic(self):
        qb = QuantileBins(edges={"X": [0, 10, 20, 30]})
        assert qb.get_bin("X", 5) == 0
        assert qb.get_bin("X", 15) == 1
        assert qb.get_bin("X", 25) == 2

    def test_get_bin_missing_code(self):
        qb = QuantileBins(edges={})
        assert qb.get_bin("X", 5) is None

    def test_roundtrip(self, tmp_path):
        qb = QuantileBins(edges={"A": [0, 1, 2, 3]})
        qb.save(tmp_path / "qb.json")
        loaded = QuantileBins.load(tmp_path / "qb.json")
        assert loaded.edges == qb.edges


# ---------------------------------------------------------------------------
# VocabularyBuilder
# ---------------------------------------------------------------------------


class TestVocabularyBuilder:
    def test_build_from_events(self, sample_events):
        builder = VocabularyBuilder(n_bins=5)
        builder.add_events(sample_events)
        vocab = builder.build()
        assert vocab.size > 0
        assert "[PAD]" in vocab.token_to_id
        assert "[BOS]" in vocab.token_to_id
        assert "[EOS]" in vocab.token_to_id

    def test_special_tokens_first(self, sample_events):
        builder = VocabularyBuilder()
        builder.add_events(sample_events)
        vocab = builder.build()
        assert vocab.token_to_id["[PAD]"] == 0
        assert vocab.token_to_id["[UNK]"] == 1
        assert vocab.token_to_id["[BOS]"] == 2
        assert vocab.token_to_id["[EOS]"] == 3

    def test_all_codes_in_vocab(self, sample_events):
        builder = VocabularyBuilder()
        builder.add_events(sample_events)
        vocab = builder.build()
        for e in sample_events:
            assert e["code"] in vocab.token_to_id, f"{e['code']} not in vocab"

    def test_quantile_bins_computed(self, sample_events):
        builder = VocabularyBuilder(n_bins=3)
        builder.add_events(sample_events)
        vocab = builder.build()
        # VITAL//hr has numeric values
        if "VITAL//hr" in vocab.quantile_bins.edges:
            edges = vocab.quantile_bins.edges["VITAL//hr"]
            assert len(edges) >= 2

    def test_bin_tokens_present(self, sample_events):
        builder = VocabularyBuilder(n_bins=5)
        builder.add_events(sample_events)
        vocab = builder.build()
        assert "Q_0" in vocab.token_to_id
        assert "Q_4" in vocab.token_to_id


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestVocabulary:
    def _make_vocab(self):
        builder = VocabularyBuilder(n_bins=3)
        events = [
            {"code": "VITAL//hr", "value": 70.0},
            {"code": "VITAL//hr", "value": 80.0},
            {"code": "VITAL//hr", "value": 90.0},
            {"code": "VITAL//hr", "value": 100.0},
            {"code": "DEMO//AGE_60-69", "value": None},
            {"code": "CLOCK//08:00", "value": None},
            {"code": "DT//0-1", "value": None},
        ]
        builder.add_events(events)
        return builder.build()

    def test_encode_decode_roundtrip(self):
        vocab = self._make_vocab()
        tokens = ["[BOS]", "VITAL//hr", "Q_1", "[EOS]"]
        ids = vocab.encode(tokens)
        decoded = vocab.decode(ids)
        assert decoded == tokens

    def test_unknown_token_maps_to_unk(self):
        vocab = self._make_vocab()
        ids = vocab.encode(["NONEXISTENT_TOKEN"])
        assert ids == [vocab.token_to_id["[UNK]"]]

    def test_save_load_roundtrip(self, tmp_path):
        vocab = self._make_vocab()
        vocab.save(tmp_path / "vocab_test")
        loaded = Vocabulary.load(tmp_path / "vocab_test")
        assert loaded.token_to_id == vocab.token_to_id
        assert loaded.size == vocab.size

    def test_pad_bos_eos_ids(self):
        vocab = self._make_vocab()
        assert vocab.pad_id == 0
        assert vocab.bos_id == 2
        assert vocab.eos_id == 3


# ---------------------------------------------------------------------------
# Modality Tokenizers
# ---------------------------------------------------------------------------


class TestClinicalTokenizer:
    def test_code_only(self):
        vocab = Vocabulary(
            token_to_id={"DEMO//AGE_60-69": 5},
            id_to_token={5: "DEMO//AGE_60-69"},
        )
        tok = ClinicalTokenizer(fuse_bins=False)
        result = tok.tokenize_event(
            {"code": "DEMO//AGE_60-69", "value": None}, vocab
        )
        assert result == ["DEMO//AGE_60-69"]

    def test_with_bin(self):
        vocab = Vocabulary(
            token_to_id={"VITAL//hr": 5, "Q_2": 6},
            id_to_token={5: "VITAL//hr", 6: "Q_2"},
            quantile_bins=QuantileBins(edges={"VITAL//hr": [0, 50, 80, 120]}),
        )
        tok = ClinicalTokenizer(fuse_bins=False)
        result = tok.tokenize_event(
            {"code": "VITAL//hr", "value": 90.0}, vocab
        )
        assert result == ["VITAL//hr", "Q_2"]

    def test_fused_bins(self):
        vocab = Vocabulary(
            token_to_id={"VITAL//hr/Q_1": 5},
            id_to_token={5: "VITAL//hr/Q_1"},
            quantile_bins=QuantileBins(edges={"VITAL//hr": [0, 50, 80, 120]}),
        )
        tok = ClinicalTokenizer(fuse_bins=True)
        result = tok.tokenize_event(
            {"code": "VITAL//hr", "value": 70.0}, vocab
        )
        assert result == ["VITAL//hr/Q_1"]


class TestECGTokenizer:
    def test_class_token(self):
        vocab = Vocabulary()
        tok = ECGTokenizer()
        result = tok.tokenize_event(
            {"code": "ECG//Class/AFIB", "value": 0.95}, vocab
        )
        assert result == ["ECG//Class/AFIB"]

    def test_similarity_with_bin(self):
        vocab = Vocabulary(
            quantile_bins=QuantileBins(
                edges={"ECG//Similarity/1D": [0.0, 0.5, 1.0]}
            )
        )
        tok = ECGTokenizer()
        result = tok.tokenize_event(
            {"code": "ECG//Similarity/1D", "value": 0.8}, vocab
        )
        assert result == ["ECG//Similarity/1D", "Q_1"]


class TestTimeTokenizer:
    def test_clock(self):
        vocab = Vocabulary()
        tok = TimeTokenizer()
        result = tok.tokenize_event({"code": "CLOCK//08:00", "value": None}, vocab)
        assert result == ["CLOCK//08:00"]

    def test_gap(self):
        vocab = Vocabulary()
        tok = TimeTokenizer()
        result = tok.tokenize_event({"code": "DT//30-60", "value": None}, vocab)
        assert result == ["DT//30-60"]


# ---------------------------------------------------------------------------
# Tokenization Stage
# ---------------------------------------------------------------------------


class TestTokenizeEvents:
    def test_bos_eos_wrapping(self, sample_events):
        vocab = build_vocabulary_from_sequences([sample_events])
        ids = tokenize_events(sample_events, vocab)
        tokens = vocab.decode(ids)
        assert tokens[0] == "[BOS]"
        assert tokens[-1] == "[EOS]"

    def test_output_all_valid_ids(self, sample_events):
        vocab = build_vocabulary_from_sequences([sample_events])
        ids = tokenize_events(sample_events, vocab)
        for i in ids:
            assert i in vocab.id_to_token


class TestTokenizeAndWrite:
    def test_writes_parquet(self, tmp_path, sample_events):
        vocab = build_vocabulary_from_sequences([sample_events])
        sequences = {"H001": sample_events}
        output = tmp_path / "tokenized.parquet"
        n = tokenize_and_write(sequences, vocab, output)
        assert n == 1
        df = pl.read_parquet(output)
        assert len(df) == 1
        assert df["hospitalization_id"][0] == "H001"
        assert df["token_count"][0] > 0

    def test_token_ids_match_count(self, tmp_path, sample_events):
        vocab = build_vocabulary_from_sequences([sample_events])
        sequences = {"H001": sample_events, "H002": sample_events[:3]}
        output = tmp_path / "tok2.parquet"
        tokenize_and_write(sequences, vocab, output)
        df = pl.read_parquet(output)
        for row in df.iter_rows(named=True):
            assert len(row["token_ids"]) == row["token_count"]
