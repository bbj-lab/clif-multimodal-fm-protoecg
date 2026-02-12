"""Parallel post-processing: labels, ECG injection, time tokens.

Uses multiprocessing with 'spawn' context to avoid Polars thread-pool
deadlocks that occur with fork.  Only shared objects (label evaluator,
ECG tokenizer) are serialized to workers via a temp pickle file.
Per-hospitalization data is passed as task arguments through imap_unordered
to avoid duplicating bulk data across every worker process.
"""

from __future__ import annotations

import copy
import os
import pickle
import tempfile
from typing import Any

import polars as pl

from clif_protoecg.labels.evaluator import LabelEvaluator
from clif_protoecg.stages.label_insertion import insert_labels
from clif_protoecg.stages.ecg_prototypes import ECGPrototypeTokenizer, inject_ecg_prototypes
from clif_protoecg.stages.time_tokens import add_temporal_tokens

# Module-level shared state — only label_eval and ecg_tokenizer.
# Populated by init_context (single-threaded) or _init_worker (spawn).
_ctx: dict[str, Any] = {}


# ------------------------------------------------------------------
# Context setup (called in parent for single-threaded mode)
# ------------------------------------------------------------------

def init_context(
    label_eval: LabelEvaluator,
    ecg_tokenizer: ECGPrototypeTokenizer,
) -> None:
    """Set shared context (label evaluator + ECG tokenizer only)."""
    _ctx.update(
        label_eval=label_eval,
        ecg_tokenizer=ecg_tokenizer,
    )


def clear_context() -> None:
    """Release shared context to free memory."""
    _ctx.clear()


# ------------------------------------------------------------------
# Spawn-based parallel helpers
# ------------------------------------------------------------------

def _make_picklable_ctx() -> dict[str, Any]:
    """Return a copy of _ctx safe for pickling (strip loader from evaluator)."""
    ctx = dict(_ctx)
    # LabelEvaluator.loader is a CLIFDataLoader with file handles —
    # strip it since workers only call evaluate_hospitalization().
    label_eval = copy.copy(ctx["label_eval"])
    label_eval.loader = None  # type: ignore[assignment]
    ctx["label_eval"] = label_eval
    return ctx


def save_context() -> str:
    """Pickle _ctx to a temp file, return the path."""
    ctx = _make_picklable_ctx()
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="postprocess_ctx_")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _init_worker(path: str) -> None:
    """Pool initializer: load shared context from the pickle file."""
    with open(path, "rb") as f:
        _ctx.update(pickle.load(f))


# ------------------------------------------------------------------
# Task builder — runs in parent, yields per-hid dicts for workers
# ------------------------------------------------------------------

def build_task(
    hid: str,
    *,
    all_sequences: dict[str, list[dict]],
    hosp_to_patient: dict[str, str],
    hosp_lookup: dict[str, dict],
    hosp_single_row: dict[str, pl.DataFrame],
    label_partitions: dict[str, tuple[dict[str, pl.DataFrame], pl.DataFrame]],
    cs_partition: dict[str, pl.DataFrame],
    cs_empty: pl.DataFrame | None,
    cs_by_hosp: bool,
    ecg_partition: dict[str, pl.DataFrame],
    hosp_df_empty: pl.DataFrame,
) -> dict[str, Any]:
    """Bundle per-hospitalization data into a task dict for process_hospitalization."""
    pid = hosp_to_patient.get(hid)
    hosp_row = hosp_lookup.get(hid)
    events = all_sequences.get(hid, [])

    tables: dict[str, pl.DataFrame] = {}
    for tbl_key, (partition, empty) in label_partitions.items():
        tables[tbl_key] = partition.get(hid, empty)
    if cs_empty is not None:
        if cs_by_hosp:
            tables["clif_code_status"] = cs_partition.get(hid, cs_empty)
        else:
            tables["clif_code_status"] = cs_partition.get(pid, cs_empty) if pid else cs_empty
    tables["clif_hospitalization"] = hosp_single_row.get(hid, hosp_df_empty)

    return {
        "hid": hid,
        "pid": pid,
        "hosp_row": hosp_row,
        "events": events,
        "tables": tables,
        "ecg_rows": ecg_partition.get(hid),
    }


# ------------------------------------------------------------------
# Per-hospitalization work function
# ------------------------------------------------------------------

def process_hospitalization(task: dict[str, Any]) -> dict[str, Any] | None:
    """Post-process a single hospitalization.

    Accepts a task dict built by build_task(). Shared objects (label_eval,
    ecg_tokenizer) are read from module-level _ctx.
    """
    hid = task["hid"]
    pid = task["pid"]
    hosp_row = task["hosp_row"]
    events = task["events"]
    tables = task["tables"]
    ecg_rows = task["ecg_rows"]

    if pid is None or hosp_row is None:
        return None

    label_eval = _ctx["label_eval"]

    # Labels
    n_labels = 0
    label_results = label_eval.evaluate_hospitalization(hosp_row, tables)
    if label_results:
        events = insert_labels(events, label_results)
        n_labels = len(label_results)

    # ECG
    n_ecg_records = 0
    n_ecg_events = 0
    if ecg_rows is not None:
        ecg_tokenizer = _ctx["ecg_tokenizer"]
        ecg_events = []
        for row in ecg_rows.iter_rows(named=True):
            ecg_events.extend(ecg_tokenizer.tokenize_ecg(row))
        if ecg_events:
            events = inject_ecg_prototypes(events, ecg_events)
            n_ecg_records = len(ecg_rows)
            n_ecg_events = len(ecg_events)

    # Time tokens
    admit = hosp_row["admission_dttm"]
    discharge = hosp_row["discharge_dttm"]
    if admit is not None and discharge is not None:
        events = add_temporal_tokens(events, admit, discharge)

    return {
        "hid": hid,
        "events": events,
        "n_labels": n_labels,
        "n_ecg_records": n_ecg_records,
        "n_ecg_events": n_ecg_events,
    }
