"""Read/write event sequences as Parquet (replaces pickle serialization)."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def save_sequences(
    sequences: dict[str, list[dict]],
    path: str | Path,
) -> None:
    """Save event sequences to a Parquet file.

    Schema: hospitalization_id (str), event_idx (u32), time (datetime[μs, UTC]),
    code (str), value (f64 nullable), value_cat (str nullable).
    """
    rows: list[dict] = []
    for hid, events in sequences.items():
        for idx, evt in enumerate(events):
            rows.append({
                "hospitalization_id": hid,
                "event_idx": idx,
                "time": evt["time"],
                "code": evt["code"],
                "value": evt.get("value"),
                "value_cat": evt.get("value_cat"),
            })

    schema = {
        "hospitalization_id": pl.Utf8,
        "event_idx": pl.UInt32,
        "time": pl.Datetime("us", "UTC"),
        "code": pl.Utf8,
        "value": pl.Float64,
        "value_cat": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))


def merge_sequence_files(chunk_paths: list[Path], output_path: Path) -> int:
    """Merge multiple chunk sequence Parquet files into one.

    Returns total row count of the merged file.
    """
    if not chunk_paths:
        return 0
    lf = pl.concat([pl.scan_parquet(str(p)) for p in chunk_paths])
    df = lf.collect()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(output_path))
    return len(df)


def load_sequences(path: str | Path) -> dict[str, list[dict]]:
    """Load event sequences from a Parquet file.

    Returns {hospitalization_id: [event_dicts]} with events sorted by event_idx.
    """
    df = pl.read_parquet(str(path))
    sequences: dict[str, list[dict]] = {}

    for group_df in df.sort("event_idx").partition_by("hospitalization_id"):
        hid = group_df["hospitalization_id"][0]
        events = []
        for row in group_df.select("time", "code", "value", "value_cat").iter_rows(named=True):
            events.append(row)
        sequences[hid] = events

    return sequences
