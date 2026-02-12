"""Clock and gap token insertion."""

from __future__ import annotations

from datetime import datetime, timedelta

from clif_protoecg.core.constants import (
    CLOCK_INTERVAL_HOURS,
    CLOCK_TIMES,
    TIME_BUCKET_EDGES,
    sort_priority,
)


def insert_clock_tokens(
    events: list[dict],
    admission_time: datetime,
    discharge_time: datetime,
) -> list[dict]:
    """Insert CLOCK// tokens at fixed 4-hour intervals."""
    clock_events: list[dict] = []
    day_start = admission_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Find first clock boundary at or after admission
    current = None
    for hour in CLOCK_TIMES:
        candidate = day_start + timedelta(hours=hour)
        if candidate >= admission_time:
            current = candidate
            break
    if current is None:
        current = day_start + timedelta(days=1)

    while current <= discharge_time:
        h = current.hour
        clock_events.append({
            "time": current,
            "code": f"CLOCK//{h:02d}:00",
            "value": None,
            "value_cat": None,
        })
        current += timedelta(hours=CLOCK_INTERVAL_HOURS)

    merged = events + clock_events
    merged.sort(key=lambda e: (e["time"], sort_priority(e["code"])))
    return merged


def _get_gap_token(delta_minutes: float, edges: list[int]) -> str | None:
    """Map time delta to bucket token string. Returns None for <1 min gaps."""
    if delta_minutes < 1:
        return None
    for i in range(len(edges) - 1):
        if delta_minutes <= edges[i + 1]:
            return f"DT//{edges[i]}-{edges[i + 1]}"
    return f"DT//{edges[-1]}+"


def insert_gap_tokens(
    events: list[dict],
    bucket_edges: list[int] | None = None,
) -> list[dict]:
    """Insert DT// gap tokens between consecutive events (>=1 min apart)."""
    if bucket_edges is None:
        bucket_edges = TIME_BUCKET_EDGES

    if not events:
        return events

    result = [events[0]]
    for i in range(1, len(events)):
        prev_time = events[i - 1]["time"]
        curr_time = events[i]["time"]
        delta_min = (curr_time - prev_time).total_seconds() / 60.0
        dt_token = _get_gap_token(delta_min, bucket_edges)

        if dt_token is not None:
            result.append({
                "time": curr_time,
                "code": dt_token,
                "value": None,
                "value_cat": None,
            })
        result.append(events[i])

    return result


def add_temporal_tokens(
    events: list[dict],
    admission_time: datetime,
    discharge_time: datetime,
    bucket_edges: list[int] | None = None,
) -> list[dict]:
    """Add both clock and gap tokens. Clock first, then gaps."""
    with_clocks = insert_clock_tokens(events, admission_time, discharge_time)
    with_gaps = insert_gap_tokens(with_clocks, bucket_edges)
    return with_gaps
