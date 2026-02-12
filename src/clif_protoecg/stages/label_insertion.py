"""Insert label tokens into event sequences at first occurrence."""

from __future__ import annotations

from clif_protoecg.core.constants import sort_priority


def insert_labels(
    events: list[dict],
    labels: list[dict],
) -> list[dict]:
    """Merge label tokens into the chronological event sequence.

    Each label appears at most once. Labels sort after clinical events
    at the same timestamp (priority 3).
    """
    # De-duplicate: keep first occurrence per label code
    seen: set[str] = set()
    unique_labels: list[dict] = []
    for lab in sorted(labels, key=lambda x: x["time"]):
        if lab["code"] not in seen:
            seen.add(lab["code"])
            unique_labels.append(lab)

    label_events = [
        {"time": l["time"], "code": l["code"], "value": None, "value_cat": None}
        for l in unique_labels
    ]

    merged = events + label_events
    merged.sort(key=lambda e: (e["time"], sort_priority(e["code"])))
    return merged
