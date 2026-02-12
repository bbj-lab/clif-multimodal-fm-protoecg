# Step 5: Clock and Gap Tokens

## Goal

Add temporal encoding tokens to the event sequence. Two complementary mechanisms capture both absolute time-of-day and relative inter-event spacing:

1. **Clock tokens**: Fixed daily anchors every 4 hours (absolute time reference)
2. **Gap tokens**: Delta-time buckets between consecutive events (relative timing)

## Clock Tokens

### Design

Clock tokens are inserted at fixed 4-hour intervals throughout the hospitalization, regardless of clinical event density. They provide:
- Absolute time-of-day context (circadian patterns)
- Regular temporal anchors for the model's attention mechanism
- A stopping criterion during inference (12 clock tokens = 48 hours)

### Token Format

```
CLOCK//00:00    # Midnight
CLOCK//04:00    # Early morning
CLOCK//08:00    # Morning
CLOCK//12:00    # Noon
CLOCK//16:00    # Afternoon
CLOCK//20:00    # Evening
```

6 clock tokens per day.

### Insertion Algorithm

```python
from datetime import datetime, timedelta


CLOCK_TIMES = [0, 4, 8, 12, 16, 20]  # Hours of the day for clock tokens
CLOCK_INTERVAL_HOURS = 4


def insert_clock_tokens(
    events: list[dict],
    admission_time: datetime,
    discharge_time: datetime,
) -> list[dict]:
    """Insert clock tokens at fixed 4-hour intervals.

    Clock tokens appear at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    daily from the first relevant clock time after admission through discharge.
    """
    clock_events = []

    # Start from the first clock boundary at or after admission
    day_start = admission_time.replace(hour=0, minute=0, second=0, microsecond=0)

    for hour in CLOCK_TIMES:
        clock_time = day_start + timedelta(hours=hour)
        if clock_time >= admission_time:
            # Found first clock boundary
            current = clock_time
            break
    else:
        current = day_start + timedelta(days=1)  # Next day's midnight

    # Generate clock tokens through discharge
    while current <= discharge_time:
        hour = current.hour
        clock_events.append({
            "time": current,
            "code": f"CLOCK//{hour:02d}:00",
            "value": None,
            "value_cat": None,
        })
        current += timedelta(hours=CLOCK_INTERVAL_HOURS)

    # Merge with existing events
    all_events = events + clock_events
    all_events.sort(key=lambda e: (e["time"], _sort_priority(e["code"])))

    return all_events


def _sort_priority(code: str) -> int:
    """Sort priority for same-timestamp events (lower = earlier).

    Matches the priority table from Step 1.
    """
    if code.startswith("DEMO//") or code.startswith("ADMIT//"):
        return 0
    if code.startswith("CLOCK//"):
        return 1  # After demographics, before clinical events
    if code.startswith("ECG//"):
        return 2.5
    if code.startswith("LABEL//"):
        return 3
    if code.startswith("DISCH//"):
        return 4
    if code.startswith("ICD//") or code.startswith("DRG//"):
        # Retrospective billing codes (discharge + 1 min)
        return 5
    return 2
```

## Gap Tokens (Delta-Time)

### Design

Gap tokens encode the time elapsed between consecutive events. This captures the clinical significance of event spacing -- a vital sign measured every hour versus every 6 hours conveys different patient acuity.

### Time Buckets

Based on the reference repository, using clinically meaningful intervals:

| Token | Time Range | Clinical Interpretation |
|-------|-----------|----------------------|
| `DT//0` | 0 min | Simultaneous event |
| `DT//0-1` | (0, 1] min | Near-simultaneous |
| `DT//1-5` | (1, 5] min | Within same clinical action |
| `DT//5-15` | (5, 15] min | Same rounding period |
| `DT//15-30` | (15, 30] min | Short interval |
| `DT//30-60` | (30, 60] min | Medium interval |
| `DT//60-120` | (60, 120] min | 1-2 hours |
| `DT//120-240` | (120, 240] min | 2-4 hours |
| `DT//240-480` | (240, 480] min | 4-8 hours (shift-level) |
| `DT//480+` | >480 min | More than 8 hours |

### Configuration

```python
# Default bucket boundaries in minutes
TIME_BUCKET_EDGES = [0, 1, 5, 15, 30, 60, 120, 240, 480]

# Minimum gap to emit a DT token (gaps smaller than this are DT//0)
MIN_GAP_MINUTES = 0
```

### Insertion Algorithm

```python
def insert_gap_tokens(
    events: list[dict],
    bucket_edges: list[int] | None = None,
    min_gap_minutes: float = 0,
) -> list[dict]:
    """Insert delta-time tokens between consecutive events.

    A DT token is inserted before each event (except the first)
    indicating the time elapsed since the previous event.
    """
    if bucket_edges is None:
        bucket_edges = TIME_BUCKET_EDGES

    if not events:
        return events

    result = [events[0]]  # First event has no predecessor

    for i in range(1, len(events)):
        prev_time = events[i - 1]["time"]
        curr_time = events[i]["time"]

        delta_minutes = (curr_time - prev_time).total_seconds() / 60.0

        # Get bucket token
        dt_token = _get_gap_token(delta_minutes, bucket_edges)

        if dt_token is not None:
            result.append({
                "time": curr_time,
                "code": dt_token,
                "value": None,
                "value_cat": None,
            })

        result.append(events[i])

    return result


def _get_gap_token(delta_minutes: float, edges: list[int]) -> str | None:
    """Map a time delta to its bucket token.

    Args:
        delta_minutes: Time elapsed in minutes
        edges: Bucket boundaries

    Returns:
        Token string like "DT//30-60" or None if gap is 0
    """
    if delta_minutes <= 0:
        return "DT//0"

    for i in range(len(edges) - 1):
        if delta_minutes <= edges[i + 1]:
            return f"DT//{edges[i]}-{edges[i + 1]}"

    # Exceeds max bucket
    return f"DT//{edges[-1]}+"
```

## Combined Insertion Order

The full token insertion pipeline applies clock tokens first, then gap tokens:

```python
def add_temporal_tokens(
    events: list[dict],
    admission_time: datetime,
    discharge_time: datetime,
) -> list[dict]:
    """Add both clock and gap tokens to an event sequence.

    Order of operations:
    1. Insert clock tokens at fixed intervals
    2. Re-sort all events chronologically
    3. Insert gap tokens between consecutive events
    """
    # Step 1: Add clock tokens
    events_with_clocks = insert_clock_tokens(
        events, admission_time, discharge_time
    )

    # Step 2: Add gap tokens
    events_with_gaps = insert_gap_tokens(events_with_clocks)

    return events_with_gaps
```

## Example Sequence

Before temporal tokens:
```
DEMO//AGE_60-69 DEMO//SEX_Male VITAL//hr LAB_ORDER//glucose VITAL//sbp LAB_RESULT//glucose LABEL//mortality DISCH//Expired
```

After temporal tokens:
```
CLOCK//08:00 DEMO//AGE_60-69 DT//0 DEMO//SEX_Male DT//5-15 VITAL//hr DT//0 LAB_ORDER//glucose DT//30-60 VITAL//sbp DT//60-120 LAB_RESULT//glucose CLOCK//12:00 ... DT//480+ LABEL//mortality DT//0 DISCH//Expired
```

## Role During Inference

Clock tokens serve as the stopping criterion for horizon-based evaluation:

```python
def should_stop_generation(
    generated_tokens: list[int],
    clock_token_ids: set[int],
    death_token_ids: set[int],
    discharge_token_ids: set[int],
    eos_token_id: int,
    max_clock_tokens: int = 12,  # 12 x 4h = 48h horizon
) -> bool:
    """Check if generation should stop.

    Stop conditions:
    1. Generated EOS token
    2. Generated death/discharge token
    3. Generated max_clock_tokens clock tokens (horizon reached)
    """
    if generated_tokens[-1] == eos_token_id:
        return True
    if generated_tokens[-1] in death_token_ids:
        return True
    if generated_tokens[-1] in discharge_token_ids:
        return True

    clock_count = sum(1 for t in generated_tokens if t in clock_token_ids)
    return clock_count >= max_clock_tokens
```

## Dependencies

- Input: Step 3/4 event sequences (with labels and optional ECG tokens)
- Output: Complete tokenizable sequences for Step 6 (Model Training)
- Configuration: Bucket edges, clock interval, min gap threshold
