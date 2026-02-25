"""Vectorized batch extraction: build events for ALL hospitalizations at once.

Replaces per-hospitalization Python loops with Polars/numpy vectorized
operations for ~10-20x speedup on the extraction step.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from clif_protoecg.core.artifacts import CodeProfile
from clif_protoecg.core.constants import SORT_PRIORITY_MAP, age_bucket
from clif_protoecg.data.loader import CLIFDataLoader

# Standard event schema for concatenation
_EVENT_COLS = ["hospitalization_id", "time", "code", "value", "value_cat"]
_EVENT_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "time": pl.Datetime("us", "UTC"),
    "code": pl.Utf8,
    "value": pl.Float64,
    "value_cat": pl.Utf8,
}


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=_EVENT_SCHEMA)


def _simple_events(
    df: pl.DataFrame,
    time_col: str,
    code_expr: pl.Expr,
    val_col: str | None = None,
    val_cat_col: str | None = None,
) -> pl.DataFrame:
    """Build event DataFrame from a table with a code expression."""
    if len(df) == 0:
        return _empty_events()

    cols = [
        pl.col("hospitalization_id"),
        pl.col(time_col).alias("time"),
        code_expr.alias("code"),
        (
            pl.col(val_col).cast(pl.Float64)
            if val_col
            else pl.lit(None).cast(pl.Float64)
        ).alias("value"),
        (
            pl.col(val_cat_col).cast(pl.Utf8)
            if val_cat_col
            else pl.lit(None).cast(pl.Utf8)
        ).alias("value_cat"),
    ]
    return df.select(cols)


def _filter_included(
    df: pl.DataFrame, code_prefix: str, cat_col: str, profile: CodeProfile
) -> pl.DataFrame:
    """Filter DataFrame to only included codes."""
    if not profile.included_codes:
        return df
    included_cats = {
        c[len(code_prefix) :]
        for c in profile.included_codes
        if c.startswith(code_prefix)
    }
    if not included_cats:
        return df.clear()
    return df.filter(pl.col(cat_col).is_in(list(included_cats)))


# ---------------------------------------------------------------------------
# Build sort priority column (vectorized)
# ---------------------------------------------------------------------------


def _add_sort_priority(df: pl.DataFrame) -> pl.DataFrame:
    """Add a _priority column based on code prefix."""
    expr = pl.lit(2.0)
    for prefix, prio in SORT_PRIORITY_MAP.items():
        expr = (
            pl.when(pl.col("code").str.starts_with(prefix))
            .then(pl.lit(float(prio)))
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("_priority"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_all_base_events(
    all_hosp_ids: list[str],
    loader: CLIFDataLoader,
    hosp_lookup: dict[str, dict],
    patient_lookup: dict[str, dict],
    hosp_to_patient: dict[str, str],
    profile: CodeProfile,
) -> dict[str, list[dict[str, Any]]]:
    """Build base event sequences for all hospitalizations using vectorized ops.

    Returns {hospitalization_id: [event_dicts]} sorted by (time, priority).
    Does NOT include labels, ECG, or time tokens (those are added per-hosp).
    """
    frames: list[pl.DataFrame] = []

    # ── 1. Demographics ─────────────────────────────────────────────
    demo_rows = []
    for hid in all_hosp_ids:
        hosp = hosp_lookup.get(hid)
        pid = hosp_to_patient.get(hid)
        if hosp is None or pid is None:
            continue
        patient = patient_lookup.get(pid, {})
        admit = hosp["admission_dttm"]

        age = hosp.get("age_at_admission")
        if age is not None:
            demo_rows.append(
                (hid, admit, f"DEMO//AGE_{age_bucket(int(age))}", None, None)
            )
        sex = patient.get("sex_category")
        if sex:
            demo_rows.append((hid, admit, f"DEMO//SEX_{sex}", None, None))
        race = patient.get("race_category")
        if race:
            demo_rows.append((hid, admit, f"DEMO//RACE_{race}", None, None))
        ethnicity = patient.get("ethnicity_category")
        if ethnicity:
            demo_rows.append((hid, admit, f"DEMO//ETHNICITY_{ethnicity}", None, None))

    if demo_rows:
        frames.append(
            pl.DataFrame(
                demo_rows,
                schema=_EVENT_SCHEMA,
                orient="row",
            )
        )

    # ── 3. ADT ──────────────────────────────────────────────────────
    adt = loader.load_for_hospitalizations("adt", all_hosp_ids)
    if len(adt) > 0:
        frames.append(
            _simple_events(
                adt,
                "in_dttm",
                pl.lit("ADT//") + pl.col("location_category").cast(pl.Utf8),
            )
        )

    # ── 4. Vitals ───────────────────────────────────────────────────
    vitals = loader.load_for_hospitalizations("vitals", all_hosp_ids)
    if len(vitals) > 0:
        v = _filter_included(vitals, "VITAL//", "vital_category", profile)
        if len(v) > 0:
            frames.append(
                _simple_events(
                    v,
                    "recorded_dttm",
                    pl.lit("VITAL//") + pl.col("vital_category").cast(pl.Utf8),
                    val_col="vital_value",
                )
            )

    # ── 5. Labs ─────────────────────────────────────────────────────
    labs = loader.load_for_hospitalizations("labs", all_hosp_ids)
    if len(labs) > 0:
        lb = _filter_included(labs, "LAB_RESULT//", "lab_category", profile)
        if len(lb) > 0:
            # Lab orders
            lb_orders = lb.filter(
                pl.col("lab_order_dttm").is_not_null()
                | pl.col("lab_collect_dttm").is_not_null()
            )
            if len(lb_orders) > 0:
                frames.append(
                    lb_orders.select(
                        pl.col("hospitalization_id"),
                        pl.coalesce("lab_order_dttm", "lab_collect_dttm").alias("time"),
                        (
                            pl.lit("LAB_ORDER//") + pl.col("lab_category").cast(pl.Utf8)
                        ).alias("code"),
                        pl.lit(None).cast(pl.Float64).alias("value"),
                        pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                    )
                )

            # Lab results
            lb_results = lb.filter(pl.col("lab_value_numeric").is_not_null())
            if len(lb_results) > 0:
                # We can't be using `lab_collect_dttm` for lab results
                result_time = pl.coalesce("lab_result_dttm")
                lb_with_time = lb_results.filter(result_time.is_not_null())
                if len(lb_with_time) > 0:
                    frames.append(
                        lb_with_time.select(
                            pl.col("hospitalization_id"),
                            result_time.alias("time"),
                            (
                                pl.lit("LAB_RESULT//")
                                + pl.col("lab_category").cast(pl.Utf8)
                            ).alias("code"),
                            pl.col("lab_value_numeric").cast(pl.Float64).alias("value"),
                            pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                        )
                    )

    # ── 6. Meds Continuous ──────────────────────────────────────────
    meds_c = loader.load_for_hospitalizations(
        "medication_admin_continuous", all_hosp_ids
    )
    if len(meds_c) > 0:
        # Stop events (dose==0 or stop/paused/held)
        stop_mask = (
            pl.col("mar_action_category").is_in(["stop"])
            | (pl.col("med_dose").is_null())
            | (pl.col("med_dose") == 0)
        )
        stops = meds_c.filter(stop_mask)
        if len(stops) > 0:
            frames.append(
                _simple_events(
                    stops,
                    "admin_dttm",
                    pl.lit("MED_CONT_STOP//") + pl.col("med_category").cast(pl.Utf8),
                )
            )

        # Active infusions
        active = _filter_included(
            meds_c.filter(~stop_mask), "MED_CONT//", "med_category", profile
        )
        if len(active) > 0:
            frames.append(
                _simple_events(
                    active,
                    "admin_dttm",
                    pl.lit("MED_CONT//") + pl.col("med_category").cast(pl.Utf8),
                    val_col="med_dose",
                )
            )

    # ── 7. Meds Intermittent ────────────────────────────────────────
    meds_i = loader.load_for_hospitalizations(
        "medication_admin_intermittent", all_hosp_ids
    ).filter(pl.col("mar_action_category").is_in(["given", "bolus"]))
    if len(meds_i) > 0:
        mi = _filter_included(meds_i, "MED_BOLUS//", "med_category", profile)
        if len(mi) > 0:
            frames.append(
                _simple_events(
                    mi,
                    "admin_dttm",
                    pl.lit("MED_BOLUS//") + pl.col("med_category").cast(pl.Utf8),
                    val_col="med_dose",
                )
            )

    # ── 8. Respiratory ──────────────────────────────────────────────
    resp = loader.load_for_hospitalizations("respiratory_support", all_hosp_ids)
    if len(resp) > 0:
        # Device events
        frames.append(
            _simple_events(
                resp,
                "recorded_dttm",
                pl.lit("RESP//")
                + pl.col("device_category").fill_null("unknown").cast(pl.Utf8),
            )
        )
        # Parameter events
        for param_col, param_name in [
            ("fio2_set", "fio2"),
            ("peep_set", "peep"),
            ("tidal_volume_set", "tidal_volume"),
            ("resp_rate_set", "resp_rate_set"),
            ("pressure_support_set", "pressure_support"),
        ]:
            if param_col in resp.columns:
                param_df = resp.filter(pl.col(param_col).is_not_null())
                if len(param_df) > 0:
                    frames.append(
                        param_df.select(
                            pl.col("hospitalization_id"),
                            pl.col("recorded_dttm").alias("time"),
                            pl.lit(f"RESP_PARAM//{param_name}").alias("code"),
                            pl.col(param_col).cast(pl.Float64).alias("value"),
                            pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                        )
                    )

    # ── 9. Assessments ──────────────────────────────────────────────
    assess = loader.load_for_hospitalizations("patient_assessments", all_hosp_ids)
    if len(assess) > 0:
        a = _filter_included(assess, "ASSESS//", "assessment_category", profile)
        if len(a) > 0:
            frames.append(
                _simple_events(
                    a,
                    "recorded_dttm",
                    pl.lit("ASSESS//") + pl.col("assessment_category").cast(pl.Utf8),
                    val_col=(
                        "numerical_value" if "numerical_value" in a.columns else None
                    ),
                    val_cat_col=(
                        "categorical_value"
                        if "categorical_value" in a.columns
                        else None
                    ),
                )
            )

    # ── 10. Code Status ─────────────────────────────────────────────
    try:
        cs = loader.load("code_status")
        if "hospitalization_id" in cs.columns:
            cs_filtered = cs.filter(pl.col("hospitalization_id").is_in(all_hosp_ids))
            if len(cs_filtered) > 0:
                frames.append(
                    _simple_events(
                        cs_filtered,
                        "start_dttm",
                        pl.lit("CODE_STATUS//")
                        + pl.col("code_status_category").cast(pl.Utf8),
                    )
                )
        else:
            # Join with hospitalization to get hospitalization_id + time window
            hosp_map_df = pl.DataFrame(
                {
                    "hospitalization_id": list(hosp_lookup.keys()),
                    "_patient_id": [hosp_to_patient.get(hid) for hid in hosp_lookup],
                    "_admit": [h["admission_dttm"] for h in hosp_lookup.values()],
                    "_discharge": [h["discharge_dttm"] for h in hosp_lookup.values()],
                }
            )
            pids = [
                hosp_to_patient[hid] for hid in all_hosp_ids if hid in hosp_to_patient
            ]
            cs_filtered = cs.filter(pl.col("patient_id").is_in(pids))
            if len(cs_filtered) > 0:
                cs_joined = cs_filtered.join(
                    hosp_map_df,
                    left_on="patient_id",
                    right_on="_patient_id",
                    how="inner",
                ).filter(
                    (pl.col("start_dttm") >= pl.col("_admit"))
                    & (pl.col("start_dttm") <= pl.col("_discharge"))
                )
                if len(cs_joined) > 0:
                    frames.append(
                        _simple_events(
                            cs_joined,
                            "start_dttm",
                            pl.lit("CODE_STATUS//")
                            + pl.col("code_status_category").cast(pl.Utf8),
                        )
                    )
    except FileNotFoundError:
        pass

    # ── 11. Position ────────────────────────────────────────────────
    pos = loader.load_for_hospitalizations("position", all_hosp_ids)
    if len(pos) > 0:
        frames.append(
            _simple_events(
                pos,
                "recorded_dttm",
                pl.lit("POSITION//") + pl.col("position_category").cast(pl.Utf8),
            )
        )

    # ── 12. CRRT ────────────────────────────────────────────────────
    crrt = loader.load_for_hospitalizations("crrt_therapy", all_hosp_ids)
    if len(crrt) > 0:
        frames.append(
            _simple_events(
                crrt,
                "recorded_dttm",
                pl.lit("CRRT//")
                + pl.col("crrt_mode_category").fill_null("unknown").cast(pl.Utf8),
            )
        )
        for param_col, param_name in [
            ("blood_flow_rate", "blood_flow"),
            ("ultrafiltration_out", "uf_out"),
        ]:
            if param_col in crrt.columns:
                p_df = crrt.filter(pl.col(param_col).is_not_null())
                if len(p_df) > 0:
                    frames.append(
                        p_df.select(
                            pl.col("hospitalization_id"),
                            pl.col("recorded_dttm").alias("time"),
                            pl.lit(f"CRRT_PARAM//{param_name}").alias("code"),
                            pl.col(param_col).cast(pl.Float64).alias("value"),
                            pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                        )
                    )

    # ── 13. ECMO ────────────────────────────────────────────────────
    ecmo = loader.load_for_hospitalizations("ecmo_mcs", all_hosp_ids)
    if len(ecmo) > 0:
        frames.append(
            _simple_events(
                ecmo,
                "recorded_dttm",
                pl.lit("ECMO//")
                + pl.col("device_category").fill_null("ECMO").cast(pl.Utf8),
            )
        )
        if "flow" in ecmo.columns:
            flow_df = ecmo.filter(pl.col("flow").is_not_null())
            if len(flow_df) > 0:
                frames.append(
                    flow_df.select(
                        pl.col("hospitalization_id"),
                        pl.col("recorded_dttm").alias("time"),
                        pl.lit("ECMO_PARAM//flow").alias("code"),
                        pl.col("flow").cast(pl.Float64).alias("value"),
                        pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                    )
                )

    # ── 14. Procedures ──────────────────────────────────────────────
    procs = loader.load_for_hospitalizations("patient_procedures", all_hosp_ids)
    if len(procs) > 0:
        p = _filter_included(procs, "PROC//", "procedure_code", profile)
        if len(p) > 0 and "procedure_billed_dttm" in p.columns:
            p = p.filter(
                pl.col("procedure_billed_dttm").is_not_null()
            ).with_columns(
                pl.col("procedure_billed_dttm").dt.replace(
                    hour=23, minute=59, second=59
                )  # we don't have the exact time for procedures, so the cast to time needs to place them at EOD
            )
            if len(p) > 0:
                frames.append(
                    _simple_events(
                        p,
                        "procedure_billed_dttm",
                        pl.lit("PROC//") + pl.col("procedure_code").cast(pl.Utf8),
                    )
                )

    # ── 15. Discharge ───────────────────────────────────────────────
    disch_rows = []
    for hid, hosp in hosp_lookup.items():
        discharge = hosp.get("discharge_dttm")
        if discharge is None:
            continue
        disch_rows.append((hid, discharge, "DISCH//GENERAL", None, None))
        cat = hosp.get("discharge_category")
        if cat:
            disch_rows.append((hid, discharge, f"DISCH//{cat}", None, None))
    if disch_rows:
        frames.append(pl.DataFrame(disch_rows, schema=_EVENT_SCHEMA, orient="row"))

    # ── 16. Billing Codes ───────────────────────────────────────────
    diags = loader.load_for_hospitalizations("hospital_diagnosis", all_hosp_ids)
    if len(diags) > 0:
        # Build discharge+1min times
        disch_df = pl.DataFrame(
            {
                "hospitalization_id": [
                    hid
                    for hid, h in hosp_lookup.items()
                    if h.get("discharge_dttm") is not None
                ],
                "_billing_time": [
                    h["discharge_dttm"] + timedelta(minutes=1)
                    for h in hosp_lookup.values()
                    if h.get("discharge_dttm") is not None
                ],
            }
        )
        billing = diags.join(disch_df, on="hospitalization_id", how="inner")
        if len(billing) > 0:
            frames.append(
                billing.select(
                    pl.col("hospitalization_id"),
                    pl.col("_billing_time").alias("time"),
                    (pl.lit("ICD//") + pl.col("diagnosis_code").cast(pl.Utf8)).alias(
                        "code"
                    ),
                    pl.lit(None).cast(pl.Float64).alias("value"),
                    pl.lit(None).cast(pl.Utf8).alias("value_cat"),
                )
            )

    # ── Concat + Sort + Partition ───────────────────────────────────
    if not frames:
        return {hid: [] for hid in all_hosp_ids}

    # Filter out empty frames
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return {hid: [] for hid in all_hosp_ids}

    all_events = pl.concat(frames, how="diagonal_relaxed")

    # Add sort priority and sort
    all_events = _add_sort_priority(all_events)
    all_events = all_events.sort("hospitalization_id", "time", "_priority")

    # Clip to admission window per hospitalization
    admit_df = pl.DataFrame(
        {
            "hospitalization_id": list(hosp_lookup.keys()),
            "_admit": [h["admission_dttm"] for h in hosp_lookup.values()],
        }
    )
    all_events = (
        all_events.join(admit_df, on="hospitalization_id", how="left")
        .filter(pl.col("time") >= pl.col("_admit"))
        .drop("_admit", "_priority")
    )

    # Partition by hospitalization_id -> dict of event lists
    result: dict[str, list[dict]] = {hid: [] for hid in all_hosp_ids}
    for group_df in all_events.partition_by("hospitalization_id"):
        if len(group_df) == 0:
            continue
        hid = group_df["hospitalization_id"][0]
        result[hid] = group_df.drop("hospitalization_id").to_dicts()

    return result
