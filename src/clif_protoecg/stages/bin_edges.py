"""Value discretization: compute bin edges per code."""

from __future__ import annotations

import numpy as np
import polars as pl

from clif_protoecg.core.artifacts import BinEdges
from clif_protoecg.core.constants import CLINICAL_BINS
from clif_protoecg.data.loader import CLIFDataLoader
from clif_protoecg.utils.logging import get_logger

logger = get_logger("bin_edges")


def compute_quantile_bins(values: np.ndarray, n_bins: int = 10) -> list[float]:
    """Compute quantile bin edges from an array of values."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    return np.unique(edges).tolist()


def _gather_numeric_values(
    loader: CLIFDataLoader,
    train_hosp_ids: list[str],
) -> dict[str, np.ndarray]:
    """Collect numeric values per code from training data."""
    values: dict[str, list[float]] = {}

    _SOURCES = [
        ("vitals", "vital_category", "vital_value", "VITAL//"),
        ("labs", "lab_category", "lab_value_numeric", "LAB_RESULT//"),
        ("medication_admin_continuous", "med_category", "med_dose", "MED_CONT//"),
        ("medication_admin_intermittent", "med_category", "med_dose", "MED_BOLUS//"),
    ]

    for table_name, cat_col, val_col, prefix in _SOURCES:
        try:
            df = loader.load_for_hospitalizations(table_name, train_hosp_ids)
        except FileNotFoundError:
            continue

        if len(df) == 0 or val_col not in df.columns:
            continue

        # Vectorized group-by to collect values per code
        grouped = (
            df.filter(pl.col(val_col).is_not_null())
            .group_by(cat_col)
            .agg(pl.col(val_col).alias("vals"))
        )
        n_codes = 0
        n_values = 0
        for row in grouped.iter_rows(named=True):
            code = f"{prefix}{row[cat_col]}"
            vals = row["vals"]
            values.setdefault(code, []).extend(vals)
            n_codes += 1
            n_values += len(vals)

        logger.info(
            f"  {prefix.rstrip('/'):20s} {n_codes:>5} codes, "
            f"{n_values:>9,} numeric values"
        )

    return {k: np.array(v) for k, v in values.items()}


def compute_bin_edges(
    loader: CLIFDataLoader,
    train_hosp_ids: list[str],
    bin_mode: str = "hybrid",
    n_bins: int = 10,
) -> BinEdges:
    """Compute discretisation bin edges for numeric codes.

    Modes:
    - hybrid: clinical bins where defined, quantile fallback elsewhere
    - clinical_only: only clinical bins, no fallback
    - quantile: pure quantile bins for everything
    """
    numeric_values = _gather_numeric_values(loader, train_hosp_ids)

    edges: dict[str, list[float]] = {}

    if bin_mode == "clinical_only":
        edges = {k: list(v) for k, v in CLINICAL_BINS.items()}

    elif bin_mode == "quantile":
        for code, vals in numeric_values.items():
            if len(vals) >= n_bins:
                edges[code] = compute_quantile_bins(vals, n_bins)

    else:  # hybrid
        # Start with clinical bins
        for code, bins in CLINICAL_BINS.items():
            edges[code] = list(bins)

        # Pad clinical bins if they have fewer boundaries than n_bins + 1
        for code in list(edges.keys()):
            if code in numeric_values and len(edges[code]) < n_bins + 1:
                vals = numeric_values[code]
                if len(vals) >= n_bins:
                    q_edges = compute_quantile_bins(vals, n_bins)
                    # Merge: keep clinical boundaries, add quantile sub-bins
                    merged = sorted(set(edges[code]) | set(q_edges))
                    edges[code] = merged

        # Fill remaining codes with quantile bins
        for code, vals in numeric_values.items():
            if code not in edges and len(vals) >= n_bins:
                edges[code] = compute_quantile_bins(vals, n_bins)

    n_clinical = sum(1 for c in edges if c in CLINICAL_BINS)
    n_quantile = len(edges) - n_clinical
    logger.info(
        f"  Bin edges: {len(edges)} codes "
        f"({n_clinical} clinical, {n_quantile} quantile)"
    )

    return BinEdges(edges=edges)
