"""ECG prototype tokenization and insertion (3 routes)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from clif_protoecg.core.constants import sort_priority
from clif_protoecg.core.modalities import ECGRoute


@dataclass
class ECGPrototypeTokenizer:
    """Convert ECG prototype CSV rows into MEDS events."""

    route: ECGRoute = ECGRoute.ALL_BRANCHES
    similarity_quantile_edges: dict[str, list[float]] = field(default_factory=dict)

    def load_prototypes(
        self,
        csv_path: str,
        id_mapping: dict[int, str] | None = None,
        hosp_df: pl.DataFrame | None = None,
        lookback_days: int = 7,
    ) -> pl.DataFrame:
        """Load ECG prototype CSV, map MIMIC -> CLIF IDs, and filter to admission window.

        Parameters
        ----------
        csv_path : str
            Path to the ECG prototype CSV.
        id_mapping : dict | None
            MIMIC hadm_id -> CLIF hospitalization_id mapping.
        hosp_df : pl.DataFrame | None
            Hospitalization table with admission_dttm / discharge_dttm.
            When provided, ECGs are filtered to [admission - lookback_days, discharge].
        lookback_days : int
            Days before admission to include ECGs (default 7).
        """
        df = pl.read_csv(csv_path, infer_schema_length=10000)

        # Parse timestamp
        if "shifted_ecg_time" in df.columns:
            df = df.with_columns(
                pl.col("shifted_ecg_time").str.to_datetime().dt.replace_time_zone("UTC").alias("ecg_time")
            )

        # Map hadm_id -> hospitalization_id
        if id_mapping is not None:
            mapping_df = pl.DataFrame({
                "hadm_id": list(id_mapping.keys()),
                "hospitalization_id": list(id_mapping.values()),
            })
            # Ensure matching types
            if df["hadm_id"].dtype != mapping_df["hadm_id"].dtype:
                mapping_df = mapping_df.with_columns(
                    pl.col("hadm_id").cast(df["hadm_id"].dtype)
                )
            df = df.join(mapping_df, on="hadm_id", how="inner")

        # Filter to admission window
        if hosp_df is not None and "hospitalization_id" in df.columns:
            df = filter_ecgs_to_window(df, hosp_df, lookback_days=lookback_days)

        return df

    def compute_similarity_quantiles(
        self, df: pl.DataFrame, n_bins: int = 10
    ) -> dict[str, list[float]]:
        """Compute quantile edges for similarity scores per branch."""
        branches = {
            "1D": "1d_similarity",
            "2D_Morphology": "2d_partial_similarity",
            "2D_Global": "2d_global_similarity",
        }
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges: dict[str, list[float]] = {}
        for branch_name, col in branches.items():
            if col in df.columns:
                vals = df[col].drop_nulls().to_numpy()
                if len(vals) >= n_bins:
                    edges[branch_name] = np.unique(
                        np.percentile(vals, percentiles)
                    ).tolist()
        self.similarity_quantile_edges = edges
        return edges

    def get_similarity_bin(self, value: float, branch: str) -> int:
        """Map similarity score to quantile bin index."""
        edges = self.similarity_quantile_edges.get(branch)
        if edges is None:
            return 5
        for i in range(1, len(edges)):
            if value <= edges[i]:
                return i - 1
        return len(edges) - 2

    def tokenize_ecg(self, row: dict) -> list[dict]:
        """Convert one ECG row to MEDS event dicts."""
        if self.route == ECGRoute.NO_ECG:
            return []

        ecg_time = row["ecg_time"]
        events: list[dict] = []

        def _evt(code: str, val=None):
            return {"time": ecg_time, "code": code, "value": val, "value_cat": None}

        # Classification token
        top_class = row.get("top1class", "UNKNOWN")
        events.append(_evt(f"ECG//Class/{top_class}", row.get("top1score")))

        # 1D branch (always for fusion_class and all_branches)
        proto_1d = row.get("1d_prototype_num")
        sim_1d = row.get("1d_similarity")
        if proto_1d is not None:
            events.append(_evt(f"ECG//Prototype/1D/{int(proto_1d)}"))
        if sim_1d is not None:
            events.append(_evt("ECG//Similarity/1D", float(sim_1d)))

        if self.route == ECGRoute.ALL_BRANCHES:
            # 2D Morphology
            proto_2d = row.get("2d_partial_prototype_num")
            sim_2d = row.get("2d_partial_similarity")
            if proto_2d is not None:
                events.append(_evt(f"ECG//Prototype/2D_Morphology/{int(proto_2d)}"))
            if sim_2d is not None:
                events.append(_evt("ECG//Similarity/2D_Morphology", float(sim_2d)))

            # 2D Global
            proto_g = row.get("2d_global_prototype_num")
            sim_g = row.get("2d_global_similarity")
            if proto_g is not None:
                events.append(_evt(f"ECG//Prototype/2D_Global/{int(proto_g)}"))
            if sim_g is not None:
                events.append(_evt("ECG//Similarity/2D_Global", float(sim_g)))

        return events


def inject_ecg_prototypes(
    events: list[dict],
    ecg_events: list[dict],
) -> list[dict]:
    """Merge ECG events into existing event sequence."""
    merged = events + ecg_events
    merged.sort(key=lambda e: (e["time"], sort_priority(e["code"])))
    return merged


def filter_ecgs_to_window(
    ecg_df: pl.DataFrame,
    hosp_df: pl.DataFrame,
    lookback_days: int = 7,
) -> pl.DataFrame:
    """Keep only ECGs within [admission - lookback, discharge]."""
    original_cols = ecg_df.columns
    joined = ecg_df.join(
        hosp_df.select("hospitalization_id", "admission_dttm", "discharge_dttm"),
        on="hospitalization_id",
    )
    filtered = joined.filter(
        (pl.col("ecg_time") >= pl.col("admission_dttm") - pl.duration(days=lookback_days))
        & (pl.col("ecg_time") <= pl.col("discharge_dttm"))
    )
    return filtered.select(original_cols)
