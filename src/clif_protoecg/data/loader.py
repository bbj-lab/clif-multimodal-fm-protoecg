"""Lazy and eager Parquet loading for CLIF tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clif_protoecg.utils.logging import get_logger

logger = get_logger("loader")

# Tables small enough to load eagerly (< ~10M rows)
_EAGER_TABLES = frozenset(
    {
        "patient",
        "hospitalization",
        "adt",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "respiratory_support",
        "patient_assessments",
        "code_status",
        "position",
        "hospital_diagnosis",
        "patient_procedures",
        "crrt_therapy",
        "ecmo_mcs",
    }
)

# Large tables that benefit from lazy scanning
_LAZY_TABLES = frozenset({"vitals", "labs"})


class CLIFDataLoader:
    """Load CLIF v2.1.0 Parquet files.

    Small tables are eagerly loaded and cached.
    Large tables (vitals, labs) use scan_parquet for predicate pushdown.
    """

    CLIF_TABLES = sorted(_EAGER_TABLES | _LAZY_TABLES)

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pl.DataFrame] = {}
        self._filtered_cache: dict[str, pl.DataFrame] = {}
        self._partitions: dict[str, tuple[dict[str, pl.DataFrame], pl.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, table_name: str) -> pl.DataFrame:
        """Load a CLIF table eagerly (cached after first load)."""
        if table_name in self._cache:
            return self._cache[table_name]

        path = self._resolve_path(table_name)
        df = pl.read_parquet(path)
        self._cache[table_name] = df
        return df

    def scan(self, table_name: str) -> pl.LazyFrame:
        """Return a lazy frame for predicate-pushdown queries."""
        path = self._resolve_path(table_name)
        return pl.scan_parquet(path)

    def preload(
        self,
        hospitalization_ids: list[str],
        id_column: str = "hospitalization_id",
    ) -> None:
        """Bulk-load and partition all tables for fast per-ID lookups.

        Loads lazy tables (vitals, labs) once filtered to the given IDs,
        and partitions all tables by hospitalization_id for O(1) lookups.
        """
        tables_to_preload = [t for t in self.CLIF_TABLES if t != "patient"]
        for name in tables_to_preload:
            try:
                if name in _LAZY_TABLES:
                    logger.info(f"  Loading {name} (lazy scan + filter)...")
                    df = (
                        self.scan(name)
                        .filter(pl.col(id_column).is_in(hospitalization_ids))
                        .collect()
                    )
                    self._cache[name] = df
                    self._filtered_cache[name] = df
                    logger.info(f"  Preloaded {name}: {len(df):,} rows")
                else:
                    logger.info(f"  Loading {name}...")
                    full_df = self.load(name)
                    if id_column in full_df.columns:
                        df = full_df.filter(
                            pl.col(id_column).is_in(hospitalization_ids)
                        )
                        self._filtered_cache[name] = df
                    else:
                        df = full_df
                    logger.info(f"  Loaded {name}: {len(df):,} rows")

                # Partition by hospitalization_id for O(1) per-ID lookup
                if id_column in df.columns:
                    logger.info(f"  Partitioning {name}...")
                    partition: dict[str, pl.DataFrame] = {}
                    for group_df in df.partition_by(id_column):
                        if len(group_df) > 0:
                            key = group_df[id_column][0]
                            partition[key] = group_df
                    self._partitions[name] = (partition, df.clear())

            except FileNotFoundError:
                logger.info(f"  Skipping {name} (not found)")
                pass

    def load_for_hospitalizations(
        self,
        table_name: str,
        hospitalization_ids: list[str],
        id_column: str = "hospitalization_id",
    ) -> pl.DataFrame:
        """Load rows matching a set of hospitalization IDs.

        Uses pre-partitioned cache when available, otherwise falls back to
        lazy scan + filter for large tables, eager filter otherwise.
        """
        # Fast path: partitioned data available
        if table_name in self._partitions:
            partition, empty = self._partitions[table_name]

            # Single ID: direct dict lookup (O(1))
            if len(hospitalization_ids) == 1:
                return partition.get(hospitalization_ids[0], empty)

            n_requested = len(hospitalization_ids)
            n_partitioned = len(partition)

            # If requesting all/most data, filter the cached DataFrame
            # instead of concatenating thousands of tiny DataFrames
            if (
                n_requested >= n_partitioned * 0.5
                and table_name in self._filtered_cache
            ):
                if n_requested >= n_partitioned:
                    return self._filtered_cache[table_name]
                return self._filtered_cache[table_name].filter(
                    pl.col(id_column).is_in(hospitalization_ids)
                )

            # Small subset: gather from partitions
            parts = [partition[hid] for hid in hospitalization_ids if hid in partition]
            return pl.concat(parts) if parts else empty

        # Lazy table without preload: scan from disk
        if table_name in _LAZY_TABLES:
            if table_name in self._cache:
                return self._cache[table_name].filter(
                    pl.col(id_column).is_in(hospitalization_ids)
                )
            return (
                self.scan(table_name)
                .filter(pl.col(id_column).is_in(hospitalization_ids))
                .collect()
            )

        # Eager table: filter cached DataFrame (only if the id column exists)
        df = self.load(table_name)
        if id_column not in df.columns:
            return df
        return df.filter(pl.col(id_column).is_in(hospitalization_ids))

    def get_patient_ids(self) -> list[str]:
        """Return all unique patient IDs."""
        hosp = self.load("hospitalization")
        return hosp["patient_id"].unique().sort().to_list()

    def get_hospitalization_ids(
        self, patient_ids: list[str] | None = None
    ) -> list[str]:
        """Return hospitalization IDs, optionally filtered by patients."""
        hosp = self.load("hospitalization")
        if patient_ids is not None:
            hosp = hosp.filter(pl.col("patient_id").is_in(patient_ids))
        return hosp["hospitalization_id"].unique().sort().to_list()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_path(self, table_name: str) -> Path:
        path = self.data_dir / f"clif_{table_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"CLIF table not found: {path}. "
                f"Expected file clif_{table_name}.parquet in {self.data_dir}"
            )
        return path

    def clear_preload(self) -> None:
        """Clear preloaded/partitioned data without clearing base table cache.

        Keeps eagerly-loaded base tables (hospitalization, patient, etc.)
        but removes filtered caches and partitions, and clears lazy table
        caches so they rescan from disk on next preload.
        """
        self._filtered_cache.clear()
        self._partitions.clear()
        for name in _LAZY_TABLES:
            self._cache.pop(name, None)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._filtered_cache.clear()
        self._partitions.clear()
