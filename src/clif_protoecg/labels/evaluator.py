"""Vectorized batch label evaluation."""

from __future__ import annotations

from typing import Any

import polars as pl

from clif_protoecg.labels.base import LabelDefinition, get_registry
from clif_protoecg.data.loader import CLIFDataLoader

# Trigger auto-registration of all label definitions
import clif_protoecg.labels.definitions  # noqa: F401


class LabelEvaluator:
    """Evaluate all registered labels for a set of hospitalizations."""

    def __init__(
        self,
        loader: CLIFDataLoader,
        categories: list[str] | None = None,
    ) -> None:
        self.loader = loader
        self.categories = categories or ["clif_only", "clif_partial"]
        self._registry = get_registry()

    def evaluate_hospitalization(
        self,
        hosp_row: dict[str, Any],
        tables: dict[str, pl.DataFrame],
    ) -> list[dict]:
        """Compute all applicable labels for one hospitalization.

        Returns list of {"code": "LABEL//...", "time": datetime} dicts.
        """
        results = []
        for defn in self._registry.values():
            if defn.category not in self.categories:
                continue
            try:
                result = defn.compute_fn(hosp_row=hosp_row, tables=tables)
                if result is not None:
                    results.append(result)
            except Exception:
                # Skip labels that fail (e.g. missing columns)
                continue
        return results

    def evaluate_batch(
        self,
        hospitalization_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Evaluate labels for a batch of hospitalizations.

        Returns {hospitalization_id: [label_dicts]}.
        """
        hosp_df = self.loader.load("hospitalization")

        # Pre-load all needed tables
        tables: dict[str, pl.DataFrame] = {}
        needed = set()
        for defn in self._registry.values():
            if defn.category in self.categories:
                needed.update(defn.source_tables)

        table_name_map = {
            "clif_hospitalization": "hospitalization",
            "clif_adt": "adt",
            "clif_vitals": "vitals",
            "clif_labs": "labs",
            "clif_medication_admin_continuous": "medication_admin_continuous",
            "clif_respiratory_support": "respiratory_support",
            "clif_patient_assessments": "patient_assessments",
            "clif_code_status": "code_status",
            "clif_position": "position",
            "clif_crrt_therapy": "crrt_therapy",
            "clif_ecmo_mcs": "ecmo_mcs",
        }

        for clif_name, loader_name in table_name_map.items():
            if clif_name in needed or not needed:
                try:
                    tables[clif_name] = self.loader.load_for_hospitalizations(
                        loader_name, hospitalization_ids
                    )
                except FileNotFoundError:
                    pass

        # Always load hospitalization table
        tables["clif_hospitalization"] = hosp_df.filter(
            pl.col("hospitalization_id").is_in(hospitalization_ids)
        )

        results: dict[str, list[dict]] = {}
        for hid in hospitalization_ids:
            hosp_rows = hosp_df.filter(
                pl.col("hospitalization_id") == hid
            )
            if len(hosp_rows) == 0:
                results[hid] = []
                continue
            hosp_row = hosp_rows.row(0, named=True)
            results[hid] = self.evaluate_hospitalization(hosp_row, tables)

        return results

    @property
    def label_names(self) -> list[str]:
        return [
            defn.name
            for defn in self._registry.values()
            if defn.category in self.categories
        ]

    @property
    def label_tokens(self) -> list[str]:
        return [
            defn.token
            for defn in self._registry.values()
            if defn.category in self.categories
        ]
