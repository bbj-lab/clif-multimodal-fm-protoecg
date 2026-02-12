"""Map between MIMIC subject_id/hadm_id and CLIF patient_id/hospitalization_id."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


class IDMapper:
    """Bidirectional mapping between MIMIC and CLIF identifiers.

    The ECG prototype CSV uses MIMIC IDs (subject_id, hadm_id) while
    the rest of the pipeline uses CLIF IDs (patient_id, hospitalization_id).
    This mapper handles the translation.

    The mapping can be loaded from:
    1. A JSON file with explicit mappings
    2. CLIF tables that contain both ID systems
    3. Direct construction from dictionaries
    """

    def __init__(
        self,
        patient_map: dict[int, str] | None = None,
        hosp_map: dict[int, str] | None = None,
    ) -> None:
        # MIMIC int -> CLIF str
        self._patient_map: dict[int, str] = patient_map or {}
        self._hosp_map: dict[int, str] = hosp_map or {}
        # Reverse maps
        self._patient_rev: dict[str, int] = {v: k for k, v in self._patient_map.items()}
        self._hosp_rev: dict[str, int] = {v: k for k, v in self._hosp_map.items()}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def mimic_to_clif_patient(self, subject_id: int) -> str | None:
        return self._patient_map.get(subject_id)

    def clif_to_mimic_patient(self, patient_id: str) -> int | None:
        return self._patient_rev.get(patient_id)

    def mimic_to_clif_hosp(self, hadm_id: int) -> str | None:
        return self._hosp_map.get(hadm_id)

    def clif_to_mimic_hosp(self, hospitalization_id: str) -> int | None:
        return self._hosp_rev.get(hospitalization_id)

    @property
    def hadm_to_hosp(self) -> dict[int, str]:
        return dict(self._hosp_map)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_clif_tables(cls, hospitalization_df: pl.DataFrame) -> IDMapper:
        """Build mapping assuming CLIF IDs encode MIMIC IDs numerically.

        CLIF patient_id and hospitalization_id are strings. If they are
        numeric strings that match MIMIC subject_id/hadm_id, we can
        build the mapping directly.
        """
        unique = hospitalization_df.select("patient_id", "hospitalization_id").unique()

        # Build patient map: int(patient_id) -> str(patient_id)
        pids = unique["patient_id"].unique()
        try:
            pid_ints = pids.cast(pl.Int64)
            patient_map = dict(zip(pid_ints.to_list(), pids.cast(pl.Utf8).to_list()))
        except Exception:
            patient_map = {}

        # Build hosp map: int(hospitalization_id) -> str(hospitalization_id)
        hids = unique["hospitalization_id"].unique()
        try:
            hid_ints = hids.cast(pl.Int64)
            hosp_map = dict(zip(hid_ints.to_list(), hids.cast(pl.Utf8).to_list()))
        except Exception:
            hosp_map = {}

        return cls(patient_map=patient_map, hosp_map=hosp_map)

    @classmethod
    def from_json(cls, path: str | Path) -> IDMapper:
        data = json.loads(Path(path).read_text())
        patient_map = {int(k): v for k, v in data.get("patient_map", {}).items()}
        hosp_map = {int(k): v for k, v in data.get("hosp_map", {}).items()}
        return cls(patient_map=patient_map, hosp_map=hosp_map)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "patient_map": {str(k): v for k, v in self._patient_map.items()},
            "hosp_map": {str(k): v for k, v in self._hosp_map.items()},
        }
        path.write_text(json.dumps(data, indent=2))
