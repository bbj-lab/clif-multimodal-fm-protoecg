"""MIMIC-required label stubs (14 labels). Not computable from CLIF alone."""

from __future__ import annotations

import sys

from clif_protoecg.labels.base import label

_MIMIC_LABELS = [
    ("sepsis3", "LABEL//sepsis3", "Sepsis-3 onset"),
    ("aki_kdigo_1", "LABEL//aki_kdigo_1", "AKI KDIGO Stage 1"),
    ("aki_kdigo_2", "LABEL//aki_kdigo_2", "AKI KDIGO Stage 2"),
    ("aki_kdigo_3", "LABEL//aki_kdigo_3", "AKI KDIGO Stage 3"),
    ("qsofa_2", "LABEL//qsofa_2", "qSOFA >= 2"),
    ("cardiac_arrest", "LABEL//cardiac_arrest", "Cardiac arrest"),
    ("transfusion", "LABEL//transfusion", "Blood product transfusion"),
    ("dialysis", "LABEL//dialysis", "Dialysis (any)"),
    ("central_line", "LABEL//central_line", "Central line insertion"),
    ("charlson_high", "LABEL//charlson_high", "Charlson >= 3"),
    ("elixhauser_high", "LABEL//elixhauser_high", "Elixhauser >= 5"),
    ("phecode_chf", "LABEL//phecode_chf", "CHF"),
    ("phecode_copd", "LABEL//phecode_copd", "COPD"),
    ("phecode_diabetes", "LABEL//phecode_diabetes", "Diabetes"),
    ("phecode_ckd", "LABEL//phecode_ckd", "CKD"),
    ("phecode_cancer", "LABEL//phecode_cancer", "Cancer"),
]

_module = sys.modules[__name__]


def _make_stub(name: str, token: str, desc: str) -> None:
    @label(
        name=name,
        token=token,
        category="mimic_required",
        description=f"{desc} (requires MIMIC data)",
        source_tables=[],
    )
    def _stub(hosp_row: dict, **kw) -> dict | None:
        return None

    fn_name = f"compute_{name}"
    _stub.__name__ = fn_name
    _stub.__qualname__ = fn_name
    # Expose on module so pickle can find the function by qualname
    setattr(_module, fn_name, _stub)


for _name, _token, _desc in _MIMIC_LABELS:
    _make_stub(_name, _token, _desc)
