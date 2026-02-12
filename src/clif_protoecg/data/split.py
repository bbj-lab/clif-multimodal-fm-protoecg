"""Patient-level train / val / test splitting."""

from __future__ import annotations

import numpy as np
import polars as pl

from clif_protoecg.core.artifacts import SplitInfo


def create_patient_splits(
    hospitalization_df: pl.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> SplitInfo:
    """Chronological patient-level split.

    Strategy:
    - Sort patients by their earliest admission_dttm
    - First ~77% → train, next ~10% → val, final ~13% → test
    - All hospitalizations for a patient go into the same split

    This approximates the reference repo's anchor_year_group approach
    without requiring MIMIC postgres access.
    """
    # Earliest admission per patient
    earliest = (
        hospitalization_df
        .group_by("patient_id")
        .agg(pl.col("admission_dttm").min().alias("first_admit"))
        .sort("first_admit")
    )

    patient_ids = earliest["patient_id"].to_list()
    n = len(patient_ids)

    # Determine split boundaries
    test_frac = 0.13
    train_frac = 1.0 - test_frac - val_fraction

    n_train = int(n * train_frac)
    n_val = int(n * val_fraction)

    # Slight shuffle within chronological order to avoid bias at boundaries
    rng = np.random.default_rng(seed)
    # Only shuffle within each split bucket, not across splits
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train : n_train + n_val]
    test_ids = patient_ids[n_train + n_val :]

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return SplitInfo(
        train=[str(pid) for pid in train_ids],
        val=[str(pid) for pid in val_ids],
        test=[str(pid) for pid in test_ids],
    )
