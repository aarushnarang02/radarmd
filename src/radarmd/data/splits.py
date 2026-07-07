"""Patient-level train/val/test splitting.

ChestX-ray14 has ~112k images from ~30k patients, with many patients
contributing multiple studies. Splitting by *image* leaks the same patient's
anatomy across train and test and inflates AUROC. Every split here is by
``patient_id`` so a patient appears in exactly one fold.

Two split strategies:
  - ``random_patient_split``: the CheXNet-style random 70/10/20 patient split
    (headline benchmark, comparable to the ~0.84 mean AUROC figure).
  - ``official_split``: the NIH-provided ``train_val_list.txt`` /
    ``test_list.txt`` partition (harder; reported for transparency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def random_patient_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split ``df`` into train/val/test by unique patient_id.

    Fractions are over *patients*, not images. Deterministic given ``seed``.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in (0,1) and sum to < 1")

    patients = np.array(sorted(df["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_ids = set(patients[:n_test].tolist())
    val_ids = set(patients[n_test : n_test + n_val].tolist())
    train_ids = set(patients[n_test + n_val :].tolist())

    return {
        "train": df[df["patient_id"].isin(train_ids)].reset_index(drop=True),
        "val": df[df["patient_id"].isin(val_ids)].reset_index(drop=True),
        "test": df[df["patient_id"].isin(test_ids)].reset_index(drop=True),
    }


def _read_image_list(path: str | Path) -> set[str]:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def official_split(
    df: pd.DataFrame,
    train_val_list: str | Path,
    test_list: str | Path,
    val_frac: float = 0.125,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Use NIH's official train_val / test image lists.

    The official partition is already patient-disjoint. We carve a validation
    fold out of ``train_val_list`` *by patient* so it stays leak-free.
    ``val_frac`` is the fraction of train_val patients held out (0.125 of the
    train_val set ≈ 10% of the full dataset, matching common practice).
    """
    train_val_images = _read_image_list(train_val_list)
    test_images = _read_image_list(test_list)

    tv = df[df["image"].isin(train_val_images)]
    test = df[df["image"].isin(test_images)].reset_index(drop=True)

    tv_patients = np.array(sorted(tv["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(tv_patients)
    n_val = int(round(len(tv_patients) * val_frac))
    val_ids = set(tv_patients[:n_val].tolist())

    val = tv[tv["patient_id"].isin(val_ids)].reset_index(drop=True)
    train = tv[~tv["patient_id"].isin(val_ids)].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}


def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any patient_id appears in more than one split."""
    seen: dict[int, str] = {}
    for name, part in splits.items():
        for pid in part["patient_id"].unique():
            if pid in seen:
                raise AssertionError(
                    f"Patient {pid} leaks across splits '{seen[pid]}' and '{name}'"
                )
            seen[pid] = name
