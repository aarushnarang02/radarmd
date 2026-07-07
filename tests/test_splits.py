"""Patient-level splitting must never leak patients across folds."""

import pytest

from radarmd.data.labels import load_metadata
from radarmd.data.splits import (
    assert_no_patient_leakage,
    official_split,
    random_patient_split,
)


def test_random_split_no_leakage(metadata_csv):
    df = load_metadata(metadata_csv)
    splits = random_patient_split(df, seed=1)
    assert_no_patient_leakage(splits)  # raises on leak


def test_random_split_covers_all_images(metadata_csv):
    df = load_metadata(metadata_csv)
    splits = random_patient_split(df, seed=1)
    total = sum(len(p) for p in splits.values())
    assert total == len(df)


def test_random_split_deterministic(metadata_csv):
    df = load_metadata(metadata_csv)
    a = random_patient_split(df, seed=7)
    b = random_patient_split(df, seed=7)
    for k in a:
        assert list(a[k]["image"]) == list(b[k]["image"])


def test_random_split_seed_changes_partition(metadata_csv):
    df = load_metadata(metadata_csv)
    a = random_patient_split(df, seed=1)
    b = random_patient_split(df, seed=2)
    assert set(a["test"]["patient_id"]) != set(b["test"]["patient_id"])


def test_invalid_fractions_raise(metadata_csv):
    df = load_metadata(metadata_csv)
    with pytest.raises(ValueError):
        random_patient_split(df, val_frac=0.6, test_frac=0.6)


def test_leakage_detector_catches_injected_leak(metadata_csv):
    df = load_metadata(metadata_csv)
    splits = random_patient_split(df, seed=3)
    # Force a leak: copy one train patient's rows into val.
    leaked_pid = splits["train"]["patient_id"].iloc[0]
    splits["val"] = splits["val"].copy()
    splits["val"] = splits["train"][splits["train"]["patient_id"] == leaked_pid]
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(splits)


def test_official_split_no_leakage(tmp_path, metadata_csv):
    df = load_metadata(metadata_csv)
    tv_list = tmp_path / "train_val_list.txt"
    test_list = tmp_path / "test_list.txt"
    # Split by patient to keep the official lists themselves leak-free.
    test_patients = set(df["patient_id"].unique()[:8].tolist())
    tv_imgs = df[~df["patient_id"].isin(test_patients)]["image"]
    test_imgs = df[df["patient_id"].isin(test_patients)]["image"]
    tv_list.write_text("\n".join(tv_imgs))
    test_list.write_text("\n".join(test_imgs))

    splits = official_split(df, tv_list, test_list, seed=1)
    assert_no_patient_leakage(splits)
    assert len(splits["test"]) == len(test_imgs)
