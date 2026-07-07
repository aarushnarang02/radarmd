"""Label encoding and metadata loading."""

import numpy as np

from radarmd.data.constants import LABEL_TO_INDEX, NUM_CLASSES, PATHOLOGIES
from radarmd.data.labels import encode_labels, label_matrix, load_metadata


def test_encode_no_finding_is_all_zero():
    assert encode_labels("No Finding").sum() == 0.0
    assert encode_labels("").sum() == 0.0


def test_encode_single_and_multi_label():
    v = encode_labels("Cardiomegaly")
    assert v[LABEL_TO_INDEX["Cardiomegaly"]] == 1.0
    assert v.sum() == 1.0

    v2 = encode_labels("Cardiomegaly|Effusion")
    assert v2[LABEL_TO_INDEX["Cardiomegaly"]] == 1.0
    assert v2[LABEL_TO_INDEX["Effusion"]] == 1.0
    assert v2.sum() == 2.0


def test_encode_order_independent():
    a = encode_labels("Effusion|Cardiomegaly")
    b = encode_labels("Cardiomegaly|Effusion")
    assert np.array_equal(a, b)


def test_encode_unknown_token_ignored():
    v = encode_labels("Cardiomegaly|Bogus")
    assert v.sum() == 1.0


def test_encode_shape_and_dtype():
    v = encode_labels("Mass")
    assert v.shape == (NUM_CLASSES,)
    assert v.dtype == np.float32


def test_load_metadata_columns_and_types(metadata_csv):
    df = load_metadata(metadata_csv)
    for col in ("image", "patient_id", "labels", *PATHOLOGIES):
        assert col in df.columns
    assert df["patient_id"].dtype == int
    # multi-hot matrix matches the exploded label lists
    mat = label_matrix(df)
    assert mat.shape == (len(df), NUM_CLASSES)
    for i, labels in enumerate(df["labels"]):
        assert set(np.array(PATHOLOGIES)[mat[i].astype(bool)]) == set(labels)
