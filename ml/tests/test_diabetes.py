"""Diabetes modality: schema, simulated and prepared data, the local check, imputation and training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.common.data_check import check_modality
from ml.common.datasets import load_local_dataset
from ml.common.real_data import prepare_diabetes_csv, read_pima_arff
from ml.common.tabular import tabular_schema
from ml.common.types import TrainConfig
from ml.diabetes.features import (
    DIABETES_FEATURE_NAMES,
    NUM_DIABETES_FEATURES,
    standardize_diabetes,
    zeros_to_missing,
)
from ml.registry import get_adapter
from ml.simulation.diabetes import DiabetesProfile, simulate_diabetes
from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data, generate_reference_cases
from ml.training.local_trainer import load_training_splits, train_on_local_data


def test_schema_and_standardisation():
    schema = tabular_schema("diabetes")
    assert schema.feature_names == DIABETES_FEATURE_NAMES and NUM_DIABETES_FEATURES == 8
    assert schema.label_column == "label"
    centre, spread = schema.reference_scale
    assert np.allclose(standardize_diabetes(centre.reshape(1, -1)), 0.0)
    with pytest.raises(ValueError):
        standardize_diabetes(np.zeros((2, 14)))


def test_zeros_mean_missing_only_where_zero_is_impossible():
    row = np.zeros((1, 8))
    out = zeros_to_missing(row)[0]
    gaps = {name for name, value in zip(DIABETES_FEATURE_NAMES, out) if np.isnan(value)}
    assert gaps == {"glucose", "diastolic_bp", "skin_thickness", "insulin", "bmi"}
    assert out[DIABETES_FEATURE_NAMES.index("pregnancies")] == 0


def test_simulated_records_are_plausible_and_imbalanced():
    X, y = simulate_diabetes(1500, np.random.default_rng(0), DiabetesProfile())
    assert X.shape == (1500, 8) and set(np.unique(y)) == {0, 1}
    assert 0.2 < y.mean() < 0.45
    glucose = X[:, DIABETES_FEATURE_NAMES.index("glucose")]
    assert glucose[y == 1].mean() > glucose[y == 0].mean() + 10
    # Insulin and skin fold are sometimes unmeasured; the rest never are.
    missing = np.isnan(X).mean(axis=0)
    assert missing[DIABETES_FEATURE_NAMES.index("insulin")] > 0.1 and missing[DIABETES_FEATURE_NAMES.index("glucose")] == 0
    cases = generate_reference_cases("diabetes")
    assert not np.isnan(cases["X"]).any() and not np.isnan(cases["background"]).any()


@pytest.fixture(scope="module")
def hospital_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("diabetes")
    generate_hospital_data(root, SIMULATED_HOSPITALS[0], seed=4, modalities=["diabetes"])
    return root / SIMULATED_HOSPITALS[0].hospital_id


def test_simulated_hospital_passes_the_local_check(hospital_dir):
    result = check_modality(hospital_dir, "diabetes")
    assert result["passed"], result["problems"]
    assert result["train_records"] > result["val_records"] > 0
    assert set(result["label_counts"]) == {"no diabetes", "diabetes"}


def test_gaps_are_filled_on_the_node_and_training_learns(hospital_dir):
    train, val, report = load_training_splits(hospital_dir, "diabetes")
    assert report and report["imputation"]["method"] == "k-nearest neighbours"
    assert not np.isnan(train.X).any() and not np.isnan(val.X).any()

    adapter = get_adapter("diabetes")
    weights = adapter.init_weights(0)
    config = TrainConfig(local_epochs=2, batch_size=32, learning_rate=0.01, seed=0)
    for _ in range(15):
        weights = train_on_local_data(hospital_dir, "diabetes", weights, config).weights
    probabilities = adapter.predict_proba(weights, val.X)[:, 1]
    assert probabilities.shape == (len(val),) and np.all((0 <= probabilities) & (probabilities <= 1))
    # Better than chance at ranking held-out patients (simulated data with a clear glucose effect).
    positives, negatives = probabilities[val.y == 1], probabilities[val.y == 0]
    auc = (positives[:, None] > negatives[None, :]).mean()
    assert auc > 0.7


def test_out_of_range_units_fail_the_check(tmp_path):
    frame = pd.DataFrame(
        np.tile([2, 7.1, 70, 25, 100, 30, 0.4, 40], (40, 1)), columns=DIABETES_FEATURE_NAMES
    )  # glucose 7.1 looks like mmol/L, not mg/dL
    frame["label"] = [0, 1] * 20
    csv = tmp_path / "patients.csv"
    frame.to_csv(csv, index=False)
    prepare_diabetes_csv(csv, tmp_path / "out", hospital_ids=["h1"], alpha=None)
    result = check_modality(tmp_path / "out" / "h1", "diabetes")
    assert not result["passed"]
    assert any("'glucose' outside" in problem for problem in result["problems"])


def test_prepare_csv_with_column_mapping_and_pima_zeros(tmp_path):
    rng = np.random.default_rng(1)
    X, y = simulate_diabetes(80, rng, DiabetesProfile(unmeasured=0.0))
    frame = pd.DataFrame(X, columns=[name.upper() for name in DIABETES_FEATURE_NAMES])
    frame.loc[:4, "INSULIN"] = 0  # PIMA-style "not measured"
    frame["Outcome"] = y
    csv = tmp_path / "export.csv"
    frame.to_csv(csv, index=False)
    mapping = {name.upper(): name for name in DIABETES_FEATURE_NAMES}
    manifests = prepare_diabetes_csv(
        csv, tmp_path / "out", hospital_ids=["h1", "h2"], column_mapping=mapping, label_column="Outcome",
        zero_means_missing=True, alpha=None,
    )
    assert set(manifests) == {"h1", "h2"}
    loaded = [load_local_dataset(tmp_path / "out" / h, "diabetes", s, allow_missing_features=True) for h in ("h1", "h2") for s in ("train", "val")]
    assert sum(len(d) for d in loaded) == 80
    insulin = np.concatenate([d.X[:, DIABETES_FEATURE_NAMES.index("insulin")] for d in loaded])
    assert np.isnan(insulin).sum() == 5 and not (insulin == 0).any()


def test_read_pima_arff(tmp_path):
    rows = ["6,148,72,35,0,33.6,0.627,50,tested_positive", "1,85,66,29,0,26.6,0.351,31,tested_negative"]
    path = tmp_path / "diabetes.arff"
    path.write_text("% header\n@relation pima\n@data\n" + "\n".join(rows) + "\n", encoding="utf-8")
    X, y = read_pima_arff(path)
    assert y.tolist() == [1, 0] and X.shape == (2, 8)
    assert np.isnan(X[:, DIABETES_FEATURE_NAMES.index("insulin")]).all()
    assert X[0, DIABETES_FEATURE_NAMES.index("glucose")] == 148
