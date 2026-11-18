"""Evaluating a saved model on labelled records from a hospital with its own headings and units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.common.column_mapping import inspect_export, suggest_mapping
from ml.common.types import TrainConfig
from ml.common.weights import weights_to_bytes
from ml.diabetes.features import DIABETES_FEATURE_NAMES
from ml.prediction.evaluate import (
    binary_metrics,
    bootstrap_ci,
    duplicate_mask,
    evaluate_labelled,
    labelled_table,
    overlap_mask,
)
from ml.prediction.local import LocalPredictor, PredictionUnavailable, load_model_file
from ml.registry import get_adapter
from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
from ml.training.local_trainer import train_on_local_data
from ml.training.model_store import save_global_model

BACKEND = "numpy-reference"
#: SIMULATED test records with a hospital's own headings, glucose in mmol/L and 0 for "not measured".
RENAMED = {
    "pregnancies": "No. of pregnancies",
    "glucose": "Plasma glucose (mmol/L)",
    "diastolic_bp": "DBP",
    "skin_thickness": "Triceps skinfold (mm)",
    "insulin": "Serum insulin",
    "bmi": "Body mass index",
    "pedigree": "DPF",
    "age": "Age (years)",
}


def synthetic_export(n: int = 60, seed: int = 3) -> pd.DataFrame:
    """SIMULATED: a small labelled export, diabetes more likely with high glucose and BMI."""
    rng = np.random.default_rng(seed)
    glucose_mg = rng.normal(125, 30, n).clip(60, 220)
    bmi = rng.normal(32, 6, n).clip(18, 55)
    risk = (glucose_mg - 125) / 30 + (bmi - 32) / 6
    diabetic = (risk + rng.normal(0, 0.6, n) > 0.4).astype(int)
    frame = pd.DataFrame({
        RENAMED["pregnancies"]: rng.integers(0, 8, n),
        RENAMED["glucose"]: np.round(glucose_mg / 18.016, 1),
        RENAMED["diastolic_bp"]: rng.normal(72, 10, n).round(),
        RENAMED["skin_thickness"]: rng.normal(29, 8, n).round(),
        RENAMED["insulin"]: rng.normal(150, 60, n).clip(15, 600).round(),
        RENAMED["bmi"]: bmi.round(1),
        RENAMED["pedigree"]: rng.uniform(0.1, 1.5, n).round(3),
        RENAMED["age"]: rng.integers(21, 70, n),
        "Diabetic": np.where(diabetic == 1, "yes", "no"),
    })
    frame.loc[:5, RENAMED["insulin"]] = 0  # "not measured", as in the PIMA convention
    return frame


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("external")
    generate_hospital_data(root, SIMULATED_HOSPITALS[0], seed=5, modalities=["diabetes"])
    data_dir = root / SIMULATED_HOSPITALS[0].hospital_id
    adapter = get_adapter("diabetes", BACKEND)
    weights = adapter.init_weights(0)
    for _ in range(15):
        weights = train_on_local_data(data_dir, "diabetes", weights, TrainConfig(2, 32, 0.01, 0), BACKEND).weights
    models_dir = root / "models"
    save_global_model(models_dir, "diabetes", weights_to_bytes(weights), {"architecture": adapter.architecture,
                                                                          "model_version": 4})
    return load_model_file(models_dir / "diabetes" / "global_latest.npz", "diabetes", BACKEND), data_dir


def mapped(frame: pd.DataFrame):
    mapping = suggest_mapping("diabetes", list(frame.columns), frame)
    report = inspect_export(frame, mapping)
    mapping.label = "Diabetic"
    mapping.conversions = [c["key"] for c in report["conversions"]]
    mapping.zero_means_missing = bool(report["impossible_zeros"])
    return mapping, report


def test_mapping_matches_renamed_headers_and_converts_mmol_glucose():
    frame = synthetic_export()
    mapping, report = mapped(frame)
    assert mapping.features == {name: RENAMED[name] for name in DIABETES_FEATURE_NAMES}
    assert [c["key"] for c in report["conversions"]] == ["glucose_mmol_to_mg"]
    assert report["impossible_zeros"] == ["insulin"]
    X, y, table_report = labelled_table(frame, mapping)
    glucose = DIABETES_FEATURE_NAMES.index("glucose")
    insulin = DIABETES_FEATURE_NAMES.index("insulin")
    assert np.allclose(X[:, glucose], frame[RENAMED["glucose"]] * 18.016)  # now mg/dL
    assert np.isnan(X[:6, insulin]).all() and not np.isnan(X[6:, insulin]).any()  # 0 -> not measured
    assert y.tolist() == (frame["Diabetic"] == "yes").astype(int).tolist()  # "yes" read as 1
    assert table_report["converted"] == ["glucose: mmol/L → mg/dL"]


def test_metrics_match_a_hand_count():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.6, 0.2, 0.1, 0.7, 0.3, 0.4, 0.5])  # 0.5 counts as positive (threshold >= 0.5)
    m = binary_metrics(y, p)
    assert m["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["sensitivity"] == pytest.approx(3 / 4) and m["specificity"] == pytest.approx(3 / 4)
    assert m["f1_positive"] == pytest.approx(0.75) and m["macro_f1"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(11 / 16)  # 11 of 16 (yes, no) pairs ranked the right way
    majority = binary_metrics(y, np.zeros(len(y)))  # always "no"
    assert majority["sensitivity"] == 0 and majority["specificity"] == 1 and majority["roc_auc"] == 0.5
    low, high = bootstrap_ci(y, p, "accuracy", n_resamples=200)
    assert 0 <= low <= m["accuracy"] <= high <= 1


def test_evaluate_a_renamed_mmol_file_with_the_saved_model(model):
    local_model, data_dir = model
    frame = synthetic_export()
    mapping, _ = mapped(frame)
    result = evaluate_labelled(LocalPredictor(local_model, data_dir), frame, mapping)
    assert result["n"] == 60 and result["positives"] + result["negatives"] == 60
    c = result["confusion"]
    assert c["tp"] + c["fn"] == result["positives"] and c["tn"] + c["fp"] == result["negatives"]
    assert result["accuracy"] == pytest.approx((c["tp"] + c["tn"]) / 60)
    assert 0 <= result["roc_auc"] <= 1
    assert result["rows_with_estimated_values"] == 6
    assert "training records" in result["gaps_filled"]  # filled from the hospital's own records
    assert result["model"]["version"] == 4 and result["small_sample"]
    # Without a data folder the gaps are filled from the evaluated records' measurements, never their answers.
    alone = evaluate_labelled(LocalPredictor(local_model, None), frame, mapping)
    assert "not their answers" in alone["gaps_filled"]
    # A wrong conversion (mmol/L glucose read as mg/dL) is caught as out of range, not silently used.
    mapping.conversions = []
    _, _, report = labelled_table(frame, mapping)
    assert report["out_of_bounds_treated_as_missing"].get("glucose", 0) > 0


def test_overlap_and_repeats_are_found_with_gaps_equal_to_gaps():
    reference = np.array([[1.0, 120.0, np.nan], [2.0, 90.0, 30.0]])
    X = np.array([[1.0, 120.0, np.nan], [1.0, 120.0, 0.0], [2.0, 90.0, 30.0004], [5.0, 99.0, 30.0], [5.0, 99.0, 30.0]])
    assert overlap_mask(X, reference).tolist() == [True, False, True, False, False]
    assert duplicate_mask(X, np.array([0, 0, 1, 1, 1])).tolist() == [False, False, False, False, True]
    assert duplicate_mask(X, np.array([0, 0, 1, 1, 0])).tolist() == [False] * 5  # different answer: not a repeat


def test_a_model_file_must_match_its_recorded_hash(model, tmp_path):
    local_model, _ = model
    folder = tmp_path / "diabetes"
    save_global_model(tmp_path, "diabetes", weights_to_bytes(local_model.weights), {"model_version": 1})
    weights_file = folder / "global_latest.npz"
    assert load_model_file(weights_file, "diabetes", BACKEND).info["model_version"] == 1
    payload = bytearray(weights_file.read_bytes())
    payload[len(payload) // 2] ^= 0xFF  # one byte changed
    weights_file.write_bytes(bytes(payload))
    with pytest.raises(PredictionUnavailable, match="SHA-256"):
        load_model_file(weights_file, "diabetes", BACKEND)
    with pytest.raises(PredictionUnavailable, match="both needed"):
        load_model_file(tmp_path / "nothing.npz", "diabetes", BACKEND)
