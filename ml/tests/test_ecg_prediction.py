"""Labelling new heartbeats with the saved global ECG model, on the hospital's own computer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.common.types import TrainConfig
from ml.common.weights import weights_to_bytes
from ml.ecg.preprocess import ECG_BEAT_LENGTH
from ml.prediction.ecg import (
    CODES,
    ECGPredictor,
    describe_summary,
    ecg_model_available,
    load_ecg_model,
    read_beats,
)
from ml.prediction.local import DISCLAIMER, PredictionUnavailable
from ml.registry import get_adapter
from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
from ml.training.local_trainer import train_on_local_data
from ml.training.model_store import save_global_model


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("ecgpredict")
    generate_hospital_data(root, SIMULATED_HOSPITALS[0], seed=3, modalities=["ecg"])
    data_dir = root / SIMULATED_HOSPITALS[0].hospital_id
    adapter = get_adapter("ecg")
    weights = adapter.init_weights(0)
    weights = train_on_local_data(data_dir, "ecg", weights, TrainConfig(3, 32, 0.003, 0)).weights
    models_dir = root / "models"
    save_global_model(models_dir, "ecg", weights_to_bytes(weights),
                      {"architecture": adapter.architecture, "model_version": 2, "local_data_simulated": True})
    with np.load(data_dir / "ecg" / "val.npz") as z:
        beats, labels = z["signals"].astype(np.float32), z["labels"].astype(int)
    return models_dir, beats, labels, root


def test_no_model_yet(tmp_path):
    assert ecg_model_available(tmp_path) is None
    with pytest.raises(PredictionUnavailable, match="No global ECG model"):
        load_ecg_model(tmp_path)


def test_labels_beats_with_probabilities_and_explanation(setup):
    models_dir, beats, labels, _ = setup
    assert ecg_model_available(models_dir)["model_version"] == 2
    predictor = ECGPredictor(load_ecg_model(models_dir))
    results = predictor.predict_beats(beats[:5])
    assert len(results) == 5
    for r in results:
        assert r["code"] in CODES
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-4
        assert r["confidence"] == max(r["probabilities"].values())
        e = r["explanation"]
        assert 0 <= e["start_sample"] < e["end_sample"] <= ECG_BEAT_LENGTH
        assert len(e["importance"]) == ECG_BEAT_LENGTH


def test_scores_a_kaggle_style_csv_with_reference_labels(setup, tmp_path):
    models_dir, beats, labels, _ = setup
    path = tmp_path / "beats.csv"
    pd.DataFrame(np.column_stack([beats[:40], labels[:40]])).to_csv(path, header=False, index=False)
    X, y = read_beats(path)
    assert X.shape == (40, ECG_BEAT_LENGTH) and list(y) == list(labels[:40])
    table, summary = ECGPredictor(load_ecg_model(models_dir)).score(X, y)
    assert len(table) == 40 and set(table["prediction"]) <= set(CODES)
    assert sum(summary["counts"].values()) == 40
    assert 0 <= summary["agreement"] <= 1 and summary["reference_beats"] == 40
    assert summary["disclaimer"] == DISCLAIMER
    assert "beat(s)" in describe_summary(summary)


def test_reads_beats_without_labels_and_rejects_wrong_shapes(setup, tmp_path):
    _, beats, _, root = setup
    plain = tmp_path / "plain.csv"
    pd.DataFrame(beats[:3]).to_csv(plain, index=False)  # with a header row
    X, y = read_beats(plain)
    assert X.shape == (3, ECG_BEAT_LENGTH) and y is None
    X, y = read_beats(root / SIMULATED_HOSPITALS[0].hospital_id / "ecg" / "val.npz")
    assert y is not None and len(X) == len(y)
    bad = tmp_path / "bad.csv"
    pd.DataFrame(np.zeros((2, 50))).to_csv(bad, header=False, index=False)
    with pytest.raises(ValueError, match="187 samples"):
        read_beats(bad)
