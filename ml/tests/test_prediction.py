"""Using the saved global model on new patients, on the hospital's own computer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.common.types import TrainConfig
from ml.common.weights import weights_to_bytes
from ml.prediction.local import (
    DISCLAIMER,
    LocalPredictor,
    PredictionUnavailable,
    available_models,
    describe_factors,
    load_local_model,
)
from ml.registry import get_adapter
from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
from ml.training.local_trainer import train_on_local_data
from ml.training.model_store import save_global_model

BACKEND = "numpy-reference"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("predict")
    generate_hospital_data(root, SIMULATED_HOSPITALS[0], seed=5, modalities=["diabetes", "ehr"])
    data_dir = root / SIMULATED_HOSPITALS[0].hospital_id
    adapter = get_adapter("diabetes", BACKEND)
    weights = adapter.init_weights(0)
    for _ in range(20):
        weights = train_on_local_data(data_dir, "diabetes", weights, TrainConfig(2, 32, 0.01, 0), BACKEND).weights
    models_dir = root / "models"
    save_global_model(
        models_dir, "diabetes", weights_to_bytes(weights), {"architecture": adapter.architecture, "model_version": 3}
    )
    return data_dir, models_dir


HIGH_RISK = {"pregnancies": 4, "glucose": 185, "diastolic_bp": 84, "skin_thickness": 35, "insulin": 240, "bmi": 41.0,
             "pedigree": 1.1, "age": 55}
LOW_RISK = {"pregnancies": 0, "glucose": 88, "diastolic_bp": 66, "skin_thickness": 18, "insulin": 60, "bmi": 22.0,
            "pedigree": 0.15, "age": 23}


def test_saved_models_are_listed_and_checked(setup):
    _, models_dir = setup
    assert set(available_models(models_dir)) == {"diabetes"}
    model = load_local_model(models_dir, "diabetes", BACKEND)
    assert model.info["model_version"] == 3
    with pytest.raises(PredictionUnavailable, match="No global ehr model"):
        load_local_model(models_dir, "ehr", BACKEND)
    with pytest.raises(PredictionUnavailable, match="tabular|available for"):
        load_local_model(models_dir, "ecg", BACKEND)


def test_tampered_or_mismatched_model_is_refused(setup, tmp_path):
    _, models_dir = setup
    copy = tmp_path / "models" / "diabetes"
    copy.mkdir(parents=True)
    payload = (models_dir / "diabetes" / "global_latest.npz").read_bytes()
    (copy / "global_latest.npz").write_bytes(payload + b"x")
    (copy / "global_latest.json").write_text((models_dir / "diabetes" / "global_latest.json").read_text())
    with pytest.raises(PredictionUnavailable, match="SHA-256"):
        load_local_model(tmp_path / "models", "diabetes", BACKEND)
    # An EHR model saved under the diabetes name has the wrong architecture.
    ehr = get_adapter("ehr", BACKEND)
    save_global_model(tmp_path / "other", "diabetes", weights_to_bytes(ehr.init_weights(0)), {"architecture": "x"})
    with pytest.raises(PredictionUnavailable, match="does not match"):
        load_local_model(tmp_path / "other", "diabetes", BACKEND)


def test_predict_one_patient_with_reasons(setup):
    data_dir, models_dir = setup
    predictor = LocalPredictor(load_local_model(models_dir, "diabetes", BACKEND), data_dir)
    high = predictor.predict_one(HIGH_RISK)
    low = predictor.predict_one(LOW_RISK)
    assert high["probability"] > low["probability"]
    assert high["positive"] and high["prediction"] == "Diabetes"
    assert not low["positive"] and low["prediction"] == "No diabetes"
    assert high["disclaimer"] == DISCLAIMER and "not a medical device" in DISCLAIMER
    assert len(high["factors"]) == 8 and high["factors"][0]["direction"] in ("raises", "lowers")
    top = {f["feature"] for f in high["factors"][:3] if f["direction"] == "raises"}
    assert "glucose" in top
    # The explanation adds up to the model's output (Shapley efficiency, up to sampling of orderings).
    assert high["explanation_method"].startswith("Shapley") and "training records" in high["explanation_method"]


def test_missing_values_are_estimated_from_local_records(setup):
    data_dir, models_dir = setup
    predictor = LocalPredictor(load_local_model(models_dir, "diabetes", BACKEND), data_dir)
    partial = {k: v for k, v in HIGH_RISK.items() if k not in ("insulin", "skin_thickness")}
    result = predictor.predict_one(partial)
    estimated = {i["feature"] for i in result["inputs"] if i["estimated"]}
    assert estimated == {"insulin", "skin_thickness"}
    assert all(np.isfinite(i["value"]) for i in result["inputs"])
    # Without the data folder there is nothing to estimate from: the operator is told what to fill in.
    bare = LocalPredictor(load_local_model(models_dir, "diabetes", BACKEND))
    with pytest.raises(ValueError, match="Missing value"):
        bare.predict_one(partial)
    assert bare.predict_one(HIGH_RISK)["factors"] == []  # no background sample: prediction without reasons


def test_wrong_units_and_unknown_inputs_are_refused(setup):
    data_dir, models_dir = setup
    predictor = LocalPredictor(load_local_model(models_dir, "diabetes", BACKEND), data_dir)
    with pytest.raises(ValueError, match="outside .* check the units"):
        predictor.predict_one({**HIGH_RISK, "glucose": 9.5})  # mmol/L instead of mg/dL
    with pytest.raises(ValueError, match="Unknown input"):
        predictor.predict_one({**HIGH_RISK, "cholesterol": 200})
    with pytest.raises(ValueError, match="must be a number"):
        predictor.predict_one({**HIGH_RISK, "age": "fifty"})


def test_score_a_table(setup):
    data_dir, models_dir = setup
    predictor = LocalPredictor(load_local_model(models_dir, "diabetes", BACKEND), data_dir)
    frame = pd.DataFrame([HIGH_RISK, LOW_RISK, {**LOW_RISK, "insulin": None}, {**LOW_RISK, "glucose": 5}])
    frame.insert(0, "patient", ["p1", "p2", "p3", "p4"])
    scored = predictor.predict_frame(frame)
    assert list(scored["patient"]) == ["p1", "p2", "p3", "p4"]
    assert scored.loc[0, "prediction"] == "Diabetes" and scored.loc[1, "prediction"] == "No diabetes"
    assert scored.loc[2, "estimated_inputs"] == "Insulin (2-h serum)"
    assert "raises risk" in scored.loc[0, "main_reasons"]
    assert scored.loc[3, "prediction"] == "" and "outside" in scored.loc[3, "problem"]
    with pytest.raises(ValueError, match="missing column"):
        predictor.predict_frame(frame.drop(columns=["bmi"]))


def test_describe_factors():
    factors = [
        {"label": "Glucose (2-h OGTT)", "value": 168.0, "unit": "mg/dL", "direction": "raises", "estimated": False},
        {"label": "Pregnancies", "value": 0.0, "unit": "count", "direction": "lowers", "estimated": False},
        {"label": "Insulin (2-h serum)", "value": 90.24, "unit": "µU/mL", "direction": "neutral", "estimated": True},
    ]
    assert describe_factors(factors) == "Glucose (2-h OGTT) 168 mg/dL raises risk; Pregnancies 0 lowers risk"


def test_saved_model_info_is_json(setup):
    _, models_dir = setup
    info = json.loads((models_dir / "diabetes" / "global_latest.json").read_text())
    assert info["modality"] == "diabetes" and len(info["sha256"]) == 64
