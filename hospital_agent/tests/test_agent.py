"""MediFed Hospital Agent: token storage, local data check, sample data, take-part switch, no token in logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "hospital_agent")]

from medifed_agent import settings  # noqa: E402
from medifed_agent.session import AgentError, AgentSession  # noqa: E402

TOKEN = "secret-node-token-123"


@pytest.fixture(autouse=True)
def agent_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIFED_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEDIFED_AGENT_NO_KEYRING", "1")
    return tmp_path / "home"


def test_settings_never_hold_the_token(agent_home):
    settings.save_settings({"server": "https://x.example", "hospital_id": "h", "token": TOKEN})
    text = (agent_home / "settings.json").read_text(encoding="utf-8")
    assert TOKEN not in text and json.loads(text)["hospital_id"] == "h"
    assert "token" not in settings.load_settings()


def test_token_fallback_file_round_trip(agent_home):
    where = settings.save_token("https://x.example", TOKEN)
    assert where == "a file only you can read"
    assert settings.load_token("https://x.example") == TOKEN
    assert settings.load_token("https://other.example") is None  # bound to its server
    assert TOKEN not in (agent_home / "settings.json").read_text(encoding="utf-8") if (agent_home / "settings.json").exists() else True
    settings.forget_token("https://x.example")
    assert settings.load_token("https://x.example") is None


def test_identify_refuses_bad_input_before_any_request():
    session = AgentSession(log=lambda line: None)
    with pytest.raises(AgentError, match="https://"):
        session.identify("ftp://x", TOKEN)
    with pytest.raises(AgentError, match="token"):
        session.identify("https://x.example", "  ")


def test_sample_data_is_simulated_and_passes_the_local_check(agent_home):
    session = AgentSession(log=lambda line: None)
    session.profile = {"hospital_id": "hospital_test_city_hospital", "name": "Test City Hospital", "declared_modalities": ["ecg"]}
    folder = session.make_sample_data()
    assert folder.is_relative_to(agent_home)
    manifest = json.loads((folder / "ecg" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["simulated"] is True
    results = session.check_data(folder)
    assert list(results) == ["ecg"]  # only what the hospital declared
    assert results["ecg"]["passed"] and results["ecg"]["train_records"] > 0
    assert not (folder / "ehr").exists() and not (folder / "xray").exists()


class FakeTransport:
    def __init__(self, participating=True):
        self.calls = []
        self.participating = participating

    def profile(self, hospital_id):
        return {"hospital_id": hospital_id, "name": "Test City Hospital", "registered": True, "declared_modalities": ["ecg"]}

    def report_data_check(self, hospital_id, modalities):
        self.calls.append(("check", modalities))
        return {"invited_modalities": ["ecg"]}

    def pending(self, hospital_id):
        return {"round_ids": ["ecg-r0001"], "participating": self.participating}

    def report_failure(self, round_id, hospital_id, stage, message):
        self.calls.append(("failure", round_id, stage))


def _runner(tmp_path, lines):
    from ml.simulation.hospitals import generate_hospital_data, sample_hospital
    from ml.training.node_runner import NodeRunner

    hospital = sample_hospital("hospital_test_city_hospital", "Test City Hospital")
    generate_hospital_data(tmp_path / "data", hospital, modalities=["ecg"])
    runner = NodeRunner("hospital_test_city_hospital", "https://x.example", TOKEN, tmp_path / "data" / hospital.hospital_id, log=lines.append)
    runner.transport = FakeTransport()
    return runner


def test_data_check_report_and_logs_never_carry_the_token(tmp_path):
    lines = []
    runner = _runner(tmp_path, lines)
    from ml.common.datasets import local_data_summary

    runner._check_and_report(local_data_summary(runner.data_dir))
    kind, report = runner.transport.calls[0]
    assert kind == "check" and report["ecg"]["passed"] is True
    assert set(report["ecg"]) == {"passed", "train_records", "val_records", "label_counts", "problems", "coverage"}
    assert report["ecg"]["coverage"] is None  # only tabular data has input coverage
    assert runner.name == "Test City Hospital"
    assert lines and not any(TOKEN in line for line in lines)


def test_take_part_switch_off_declines_rounds(tmp_path):
    lines = []
    runner = _runner(tmp_path, lines)
    runner.local_participation = False
    runner.poll_once()
    assert ("failure", "ecg-r0001", "declined") in runner.transport.calls


def test_predict_with_the_saved_global_model_on_this_computer(agent_home, tmp_path, monkeypatch):
    """The agent predicts with the model it kept from a round, fills gaps from its own records, sends nothing."""
    from ml.common.types import TrainConfig
    from ml.common.weights import weights_to_bytes
    from ml.registry import get_adapter
    from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
    from ml.training.local_trainer import train_on_local_data
    from ml.training.model_store import save_global_model

    monkeypatch.setenv("ML_BACKEND", "numpy-reference")
    generate_hospital_data(tmp_path / "data", SIMULATED_HOSPITALS[1], seed=2, modalities=["diabetes"])
    folder = tmp_path / "data" / SIMULATED_HOSPITALS[1].hospital_id
    session = AgentSession(log=lambda line: None)
    assert session.saved_models() == {}
    with pytest.raises(AgentError, match="No global diabetes model"):
        session.predictor("diabetes", folder)

    adapter = get_adapter("diabetes", "numpy-reference")
    weights = train_on_local_data(folder, "diabetes", adapter.init_weights(0), TrainConfig(2, 32, 0.01, 0), "numpy-reference").weights
    save_global_model(session.models_dir, "diabetes", weights_to_bytes(weights), {"architecture": adapter.architecture})
    assert set(session.saved_models()) == {"diabetes"}
    predictor = session.predictor("diabetes", folder)
    assert session.predictor("diabetes", folder) is predictor  # reused until the model or folder changes
    result = predictor.predict_one({"glucose": 150, "bmi": 33, "age": 45, "pregnancies": 1, "pedigree": 0.5, "diastolic_bp": 75})
    assert 0 <= result["probability"] <= 1 and {i["feature"] for i in result["inputs"] if i["estimated"]} == {
        "skin_thickness",
        "insulin",
    }


def test_label_heartbeats_with_the_saved_global_ecg_model(agent_home, tmp_path, monkeypatch):
    """The agent labels a file of heartbeats with the ECG model it kept from a round, on this computer."""
    import numpy as np

    from ml.common.types import TrainConfig
    from ml.common.weights import weights_to_bytes
    from ml.prediction.ecg import CODES, read_beats
    from ml.registry import get_adapter
    from ml.simulation.hospitals import SIMULATED_HOSPITALS, generate_hospital_data
    from ml.training.local_trainer import train_on_local_data
    from ml.training.model_store import save_global_model

    monkeypatch.setenv("ML_BACKEND", "numpy-reference")
    generate_hospital_data(tmp_path / "data", SIMULATED_HOSPITALS[0], seed=4, modalities=["ecg"])
    folder = tmp_path / "data" / SIMULATED_HOSPITALS[0].hospital_id
    session = AgentSession(log=lambda line: None)
    assert session.ecg_model_info() is None
    with pytest.raises(AgentError, match="No global ECG model"):
        session.ecg_predictor()

    adapter = get_adapter("ecg", "numpy-reference")
    weights = train_on_local_data(folder, "ecg", adapter.init_weights(0), TrainConfig(2, 32, 0.01, 0), "numpy-reference").weights
    save_global_model(session.models_dir, "ecg", weights_to_bytes(weights), {"architecture": adapter.architecture, "model_version": 1})
    assert session.ecg_model_info()["model_version"] == 1
    predictor = session.ecg_predictor()
    assert session.ecg_predictor() is predictor
    beats, labels = read_beats(folder / "ecg" / "val.npz")
    table, summary = predictor.score(beats[:20], labels[:20])
    assert len(table) == 20 and set(table["prediction"]) <= set(CODES)
    assert summary["beats"] == 20 and "agreement" in summary
    assert np.isfinite(table["confidence"]).all()


def test_import_an_export_with_own_headings_then_score_new_patients(agent_home, tmp_path, monkeypatch):
    """Import with the hospital's own column names; later files with the same headings are scored the same way."""
    import numpy as np
    import pandas as pd

    from ml.common.column_mapping import suggest_mapping
    from ml.common.types import TrainConfig
    from ml.common.weights import weights_to_bytes
    from ml.registry import get_adapter
    from ml.simulation.diabetes import DiabetesProfile, simulate_diabetes
    from ml.training.local_trainer import train_on_local_data
    from ml.training.model_store import save_global_model

    monkeypatch.setenv("ML_BACKEND", "numpy-reference")
    X, y = simulate_diabetes(160, np.random.default_rng(0), DiabetesProfile(unmeasured=0.0))
    headings = ["Preg", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
    export = pd.DataFrame(X, columns=headings)
    export["Result"] = np.where(y == 1, "tested_positive", "tested_negative")
    session = AgentSession(log=lambda line: None)
    assert session.importable_modalities() == ["ehr", "diabetes"]

    mapping = suggest_mapping("diabetes", export.columns)
    mapping.label = "Result"
    mapping.label_positive = ["tested_positive"]
    result = session.import_export(export, mapping, None, "export.csv")
    assert result["check"]["passed"], result["check"]["problems"]
    assert result["train"] + result["val"] == 160 and (agent_home / "mappings" / "diabetes.json").exists()

    folder = result["folder"]
    adapter = get_adapter("diabetes", "numpy-reference")
    weights = train_on_local_data(folder, "diabetes", adapter.init_weights(0), TrainConfig(2, 32, 0.01, 0), "numpy-reference").weights
    save_global_model(session.models_dir, "diabetes", weights_to_bytes(weights), {"architecture": adapter.architecture})

    new_patients = pd.DataFrame(X[:5], columns=headings).drop(columns=["Insulin"])  # not measured this time
    new_patients.insert(0, "Patient", [f"N{i}" for i in range(5)])
    scored = session.score_frame("diabetes", folder, new_patients)
    assert list(scored["Patient"]) == [f"N{i}" for i in range(5)]
    assert scored["probability"].between(0, 1).all() and (scored["estimated_inputs"].str.contains("Insulin")).all()
    with pytest.raises(AgentError, match="Import data"):
        session.score_frame("diabetes", folder, pd.DataFrame({"x": [1.0]}))


def test_evaluate_the_global_model_on_own_labelled_records(agent_home, tmp_path, monkeypatch):
    """Evaluate with a labelled file: scores computed on this computer; the import matching is not overwritten."""
    import numpy as np
    import pandas as pd

    from ml.common.column_mapping import save_mapping, suggest_mapping
    from ml.common.weights import weights_to_bytes
    from ml.registry import get_adapter
    from ml.simulation.diabetes import DiabetesProfile, simulate_diabetes
    from ml.training.model_store import save_global_model

    monkeypatch.setenv("ML_BACKEND", "numpy-reference")
    session = AgentSession(log=lambda line: None)
    adapter = get_adapter("diabetes", "numpy-reference")
    save_global_model(session.models_dir, "diabetes", weights_to_bytes(adapter.init_weights(0)),
                      {"architecture": adapter.architecture, "model_version": 2})
    X, y = simulate_diabetes(40, np.random.default_rng(1), DiabetesProfile(unmeasured=0.0))  # SIMULATED
    labelled = pd.DataFrame(X, columns=["Preg", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"])
    labelled["Truth"] = np.where(y == 1, "Positive", "Negative")
    mapping = suggest_mapping("diabetes", labelled.columns, labelled)
    saved = save_mapping(session.mappings_dir, suggest_mapping("diabetes", ["Glucose"]))
    before = saved.read_text(encoding="utf-8")
    with pytest.raises(AgentError, match="true answer"):
        session.evaluate_frame("diabetes", None, labelled, mapping)
    mapping.label, mapping.label_positive = "Truth", ["Positive"]
    result = session.evaluate_frame("diabetes", None, labelled, mapping)
    assert result["n"] == 40 and result["positives"] == int(y.sum())
    c = result["confusion"]
    assert c["tp"] + c["fn"] + c["tn"] + c["fp"] == 40 and result["model"]["version"] == 2
    assert saved.read_text(encoding="utf-8") == before
