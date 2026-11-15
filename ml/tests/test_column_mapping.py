"""A hospital's own export, matched to the model's inputs on its computer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.common.column_mapping import (
    ColumnMapping,
    apply_mapping,
    import_export,
    inspect_export,
    load_mapping,
    save_mapping,
    suggest_mapping,
)
from ml.common.data_check import check_modality
from ml.common.datasets import load_local_dataset
from ml.diabetes.features import DIABETES_FEATURE_NAMES
from ml.simulation.diabetes import DiabetesProfile, simulate_diabetes
from ml.simulation.ehr import simulate_ehr


def messy_ehr(n: int = 120) -> pd.DataFrame:
    """An EHR export with its own headings, °F temperature, mmol/L glucose, text sex and outcome, no lactate."""
    X, y = simulate_ehr(n, np.random.default_rng(3))
    frame = pd.DataFrame(
        {
            "MRN": [f"P{i // 2:04d}" for i in range(n)],  # two admissions per patient
            "Patient Age": X[:, 0],
            "Gender": np.where(X[:, 1] == 1, "M", "F"),
            "Pulse": X[:, 2],
            "SBP": X[:, 3],
            "RR": X[:, 4],
            "Temp (F)": np.round(X[:, 5] * 9 / 5 + 32, 1),
            "O2 Sat": X[:, 6],
            "TLC": X[:, 7],
            "Creatinine 2160-0": X[:, 8],
            "Blood sugar": np.round(X[:, 10] / 18.016, 1),
            "Hb": X[:, 11],
            "Previous admissions": X[:, 12],
            "Charlson": X[:, 13],
            "Outcome": np.where(y == 1, "Died", "Discharged"),
            "Ward": "Medicine",
        }
    )
    return frame


def test_suggestions_use_common_names_and_loinc_codes():
    mapping = suggest_mapping("ehr", messy_ehr().columns)
    assert mapping.features["age"] == "Patient Age"
    assert mapping.features["sex_male"] == "Gender"
    assert mapping.features["heart_rate"] == "Pulse"
    assert mapping.features["temperature_c"] == "Temp (F)"
    assert mapping.features["creatinine"] == "Creatinine 2160-0"  # by its LOINC code
    assert mapping.features["glucose"] == "Blood sugar"
    assert mapping.features["lactate"] is None  # not recorded: left for the person, never invented
    assert mapping.label == "Outcome" and mapping.group == "MRN"
    # A heading is used for at most one input.
    used = [c for c in mapping.features.values() if c]
    assert len(used) == len(set(used))


def test_inspect_finds_text_values_and_unit_problems():
    frame = messy_ehr()
    mapping = suggest_mapping("ehr", frame.columns)
    mapping.features["spo2"] = "O2 Sat"
    report = inspect_export(frame, mapping)
    assert set(report["label_values"]) == {"Died", "Discharged"}
    assert set(report["sex_values"]) == {"M", "F"}
    found = {c["key"] for c in report["conversions"]}
    assert found == {"temperature_f_to_c", "glucose_mmol_to_mg"}


def test_import_writes_files_that_pass_the_local_check(tmp_path):
    frame = messy_ehr()
    mapping = suggest_mapping("ehr", frame.columns)
    mapping.features["spo2"] = "O2 Sat"
    mapping.label_positive = ["Died"]
    mapping.male_values = ["M"]
    mapping.conversions = ["temperature_f_to_c", "glucose_mmol_to_mg"]
    result = import_export(frame, mapping, tmp_path / "data", source_name="his_export.csv")
    assert "lactate" in result["report"]["filled_with_reference"]
    check = check_modality(tmp_path / "data", "ehr")
    assert check["passed"], check["problems"]
    train = load_local_dataset(tmp_path / "data", "ehr", "train")
    val = load_local_dataset(tmp_path / "data", "ehr", "val")
    assert len(train) + len(val) == len(frame)
    temperature = np.concatenate([train.X[:, 5], val.X[:, 5]])
    assert 34 < temperature.mean() < 40  # converted to °C
    assert set(np.unique(np.concatenate([train.y, val.y]))) == {0, 1}
    assert "not recorded" in result["manifest"]["description"] and "°F → °C" in result["manifest"]["description"]


def test_without_conversion_the_check_catches_wrong_units(tmp_path):
    frame = messy_ehr()
    mapping = suggest_mapping("ehr", frame.columns)
    mapping.features["spo2"] = "O2 Sat"
    import_export(frame, mapping, tmp_path / "data")
    check = check_modality(tmp_path / "data", "ehr")
    assert not check["passed"]
    assert any("temperature_c" in p for p in check["problems"])


def test_patients_stay_on_one_side_of_the_split(tmp_path):
    frame = messy_ehr(80)
    mapping = suggest_mapping("ehr", frame.columns)
    mapping.features["spo2"] = "O2 Sat"
    mapping.conversions = ["temperature_f_to_c", "glucose_mmol_to_mg"]
    # Each patient (MRN) has two admissions; with the MRN as the group both land on the same side.
    import_export(frame, mapping, tmp_path / "x")
    train = pd.read_csv(tmp_path / "x" / "ehr" / "train.csv")
    val = pd.read_csv(tmp_path / "x" / "ehr" / "val.csv")
    assert len(train) % 2 == 0 and len(val) % 2 == 0


def test_diabetes_export_with_pima_zeros(tmp_path):
    X, y = simulate_diabetes(100, np.random.default_rng(1), DiabetesProfile(unmeasured=0.0))
    frame = pd.DataFrame(X, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                                     "DiabetesPedigreeFunction", "Age"])
    frame.loc[:9, "Insulin"] = 0
    frame["Outcome"] = y
    mapping = suggest_mapping("diabetes", frame.columns)
    assert mapping.features == dict(zip(DIABETES_FEATURE_NAMES, frame.columns[:8]))
    mapping.zero_means_missing = True
    table, _ = apply_mapping(frame, mapping)
    assert table["insulin"].isna().sum() == 10 and (table["insulin"] != 0).all()
    import_export(frame, mapping, tmp_path / "d")
    assert check_modality(tmp_path / "d", "diabetes")["passed"]


def test_prediction_mapping_leaves_unrecorded_inputs_as_gaps():
    frame = messy_ehr(10).drop(columns=["Outcome"])
    mapping = suggest_mapping("ehr", frame.columns)
    table, report = apply_mapping(frame, mapping, with_label=False, fill_not_recorded=False)
    assert table["lactate"].isna().all() and report["filled_with_reference"] == []
    assert "label" not in table


def test_missing_columns_and_label_are_reported():
    frame = messy_ehr(10)
    mapping = suggest_mapping("ehr", frame.columns)
    with pytest.raises(ValueError, match="no column"):
        apply_mapping(frame.drop(columns=["Pulse"]), mapping)
    mapping.label = None
    with pytest.raises(ValueError, match="outcome"):
        apply_mapping(frame, mapping)


def test_saved_mapping_is_reused_only_for_matching_headings(tmp_path):
    frame = messy_ehr(10)
    mapping = suggest_mapping("ehr", frame.columns)
    mapping.male_values = ["M"]
    save_mapping(tmp_path, mapping)
    again = load_mapping(tmp_path, "ehr", frame.columns)
    assert isinstance(again, ColumnMapping) and again.male_values == ["M"] and again.features == mapping.features
    assert load_mapping(tmp_path, "ehr", ["something", "else"]) is None
    assert load_mapping(tmp_path, "diabetes") is None


def test_word_pieces_and_spelling_variants_are_matched_and_marked_for_checking():
    columns = ["Patient_ID", "AGE_YRS", "Sex", "HR_BPM", "BP Systolic (mmHg)", "Resp. Rate", "Body Temp", "SpO2 %",
               "WBC x10^9/L", "S.Creatnine", "Lactic Acid", "Blood Glucose (mg/dL)", "HGB g/dL", "CCI", "Mortality"]
    mapping = suggest_mapping("ehr", columns)
    expected = {"age": "AGE_YRS", "heart_rate": "HR_BPM", "systolic_bp": "BP Systolic (mmHg)",
                "resp_rate": "Resp. Rate", "temperature_c": "Body Temp", "spo2": "SpO2 %", "wbc": "WBC x10^9/L",
                "creatinine": "S.Creatnine", "lactate": "Lactic Acid", "glucose": "Blood Glucose (mg/dL)",
                "hemoglobin": "HGB g/dL", "comorbidity_index": "CCI"}
    for name, column in expected.items():
        assert mapping.features[name] == column, name
    assert mapping.label == "Mortality" and mapping.group == "Patient_ID"
    assert mapping.reasons["heart_rate"] == "matched by name"
    assert "check" in mapping.reasons["creatinine"]  # a spelling variant is never presented as certain


def test_coded_headings_are_guessed_from_their_values():
    frame = messy_ehr(200).rename(columns={"Pulse": "obs_01", "SBP": "obs_02", "RR": "obs_03", "O2 Sat": "obs_04",
                                           "Hb": "obs_05"})
    mapping = suggest_mapping("ehr", frame.columns, frame)
    assert mapping.features["heart_rate"] == "obs_01" and mapping.features["systolic_bp"] == "obs_02"
    assert mapping.features["resp_rate"] == "obs_03" and mapping.features["spo2"] == "obs_04"
    assert mapping.features["hemoglobin"] == "obs_05"
    assert all(mapping.reasons[n].startswith("guessed from the values") for n in
               ("heart_rate", "systolic_bp", "resp_rate", "spo2", "hemoglobin"))
    # Without the values there is nothing to go on: those inputs stay unmatched rather than guessed.
    blind = suggest_mapping("ehr", frame.columns)
    assert blind.features["heart_rate"] is None


def test_strong_matches_win_over_weak_ones():
    # "Glucose tolerance" only contains a common word; "Glucose" is an exact name and must get the input.
    mapping = suggest_mapping("diabetes", ["Glucose tolerance note", "Glucose", "BMI", "Outcome"])
    assert mapping.features["glucose"] == "Glucose" and mapping.reasons["glucose"] == "matched by name"


def test_hospitals_with_fewer_inputs_can_join_and_say_what_they_record(tmp_path):
    from ml.common.datasets import local_data_summary

    frame = messy_ehr(160)
    frame["Hb"] = np.where(np.arange(len(frame)) % 5 == 0, frame["Hb"], np.nan)  # haemoglobin in 20% of rows only
    mapping = suggest_mapping("ehr", frame.columns, frame)
    mapping.conversions = ["temperature_f_to_c", "glucose_mmol_to_mg"]
    report = inspect_export(frame, mapping)
    assert report["mostly_empty"] == {"hemoglobin": pytest.approx(0.2)}
    result = import_export(frame, mapping, tmp_path / "h")
    assert set(result["report"]["filled_with_reference"]) == {"lactate", "hemoglobin"}
    check = check_modality(tmp_path / "h", "ehr")
    assert check["passed"], check["problems"]
    assert check["coverage"] == {"inputs_total": 14, "inputs_recorded": 12, "not_recorded": ["lactate", "hemoglobin"]}
    summary = local_data_summary(tmp_path / "h")["ehr"]
    assert summary["inputs_recorded"] == 12 and summary["not_recorded"] == ["lactate", "hemoglobin"]


def test_full_records_report_full_coverage(tmp_path):
    X, y = simulate_diabetes(80, np.random.default_rng(2), DiabetesProfile(unmeasured=0.3))
    frame = pd.DataFrame(X, columns=DIABETES_FEATURE_NAMES)
    frame["label"] = y
    mapping = suggest_mapping("diabetes", frame.columns)
    import_export(frame, mapping, tmp_path / "d")
    assert check_modality(tmp_path / "d", "diabetes")["coverage"]["not_recorded"] == []


def test_impossible_zeros_are_detected_in_diabetes_exports():
    frame = pd.DataFrame({"Glucose": [0, 120, 140, 99], "BMI": [30, 0, 28, 33], "Age": [40, 50, 60, 33],
                          "Outcome": [0, 1, 1, 0]})
    report = inspect_export(frame, suggest_mapping("diabetes", frame.columns))
    assert report["impossible_zeros"] == ["glucose", "bmi"]
