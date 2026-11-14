"""Matching a hospital's own export to a tabular model's inputs (EHR, diabetes), on the hospital's computer.

Every hospital names its columns differently ("Age", "AGE_YRS", "pt_age"; "Pulse", "HR", "heart_rate_bpm"), so
the agent does not guess silently. ``suggest_mapping`` proposes a column for each model input from common names
and LOINC codes in the headings; the person checks every match. ``inspect_export`` reports what the export
looks like (text values to translate, values whose units look wrong). ``apply_mapping`` turns the export into
the model's columns with the conversions the person accepted, and ``import_export`` writes the training and
held-out files the node reads. ``ColumnMapping`` is saved, so the next export with the same headings imports in
one step.

Nothing here leaves the computer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ml.common.datasets import save_local_dataset
from ml.common.real_data import split_train_val
from ml.common.tabular import tabular_schema

NOT_RECORDED = None

#: Common headings per input, compared after lower-casing and dropping everything but letters and digits.
SYNONYMS: Dict[str, Dict[str, Sequence[str]]] = {
    "ehr": {
        "age": ("age", "ageyears", "ageyrs", "ptage", "patientage", "anchorage", "ageatadmission"),
        "sex_male": ("sexmale", "sex", "gender", "male", "ptsex", "patientsex"),
        "heart_rate": ("heartrate", "hr", "pulse", "pulserate", "heartratebpm", "hrbpm"),
        "systolic_bp": ("systolicbp", "sbp", "systolic", "sysbp", "bpsystolic", "systolicbloodpressure"),
        "resp_rate": ("resprate", "rr", "respiratoryrate", "respiration", "breathsperminute"),
        "temperature_c": ("temperaturec", "temperature", "temp", "tempc", "bodytemp", "temperaturef", "tempf"),
        "spo2": ("spo2", "sp02", "o2sat", "oxygensaturation", "sao2", "saturation"),
        "wbc": ("wbc", "whitebloodcells", "wbccount", "tlc", "leukocytes", "totalleucocytecount"),
        "creatinine": ("creatinine", "creat", "cr", "serumcreatinine", "screatinine"),
        "lactate": ("lactate", "lacticacid", "lact"),
        "glucose": ("glucose", "glu", "bloodglucose", "rbs", "bloodsugar"),
        "hemoglobin": ("hemoglobin", "haemoglobin", "hb", "hgb"),
        "prior_admissions_12m": ("prioradmissions12m", "prioradmissions", "previousadmissions", "admissions12m"),
        "comorbidity_index": ("comorbidityindex", "charlson", "cci", "charlsonindex", "comorbidityscore"),
    },
    "diabetes": {
        "pregnancies": ("pregnancies", "preg", "pregnant", "numpregnancies", "timespregnant", "gravida"),
        "glucose": ("glucose", "plas", "plasmaglucose", "ogtt", "glucose2h", "glu", "bloodglucose"),
        "diastolic_bp": ("diastolicbp", "pres", "bloodpressure", "dbp", "diastolic", "bpdiastolic"),
        "skin_thickness": ("skinthickness", "skin", "triceps", "tricepsskinfold", "tsf", "skinfold"),
        "insulin": ("insulin", "insu", "seruminsulin", "insulin2h"),
        "bmi": ("bmi", "mass", "bodymassindex"),
        "pedigree": ("pedigree", "pedi", "diabetespedigreefunction", "dpf", "pedigreefunction"),
        "age": ("age", "ageyears", "ageyrs", "patientage"),
    },
}
LABEL_SYNONYMS = (
    "label", "outcome", "target", "class", "diabetes", "diagnosis", "died", "death", "mortality", "expired",
    "hospitalexpireflag", "adverseoutcome",
)
GROUP_SYNONYMS = ("patientid", "subjectid", "mrn", "uhid", "patient", "ptid", "hadmid")

#: LOINC codes that identify an input for certain when a heading contains them.
LOINC: Dict[str, Dict[str, Sequence[str]]] = {
    "ehr": {
        "age": ("30525-0",),
        "heart_rate": ("8867-4",),
        "systolic_bp": ("8480-6",),
        "resp_rate": ("9279-1",),
        "temperature_c": ("8310-5",),
        "spo2": ("59408-5", "2708-6"),
        "wbc": ("6690-2",),
        "creatinine": ("2160-0",),
        "lactate": ("2524-7", "32693-4"),
        "glucose": ("2345-7", "2339-0"),
        "hemoglobin": ("718-7",),
    },
    "diabetes": {
        "age": ("30525-0",),
        "glucose": ("1518-0", "2345-7"),
        "diastolic_bp": ("8462-4",),
        "insulin": ("20448-7",),
        "bmi": ("39156-5",),
    },
}

#: Text values read as 1 ("yes") for the binary columns.
POSITIVE_WORDS = {"1", "yes", "y", "true", "positive", "testedpositive", "died", "dead", "death", "expired",
                  "deceased", "adverse", "diabetic", "diabetes"}
MALE_WORDS = {"1", "m", "male", "man"}

#: Unit conversions offered when a column's typical value says the units differ from the model's.
#: (feature, test on the median, factor/function, description)
CONVERSIONS = {
    "temperature_f_to_c": ("temperature_c", lambda m: m > 50, lambda x: (x - 32.0) * 5.0 / 9.0, "°F → °C"),
    "glucose_mmol_to_mg": ("glucose", lambda m: m < 35, lambda x: x * 18.016, "mmol/L → mg/dL"),
    "creatinine_umol_to_mg": ("creatinine", lambda m: m > 30, lambda x: x / 88.42, "µmol/L → mg/dL"),
    "hemoglobin_gl_to_gdl": ("hemoglobin", lambda m: m > 30, lambda x: x / 10.0, "g/L → g/dL"),
    "spo2_fraction_to_percent": ("spo2", lambda m: m <= 1.0, lambda x: x * 100.0, "fraction → %"),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def headings_signature(columns: Iterable[str]) -> str:
    """A short fingerprint of an export's headings, to recognise the same export layout next time."""
    return hashlib.sha256("\n".join(str(c) for c in columns).encode("utf-8")).hexdigest()[:16]


@dataclass
class ColumnMapping:
    """Which export column feeds each input, and how its values are translated. Saved for the next import."""

    modality: str
    #: model input -> export column, or None when the hospital does not record it
    features: Dict[str, Optional[str]]
    label: Optional[str] = None
    #: optional patient identifier: keeps one patient's rows on one side of the train / held-out split
    group: Optional[str] = None
    #: text values that mean 1 in the label (and, for EHR, values that mean male in the sex column)
    label_positive: List[str] = field(default_factory=list)
    male_values: List[str] = field(default_factory=list)
    #: accepted unit conversions (keys of CONVERSIONS)
    conversions: List[str] = field(default_factory=list)
    #: diabetes: a 0 in glucose, blood pressure, skin fold, insulin or BMI means "not measured" (PIMA convention)
    zero_means_missing: bool = False
    headings: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ColumnMapping":
        data = json.loads(text)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def not_recorded(self) -> List[str]:
        return [name for name, column in self.features.items() if column is None]


def _match(columns: Sequence[str], names: Sequence[str], codes: Sequence[str], taken: set) -> Optional[str]:
    normalised = {column: _norm(column) for column in columns if column not in taken}
    for column, key in normalised.items():  # exact common name first
        if key in names:
            return column
    for column in normalised:  # then a LOINC code anywhere in the heading
        if any(code in str(column) for code in codes):
            return column
    return None


def suggest_mapping(modality: str, columns: Sequence[str]) -> ColumnMapping:
    """A first guess for every input. The person checks each one; nothing is accepted silently."""
    schema = tabular_schema(modality)
    columns = [str(c) for c in columns]
    taken: set = set()
    label = _match(columns, LABEL_SYNONYMS, (), taken)
    if label:
        taken.add(label)
    group = _match(columns, GROUP_SYNONYMS, (), taken)
    if group:
        taken.add(group)
    features: Dict[str, Optional[str]] = {}
    for name in schema.feature_names:
        chosen = _match(columns, SYNONYMS[modality].get(name, (name,)), LOINC[modality].get(name, ()), taken)
        features[name] = chosen
        if chosen:
            taken.add(chosen)
    return ColumnMapping(modality, features, label=label, group=group, headings=headings_signature(columns))


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _text_values(series: pd.Series) -> List[str]:
    """Distinct non-numeric values of a column, most frequent first (at most 12)."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    text = values[pd.to_numeric(values, errors="coerce").isna()]
    if text.empty:
        return []
    return list(values.value_counts().index[:12])


def inspect_export(frame: pd.DataFrame, mapping: ColumnMapping) -> dict:
    """What the person needs to decide before importing: text values to translate and unit conversions."""
    report: dict = {"rows": int(len(frame)), "label_values": [], "sex_values": [], "conversions": [], "empty": []}
    if mapping.label and mapping.label in frame:
        report["label_values"] = _text_values(frame[mapping.label]) or sorted(
            str(v) for v in frame[mapping.label].dropna().unique()[:12]
        )
    sex = mapping.features.get("sex_male")
    if sex and sex in frame:
        report["sex_values"] = _text_values(frame[sex])
    for key, (feature, test, _fn, description) in CONVERSIONS.items():
        column = mapping.features.get(feature)
        if not column or column not in frame:
            continue
        values = _numeric(frame[column]).dropna()
        if mapping.zero_means_missing:
            values = values[values != 0]
        if len(values) and test(float(values.median())):
            report["conversions"].append(
                {"key": key, "feature": feature, "column": column, "median": float(values.median()), "description": description}
            )
    for name, column in mapping.features.items():
        if column and column in frame and _numeric(frame[column]).notna().sum() == 0 and name != "sex_male":
            report["empty"].append(name)
    return report


def _binary(series: pd.Series, positive: Iterable[str], default_words: set) -> pd.Series:
    """0 / 1 from numbers or text; text in ``positive`` (or, if none given, a common yes-word) is 1."""
    chosen = {str(v).strip().lower() for v in positive}
    numbers = _numeric(series)

    def convert(value):
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        text = str(value).strip().lower()
        if chosen:
            return 1.0 if text in chosen else 0.0
        number = pd.to_numeric(text, errors="coerce")
        if not pd.isna(number):
            return float(number)
        return 1.0 if _norm(text) in default_words else 0.0

    if numbers.notna().sum() == series.notna().sum() and not chosen:
        return numbers
    return series.map(convert)


def apply_mapping(
    frame: pd.DataFrame, mapping: ColumnMapping, with_label: bool = True, fill_not_recorded: bool = True
) -> tuple:
    """The export as the model's columns (plus ``label`` when ``with_label``), and a report of what changed.

    An input the hospital does not record is filled with its reference value for every row when
    ``fill_not_recorded`` (training: the model then gets no information from it; the report lists these), or left
    as gaps (prediction: the node estimates them from its own records). Values that are not numbers become gaps."""
    schema = tabular_schema(mapping.modality)
    needed = [c for c in mapping.features.values() if c]
    if with_label and mapping.label:
        needed.append(mapping.label)
    missing = [c for c in needed if c not in frame]
    if missing:
        raise ValueError(f"The file has no column(s): {', '.join(missing)}")
    if with_label and not mapping.label:
        raise ValueError("Choose the column that holds the outcome (the label).")
    out = pd.DataFrame(index=frame.index)
    report: dict = {"filled_with_reference": [], "converted": [], "not_numbers": {}}
    centre, _spread = schema.reference_scale
    for i, name in enumerate(schema.feature_names):
        column = mapping.features.get(name)
        if column is None:
            if fill_not_recorded:
                out[name] = float(centre[i])
                report["filled_with_reference"].append(name)
            else:
                out[name] = np.nan
            continue
        if name == "sex_male":
            values = _binary(frame[column], mapping.male_values, MALE_WORDS)
        else:
            values = _numeric(frame[column])
            bad = int((values.isna() & frame[column].notna() & (frame[column].astype(str).str.strip() != "")).sum())
            if bad:
                report["not_numbers"][name] = bad
        if mapping.zero_means_missing and mapping.modality == "diabetes":
            from ml.diabetes.features import ZERO_MEANS_MISSING

            if name in ZERO_MEANS_MISSING:
                values = values.mask(values == 0)
        for key in mapping.conversions:
            feature, _test, convert, description = CONVERSIONS[key]
            if feature == name:
                values = convert(values)
                report["converted"].append(f"{name}: {description}")
        out[name] = values.astype(float)
    if with_label:
        out[schema.label_column] = _binary(frame[mapping.label], mapping.label_positive, POSITIVE_WORDS)
    return out, report


def import_export(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    data_dir: Path,
    source_name: str = "export",
    val_fraction: float = 0.25,
    seed: int = 0,
) -> dict:
    """Write ``<data_dir>/<modality>/train.csv`` and ``val.csv`` (75% / 25%, one patient per side) from an export."""
    schema = tabular_schema(mapping.modality)
    table, report = apply_mapping(frame, mapping)
    keep = table[schema.label_column].notna()
    dropped = int((~keep).sum())
    table = table[keep]
    labels = table[schema.label_column].to_numpy()
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("The outcome must be yes / no. Tick which values of the outcome column mean 'yes'.")
    if mapping.group and mapping.group in frame:
        groups = frame.loc[table.index, mapping.group].astype(str).to_numpy()
    else:
        groups = np.arange(len(table))
    train, val = split_train_val(groups, np.random.default_rng(seed), val_fraction)
    X = table[schema.feature_names].to_numpy(dtype=np.float32)
    y = labels.astype(np.int64)
    notes = [f"Imported from {source_name} with the hospital's column mapping"]
    if dropped:
        notes.append(f"{dropped} row(s) without an outcome dropped")
    if report["filled_with_reference"]:
        notes.append("not recorded (filled with reference values): " + ", ".join(report["filled_with_reference"]))
    if report["converted"]:
        notes.append("converted: " + "; ".join(report["converted"]))
    manifest = save_local_dataset(
        Path(data_dir), mapping.modality, train=(X[train], y[train]), val=(X[val], y[val]), source="hospital-export",
        description=". ".join(notes) + ".",
    )
    return {"manifest": manifest, "report": report, "dropped_without_label": dropped}


# --------------------------------------------------------------------------- saved mappings
def save_mapping(folder: Path, mapping: ColumnMapping) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{mapping.modality}.json"
    path.write_text(mapping.to_json(), encoding="utf-8")
    return path


def load_mapping(folder: Path, modality: str, columns: Optional[Sequence[str]] = None) -> Optional[ColumnMapping]:
    """The saved mapping for ``modality``; with ``columns``, only if it was made for the same headings."""
    path = Path(folder) / f"{modality}.json"
    if not path.exists():
        return None
    try:
        mapping = ColumnMapping.from_json(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None
    if columns is not None:
        present = set(map(str, columns))
        used = [c for c in mapping.features.values() if c]
        if not set(used) <= present:
            return None
    return mapping
