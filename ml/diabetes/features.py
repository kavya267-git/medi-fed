"""Diabetes feature schema (the eight PIMA Indians Diabetes Database measurements).

Standardisation uses fixed reference values: the mean and standard deviation of each measurement in the
published PIMA dataset (768 women; zeros that stand for "not measured" excluded). They are constants in the
code, so preprocessing is identical at every hospital and no hospital shares its own statistics.

In the PIMA file a zero in glucose, blood pressure, skin fold, insulin or BMI means "not measured". Data
preparation turns those zeros into gaps (``ZERO_MEANS_MISSING``), and each hospital fills gaps by KNN
imputation fitted on its own training split, as for EHR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class DiabetesFeature:
    name: str
    label: str
    unit: str
    ref_mean: float
    ref_std: float
    description: str


DIABETES_FEATURES: List[DiabetesFeature] = [
    DiabetesFeature("pregnancies", "Pregnancies", "count", 3.85, 3.37, "Number of times pregnant."),
    DiabetesFeature(
        "glucose", "Glucose (2-h OGTT)", "mg/dL", 121.69, 30.52, "Plasma glucose 2 hours into an oral glucose tolerance test."
    ),
    DiabetesFeature("diastolic_bp", "Diastolic blood pressure", "mmHg", 72.41, 12.37, "Diastolic blood pressure."),
    DiabetesFeature("skin_thickness", "Triceps skin fold", "mm", 29.15, 10.47, "Triceps skin-fold thickness."),
    DiabetesFeature("insulin", "Insulin (2-h serum)", "µU/mL", 155.55, 118.63, "2-hour serum insulin."),
    DiabetesFeature("bmi", "Body mass index", "kg/m²", 32.46, 6.92, "Weight in kg divided by height in m squared."),
    DiabetesFeature(
        "pedigree", "Diabetes pedigree function", "score", 0.47, 0.33, "Score of diabetes history in the family."
    ),
    DiabetesFeature("age", "Age", "years", 33.24, 11.75, "Age in years."),
]

DIABETES_FEATURE_NAMES: List[str] = [feature.name for feature in DIABETES_FEATURES]
DIABETES_LABEL_COLUMN = "label"
NUM_DIABETES_FEATURES = len(DIABETES_FEATURES)

#: Measurements where the PIMA file writes 0 for "not measured" (a true zero is impossible).
ZERO_MEANS_MISSING = ("glucose", "diastolic_bp", "skin_thickness", "insulin", "bmi")

#: (low, high) plausibility bounds per feature, to catch unit mix-ups (for example glucose in mmol/L).
#: Not clinical reference ranges.
DIABETES_BOUNDS: Dict[str, tuple] = {
    "pregnancies": (0, 25),
    "glucose": (30, 600),
    "diastolic_bp": (20, 160),
    "skin_thickness": (2, 120),
    "insulin": (2, 1200),
    "bmi": (10, 90),
    "pedigree": (0, 3),
    "age": (18, 120),
}

_REF_MEAN = np.array([f.ref_mean for f in DIABETES_FEATURES], dtype=np.float32)
_REF_STD = np.array([f.ref_std for f in DIABETES_FEATURES], dtype=np.float32)
#: (centre, spread) per feature; also used for imputation distances.
DIABETES_REFERENCE_SCALE = (_REF_MEAN, _REF_STD)


def standardize_diabetes(X: np.ndarray) -> np.ndarray:
    """Scale raw values by the fixed reference mean and spread."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != NUM_DIABETES_FEATURES:
        raise ValueError(f"Diabetes input must have shape [n, {NUM_DIABETES_FEATURES}], got {X.shape}")
    return np.clip((X - _REF_MEAN) / _REF_STD, -6.0, 6.0)


def zeros_to_missing(X: np.ndarray) -> np.ndarray:
    """Turn the PIMA convention (0 = not measured) into NaN gaps for the columns where 0 is impossible."""
    X = np.array(X, dtype=np.float64, copy=True)
    for name in ZERO_MEANS_MISSING:
        column = DIABETES_FEATURE_NAMES.index(name)
        X[X[:, column] == 0, column] = np.nan
    return X


def feature_metadata() -> List[dict]:
    return [{"name": f.name, "label": f.label, "unit": f.unit, "description": f.description} for f in DIABETES_FEATURES]
