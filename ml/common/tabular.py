"""Schemas of the tabular modalities (EHR and diabetes): column names, label column and reference scale.

Code that handles any table-shaped modality (loading, checking, imputation, explanations) looks the details
up here instead of assuming the 14 EHR features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from ml.diabetes.features import (
    DIABETES_FEATURE_NAMES,
    DIABETES_FEATURES,
    DIABETES_LABEL_COLUMN,
    DIABETES_REFERENCE_SCALE,
    standardize_diabetes,
)
from ml.ehr.features import EHR_FEATURE_NAMES, EHR_FEATURES, EHR_LABEL_COLUMN, EHR_REFERENCE_SCALE, standardize_ehr


@dataclass(frozen=True)
class TabularSchema:
    modality: str
    feature_names: List[str]
    label_column: str
    #: (centre, spread) per feature, fixed constants.
    reference_scale: tuple
    standardize: Callable[[np.ndarray], np.ndarray]
    features: list

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def feature_metadata(self) -> List[dict]:
        return [{"name": f.name, "label": f.label, "unit": f.unit, "description": f.description} for f in self.features]


TABULAR_SCHEMAS: Dict[str, TabularSchema] = {
    "ehr": TabularSchema("ehr", EHR_FEATURE_NAMES, EHR_LABEL_COLUMN, EHR_REFERENCE_SCALE, standardize_ehr, EHR_FEATURES),
    "diabetes": TabularSchema(
        "diabetes",
        DIABETES_FEATURE_NAMES,
        DIABETES_LABEL_COLUMN,
        DIABETES_REFERENCE_SCALE,
        standardize_diabetes,
        DIABETES_FEATURES,
    ),
}
TABULAR_MODALITIES = tuple(TABULAR_SCHEMAS)


def is_tabular(modality: str) -> bool:
    return modality in TABULAR_SCHEMAS


def tabular_schema(modality: str) -> TabularSchema:
    try:
        return TABULAR_SCHEMAS[modality]
    except KeyError as exc:
        raise ValueError(f"'{modality}' is not a tabular modality; expected one of {TABULAR_MODALITIES}") from exc
