"""Using the global model on new patients, inside the hospital.

After each round a hospital node keeps the aggregated model (``ml.training.model_store``). This module loads that
file and predicts for new patients of a tabular modality (EHR adverse outcome, diabetes), with a Shapley-value
explanation of each prediction. Everything runs on the hospital's own computer: the patient's values, the
prediction and the explanation are never sent anywhere. Gaps in a patient's values are filled from the hospital's
own training records (KNN imputation), and the explanation's background sample is drawn from those records too.

This is a research prototype, not a medical device; every result carries ``DISCLAIMER``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analytics.explainability.service import explain_ehr_record
from ml.common.data_check import BOUNDS
from ml.common.datasets import DatasetUnavailableError, load_local_dataset
from ml.common.tabular import TABULAR_MODALITIES, is_tabular, tabular_schema
from ml.common.weights import weights_from_bytes
from ml.ehr.imputation import EHRImputer
from ml.registry import get_adapter
from ml.training.model_store import LATEST

PREDICTABLE_MODALITIES = TABULAR_MODALITIES
THRESHOLD = 0.5
DISCLAIMER = (
    "Research prototype, not a medical device. Not for diagnosis or treatment decisions; "
    "a clinician must interpret every result."
)
#: Rows of the hospital's training records used as the explanation background, and permutations per patient.
BACKGROUND_ROWS = 32
N_PERMUTATIONS = 24
#: CSV scoring explains at most this many rows (each explanation costs a few thousand model evaluations).
MAX_EXPLAINED_ROWS = 500


class PredictionUnavailable(RuntimeError):
    """A prediction cannot be made; the message says why and what to do."""


@dataclass
class LocalModel:
    modality: str
    weights: dict
    info: dict
    adapter: object = field(repr=False)


def model_files(models_dir: Path, modality: str) -> tuple:
    folder = Path(models_dir) / modality
    return folder / f"{LATEST}.npz", folder / f"{LATEST}.json"


def available_models(models_dir: Optional[Path]) -> Dict[str, dict]:
    """Saved global models this computer can predict with: ``{modality: info}``."""
    found = {}
    if not models_dir:
        return found
    for modality in PREDICTABLE_MODALITIES:
        weights_file, info_file = model_files(models_dir, modality)
        if weights_file.exists() and info_file.exists():
            try:
                found[modality] = json.loads(info_file.read_text(encoding="utf-8"))
            except ValueError:
                continue
    return found


def load_local_model(models_dir: Path, modality: str, backend: Optional[str] = None) -> LocalModel:
    """The latest global model saved on this computer, checked against its recorded hash and architecture."""
    if not is_tabular(modality):
        raise PredictionUnavailable(f"Prediction is available for {', '.join(PREDICTABLE_MODALITIES)} only.")
    weights_file, info_file = model_files(models_dir, modality)
    if not weights_file.exists() or not info_file.exists():
        raise PredictionUnavailable(
            f"No global {modality} model on this computer yet. Connect and take part in a {modality} round first; "
            "the agent keeps the model each round produces."
        )
    return load_model_file(weights_file, modality, backend)


def load_model_file(weights_file: Path, modality: str, backend: Optional[str] = None) -> LocalModel:
    """A saved model file (``<name>.npz`` beside its ``<name>.json``), checked against the SHA-256 recorded in
    the JSON and against this code's model architecture (its signature)."""
    if not is_tabular(modality):
        raise PredictionUnavailable(f"Prediction is available for {', '.join(PREDICTABLE_MODALITIES)} only.")
    weights_file = Path(weights_file)
    info_file = weights_file.with_suffix(".json")
    if not weights_file.exists() or not info_file.exists():
        raise PredictionUnavailable(f"{weights_file} and {info_file.name} beside it are both needed.")
    payload = weights_file.read_bytes()
    info = json.loads(info_file.read_text(encoding="utf-8"))
    if info.get("sha256") and hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise PredictionUnavailable(f"{weights_file} does not match its recorded SHA-256; take part in a new round.")
    weights = weights_from_bytes(payload)
    adapter = get_adapter(modality, backend)
    if adapter.signature(weights) != adapter.signature():
        raise PredictionUnavailable(
            f"The saved {modality} model ({info.get('architecture', 'unknown architecture')}) does not match this "
            "agent's model. Update the agent, or take part in a new round."
        )
    return LocalModel(modality, weights, info, adapter)


class LocalPredictor:
    """Predicts with a saved global model, filling gaps and explaining from this hospital's own records."""

    def __init__(self, model: LocalModel, data_dir: Optional[Path] = None, seed: int = 0):
        self.model = model
        self.schema = tabular_schema(model.modality)
        self.bounds = BOUNDS[model.modality]
        self.seed = seed
        self.imputer: Optional[EHRImputer] = None
        self.background: Optional[np.ndarray] = None
        self.background_source = None
        if data_dir is not None:
            self._fit_local(Path(data_dir))

    # ------------------------------------------------------------ setup
    def _fit_local(self, data_dir: Path) -> None:
        try:
            train = load_local_dataset(data_dir, self.model.modality, "train", allow_missing_features=True)
        except (DatasetUnavailableError, ValueError):
            return
        X = np.asarray(train.X, dtype=np.float64)
        if len(X) == 0:
            return
        self.imputer = EHRImputer(scale=self.schema.reference_scale).fit(X)
        filled = self.imputer.transform(X)
        rng = np.random.default_rng(self.seed)
        rows = rng.choice(len(filled), size=min(BACKGROUND_ROWS, len(filled)), replace=False)
        self.background = filled[rows]
        self.background_source = f"{len(rows)} of this hospital's training records"

    @property
    def can_fill_gaps(self) -> bool:
        return self.imputer is not None

    # ------------------------------------------------------------ input
    def _row(self, values: Dict[str, Optional[float]]) -> np.ndarray:
        row = np.full(self.schema.num_features, np.nan, dtype=np.float64)
        unknown = sorted(set(values) - set(self.schema.feature_names))
        if unknown:
            raise ValueError(f"Unknown input(s): {', '.join(unknown)}")
        for i, name in enumerate(self.schema.feature_names):
            value = values.get(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'{self._label(name)}' must be a number, got {value!r}") from exc
            if math.isnan(number):
                continue
            low, high = self.bounds[name]
            if not low <= number <= high:
                unit = self._feature(name).unit
                raise ValueError(f"'{self._label(name)}' = {number:g} is outside {low}-{high} {unit}; check the units")
            row[i] = number
        return row

    def _feature(self, name: str):
        return self.schema.features[self.schema.feature_names.index(name)]

    def _label(self, name: str) -> str:
        return self._feature(name).label

    def _fill(self, rows: np.ndarray) -> tuple:
        gaps = np.isnan(rows)
        if not gaps.any():
            return rows.astype(np.float32), gaps
        if self.imputer is None:
            names = [self._label(n) for n, gap in zip(self.schema.feature_names, gaps.any(axis=0)) if gap]
            raise ValueError(
                f"Missing value(s): {', '.join(names)}. Fill them in, or choose the data folder so the agent can "
                "estimate them from your own training records."
            )
        return self.imputer.transform(rows), gaps

    # ------------------------------------------------------------ predict
    def predict_one(self, values: Dict[str, Optional[float]], explain: bool = True) -> dict:
        row = self._row(values)
        filled, gaps = self._fill(row.reshape(1, -1))
        return self._result(filled[0], gaps[0], explain)

    def _result(self, x: np.ndarray, gaps: np.ndarray, explain: bool) -> dict:
        adapter = self.model.adapter
        class_names = list(adapter.task.class_names)
        probability = float(adapter.predict_proba(self.model.weights, x.reshape(1, -1))[0, 1])
        positive = probability >= THRESHOLD
        result = {
            "modality": self.model.modality,
            "probability": probability,
            "threshold": THRESHOLD,
            "prediction": class_names[1] if positive else class_names[0],
            "positive": bool(positive),
            "inputs": [
                {
                    "feature": name,
                    "label": self._label(name),
                    "unit": self._feature(name).unit,
                    "value": float(value),
                    "estimated": bool(gap),
                }
                for name, value, gap in zip(self.schema.feature_names, x, gaps)
            ],
            "model": {
                "architecture": self.model.info.get("architecture"),
                "version": self.model.info.get("model_version"),
                "round_id": self.model.info.get("round_id"),
                "sha256": self.model.info.get("sha256"),
                "saved_at": self.model.info.get("downloaded_at"),
                "local_f1": self.model.info.get("local_f1"),
                "trained_on_simulated_data": self.model.info.get("local_data_simulated"),
            },
            "factors": [],
            "explanation_method": None,
            "disclaimer": DISCLAIMER,
        }
        estimated = {name for name, gap in zip(self.schema.feature_names, gaps) if gap}
        if explain and self.background is not None:
            explanation = explain_ehr_record(
                adapter,
                self.model.weights,
                x,
                self.background,
                self.schema.feature_metadata(),
                n_permutations=N_PERMUTATIONS,
                seed=self.seed,
            )
            result["factors"] = [
                {
                    "feature": c["feature"],
                    "label": c["label"],
                    "value": c["value"],
                    "unit": c["unit"],
                    "effect": c["shap_value"],
                    "direction": c["direction"],
                    "estimated": c["feature"] in estimated,
                }
                for c in explanation["contributions"]
            ]
            result["explanation_method"] = f"{explanation['method']}; background: {self.background_source}"
        return result

    def predict_frame(self, frame: pd.DataFrame, top_factors: int = 3) -> pd.DataFrame:
        """Score every row of a table with the schema's columns. Returns the table with result columns added."""
        missing = [c for c in self.schema.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"The file is missing column(s): {', '.join(missing)}")
        out = frame.copy()
        probabilities: List[float] = []
        predictions: List[str] = []
        estimated: List[str] = []
        reasons: List[str] = []
        problems: List[str] = []
        for index, (_, record) in enumerate(frame[self.schema.feature_names].iterrows()):
            try:
                row = self._row({k: (None if pd.isna(v) else v) for k, v in record.items()})
                filled, gaps = self._fill(row.reshape(1, -1))
                result = self._result(filled[0], gaps[0], explain=index < MAX_EXPLAINED_ROWS)
            except ValueError as exc:
                probabilities.append(float("nan"))
                predictions.append("")
                estimated.append("")
                reasons.append("")
                problems.append(str(exc))
                continue
            probabilities.append(round(result["probability"], 4))
            predictions.append(result["prediction"])
            estimated.append(", ".join(i["label"] for i in result["inputs"] if i["estimated"]))
            reasons.append(describe_factors(result["factors"][:top_factors]))
            problems.append("")
        out["probability"] = probabilities
        out["prediction"] = predictions
        out["estimated_inputs"] = estimated
        out["main_reasons"] = reasons
        out["problem"] = problems
        return out


#: Units not worth printing after a value.
_PLAIN_UNITS = {"count", "score", "0/1"}


def format_value(value: float, unit: str) -> str:
    text = f"{value:.4g}"
    return text if unit in _PLAIN_UNITS else f"{text} {unit}"


def describe_factors(factors: Sequence[dict]) -> str:
    """'Glucose (2-h OGTT) 168 mg/dL raises risk; Body mass index 38.2 kg/m² raises risk' for the given factors."""
    parts = []
    for factor in factors:
        if factor["direction"] == "neutral":
            continue
        estimated = " (estimated)" if factor.get("estimated") else ""
        parts.append(f"{factor['label']} {format_value(factor['value'], factor['unit'])}{estimated} {factor['direction']} risk")
    return "; ".join(parts)
