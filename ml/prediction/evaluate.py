"""Testing a saved model on labelled records: how often it agrees with the true answers.

Used by the agent's "Evaluate with labelled file" window (a hospital checks the global model on its own past
patients, on its own computer) and by ``scripts/evaluate_external.py`` (external validation on a cohort that
never took part in training). The file is translated with the same column matching as an import
(``ml.common.column_mapping``), gaps are filled by KNN imputation (``EHRImputer``) and the model standardises
with its fixed reference values, exactly as in training and prediction. A row is called positive when the
predicted probability is at least ``THRESHOLD`` (0.5), as in ``ml.prediction.local``.

Nothing here sends anything anywhere.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ml.common.column_mapping import ColumnMapping, apply_mapping
from ml.common.data_check import BOUNDS
from ml.common.tabular import tabular_schema
from ml.ehr.imputation import EHRImputer
from ml.prediction.local import THRESHOLD

#: Fewer labelled records than this and the agent warns that the numbers can swing a lot from chance alone.
SMALL_SAMPLE = 100


def binary_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float = THRESHOLD) -> dict:
    """Accuracy, macro-F1, positive-class F1, sensitivity, specificity, ROC-AUC and the confusion matrix."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and probability must be 1-D and the same length")
    if len(y) == 0:
        raise ValueError("No labelled records to evaluate")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("The true answers must be 0 / 1")
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def f1(t, f_pos, f_neg):
        return 0.0 if t == 0 else 2 * t / (2 * t + f_pos + f_neg)

    f1_pos, f1_neg = f1(tp, fp, fn), f1(tn, fn, fp)
    positives, negatives = tp + fn, tn + fp
    auc = None
    if positives and negatives:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    return {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "macro_f1": (f1_pos + f1_neg) / 2,
        "f1_positive": f1_pos,
        "sensitivity": tp / positives if positives else None,
        "specificity": tn / negatives if negatives else None,
        "roc_auc": auc,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def bootstrap_ci(y_true, probability, metric: str, n_resamples: int = 1000, seed: int = 0) -> Optional[tuple]:
    """95% percentile bootstrap interval of one metric (resampling records with replacement)."""
    y = np.asarray(y_true)
    p = np.asarray(probability)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        rows = rng.integers(0, len(y), len(y))
        if len(np.unique(y[rows])) < 2:
            continue
        value = binary_metrics(y[rows], p[rows])[metric]
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def labelled_table(frame: pd.DataFrame, mapping: ColumnMapping) -> tuple:
    """The file as the model's raw inputs (gaps as NaN) and the true answers, plus a report of what was done.

    Rows without a true answer are dropped. A value outside the model's plausibility bounds (usually a unit
    mix-up) is treated as not measured and counted in the report."""
    if not mapping.label:
        raise ValueError("Choose the column that holds the true answer (the outcome).")
    schema = tabular_schema(mapping.modality)
    table, report = apply_mapping(frame, mapping, with_label=True, fill_not_recorded=False)
    label = table[schema.label_column]
    keep = label.notna()
    report["dropped_without_label"] = int((~keep).sum())
    table = table[keep]
    y = table[schema.label_column].to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("The true answer must be yes / no. Tick which values of the outcome column mean 'yes'.")
    X = table[schema.feature_names].to_numpy(dtype=np.float64)
    out_of_bounds = {}
    for i, name in enumerate(schema.feature_names):
        low, high = BOUNDS[mapping.modality][name]
        bad = ~np.isnan(X[:, i]) & ((X[:, i] < low) | (X[:, i] > high))
        if bad.any():
            out_of_bounds[name] = int(bad.sum())
            X[bad, i] = np.nan
    report["out_of_bounds_treated_as_missing"] = out_of_bounds
    report["values_missing"] = int(np.isnan(X).sum())
    return X, y.astype(np.int64), report


def _row_keys(X: np.ndarray, decimals: int) -> list:
    rounded = np.round(np.asarray(X, dtype=np.float64), decimals)
    return [tuple(-1e9 if np.isnan(v) else v for v in row) for row in rounded]


def overlap_mask(X: np.ndarray, reference: np.ndarray, decimals: int = 3) -> np.ndarray:
    """True for each row of ``X`` whose values all equal some row of ``reference`` (after rounding).

    Used to drop records of an "external" file that are copies of training records, so that they cannot
    inflate the score. Gaps count as equal only to gaps."""
    known = set(_row_keys(reference, decimals))
    return np.array([key in known for key in _row_keys(X, decimals)], dtype=bool)


def duplicate_mask(X: np.ndarray, y: np.ndarray, decimals: int = 3) -> np.ndarray:
    """True for each row that repeats an earlier row exactly (same values and same true answer)."""
    seen, out = set(), []
    for key, label in zip(_row_keys(X, decimals), np.asarray(y).tolist()):
        out.append((key, label) in seen)
        seen.add((key, label))
    return np.array(out, dtype=bool)


def fill_gaps(X: np.ndarray, modality: str, imputer: Optional[EHRImputer] = None) -> tuple:
    """Gaps filled by KNN imputation: with ``imputer`` (fitted on the hospital's own training records) or,
    without one, an imputer fitted on these records' measurements (the true answers are not used)."""
    if not np.isnan(X).any():
        return X.astype(np.float32), "no gaps to fill"
    if imputer is not None:
        return imputer.transform(X), f"KNN imputation fitted on {imputer.fitted_rows} of your training records"
    schema = tabular_schema(modality)
    fitted = EHRImputer(scale=schema.reference_scale).fit(X)
    return fitted.transform(X), f"KNN imputation fitted on the {len(X)} evaluated records' measurements (not their answers)"


def evaluate_labelled(predictor, frame: pd.DataFrame, mapping: ColumnMapping) -> dict:
    """Score a labelled file with a ``LocalPredictor``'s model and compare with the true answers."""
    X, y, report = labelled_table(frame, mapping)
    if len(y) == 0:
        raise ValueError("No rows with a true answer.")
    filled, how = fill_gaps(X, mapping.modality, predictor.imputer)
    model = predictor.model
    probability = model.adapter.predict_proba(model.weights, filled)[:, 1]
    metrics = binary_metrics(y, probability)
    metrics.update(
        modality=mapping.modality,
        gaps_filled=how,
        rows_with_estimated_values=int(np.isnan(X).any(axis=1).sum()),
        report=report,
        model={
            "architecture": model.info.get("architecture"),
            "version": model.info.get("model_version"),
            "round_id": model.info.get("round_id"),
            "sha256": model.info.get("sha256"),
            "trained_on_simulated_data": model.info.get("local_data_simulated"),
        },
        small_sample=len(y) < SMALL_SAMPLE,
    )
    return metrics
