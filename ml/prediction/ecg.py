"""Using the global ECG model on new heartbeats, inside the hospital.

A hospital loads a file of single-beat ECG windows (187 samples at 125 Hz, the MIT-BIH heartbeat format) and the
global CNN-BiLSTM saved after the last round labels every beat as N, S, V or F. Each beat also gets a simple,
model-agnostic explanation: the part of the beat that, when flattened, lowers the predicted class's probability
the most (occlusion sensitivity). Everything runs on the hospital's own computer; the beats and the results are
never sent anywhere.

This is a research prototype, not a medical device; every result carries ``DISCLAIMER``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ml.common.weights import weights_from_bytes
from ml.ecg.preprocess import ECG_BEAT_LENGTH, ECG_SAMPLE_RATE_HZ, prepare_ecg
from ml.prediction.local import DISCLAIMER, PredictionUnavailable, model_files
from ml.registry import get_adapter

MODALITY = "ecg"
#: Short codes and plain names for the four AAMI beat classes, in the model's output order.
CODES = ("N", "S", "V", "F")
NAMES = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
}
#: Occlusion explanation: window width (samples) and step.
OCCLUDE_WIDTH = 20
OCCLUDE_STEP = 10
#: Explanations are computed for at most this many beats of a file (each costs ~18 extra model passes).
MAX_EXPLAINED_BEATS = 300


@dataclass
class ECGModel:
    weights: dict
    info: dict
    adapter: object = field(repr=False)


def ecg_model_available(models_dir: Optional[Path]) -> Optional[dict]:
    """The saved global ECG model's info, or ``None`` when this computer has none yet."""
    if not models_dir:
        return None
    weights_file, info_file = model_files(Path(models_dir), MODALITY)
    if weights_file.exists() and info_file.exists():
        try:
            return json.loads(info_file.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


def load_ecg_model(models_dir: Path, backend: Optional[str] = None) -> ECGModel:
    """The latest global ECG model saved on this computer, checked against its SHA-256 and architecture."""
    weights_file, info_file = model_files(Path(models_dir), MODALITY)
    if not weights_file.exists() or not info_file.exists():
        raise PredictionUnavailable(
            "No global ECG model on this computer yet. Connect and take part in an ECG round first; the agent keeps "
            "the model each round produces."
        )
    payload = weights_file.read_bytes()
    info = json.loads(info_file.read_text(encoding="utf-8"))
    if info.get("sha256") and hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise PredictionUnavailable(f"{weights_file} does not match its recorded SHA-256; take part in a new round.")
    weights = weights_from_bytes(payload)
    adapter = get_adapter(MODALITY, backend)
    if adapter.signature(weights) != adapter.signature():
        raise PredictionUnavailable(
            f"The saved ECG model ({info.get('architecture', 'unknown architecture')}) does not match this agent's "
            "model. Update the agent, or take part in a new round."
        )
    return ECGModel(weights, info, adapter)


# ---------------------------------------------------------------------------------------------------- input
def read_beats(path: Path) -> tuple:
    """Beats (and any reference labels) from a file.

    Accepted: a CSV or NPY with one beat per row of 187 samples, optionally followed by a 188th column holding a
    reference label (the Kaggle MIT-BIH export: 0=N, 1=S, 2=V, 3=F, 4=Q), or an NPZ with ``signals`` (and
    optionally ``labels``) as the platform writes them. Returns ``(beats [n, 187] float32, labels or None)``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    labels = None
    if suffix == ".npz":
        with np.load(path) as z:
            if "signals" not in z:
                raise ValueError(f"{path.name} has no 'signals' array")
            X = np.asarray(z["signals"], dtype=np.float32)
            labels = np.asarray(z["labels"]).astype(int) if "labels" in z else None
    else:
        if suffix == ".npy":
            X = np.load(path)
        else:
            frame = pd.read_csv(path, header=None)
            if frame.shape[0] and (not _numeric(frame.iloc[0]) or _column_numbers(frame.iloc[0])):
                frame = frame.iloc[1:]  # a header row (names, or the column numbers 0, 1, 2, ...)
            X = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == ECG_BEAT_LENGTH + 1:
            labels = X[:, -1].astype(int)
            X = X[:, :ECG_BEAT_LENGTH]
    if X.ndim != 2 or X.shape[1] != ECG_BEAT_LENGTH:
        raise ValueError(
            f"Each row must be one heartbeat of {ECG_BEAT_LENGTH} samples at {ECG_SAMPLE_RATE_HZ} Hz "
            f"(optionally plus a label column); this file has {X.shape[1] if X.ndim == 2 else '?'} columns."
        )
    if np.isnan(X).any():
        bad = int(np.isnan(X).any(axis=1).sum())
        raise ValueError(f"{bad} beat(s) contain empty or non-numeric samples; fix or remove those rows.")
    return prepare_ecg(X), labels


def _column_numbers(row) -> bool:
    try:
        values = [float(v) for v in row]
    except (TypeError, ValueError):
        return False
    return len(values) > 2 and values[:3] == [0.0, 1.0, 2.0] and all(b - a == 1 for a, b in zip(values, values[1:]))


def _numeric(row) -> bool:
    try:
        [float(v) for v in row]
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------------- predict
class ECGPredictor:
    """Labels heartbeats with the saved global ECG model, on this computer."""

    def __init__(self, model: ECGModel):
        self.model = model

    def probabilities(self, beats: np.ndarray) -> np.ndarray:
        X = prepare_ecg(beats)
        out = []
        for start in range(0, len(X), 512):
            out.append(np.asarray(self.model.adapter.predict_proba(self.model.weights, X[start : start + 512])))
        return np.vstack(out) if out else np.zeros((0, len(CODES)))

    def explain(self, beat: np.ndarray, cls: int) -> dict:
        """Occlusion sensitivity: flatten a sliding window to the beat's median and see how much the predicted
        class's probability drops. Returns the most influential window in samples and milliseconds."""
        beat = prepare_ecg(beat.reshape(1, -1))[0]
        base = float(np.median(beat))
        starts = list(range(0, ECG_BEAT_LENGTH - OCCLUDE_WIDTH + 1, OCCLUDE_STEP))
        variants = np.repeat(beat.reshape(1, -1), len(starts) + 1, axis=0)
        for k, s in enumerate(starts, start=1):
            variants[k, s : s + OCCLUDE_WIDTH] = base
        p = self.probabilities(variants)[:, cls]
        drops = p[0] - p[1:]
        best = int(np.argmax(drops))
        s = starts[best]
        importance = np.zeros(ECG_BEAT_LENGTH)
        for k, st in enumerate(starts):
            importance[st : st + OCCLUDE_WIDTH] = np.maximum(importance[st : st + OCCLUDE_WIDTH], max(0.0, drops[k]))
        ms = 1000.0 / ECG_SAMPLE_RATE_HZ
        return {
            "start_sample": s,
            "end_sample": s + OCCLUDE_WIDTH,
            "start_ms": round(s * ms),
            "end_ms": round((s + OCCLUDE_WIDTH) * ms),
            "probability_drop": round(float(drops[best]), 4),
            "importance": importance.round(4).tolist(),
            "method": f"occlusion sensitivity ({OCCLUDE_WIDTH}-sample windows flattened to the beat's median)",
        }

    def predict_beats(self, beats: np.ndarray, explain: bool = True) -> List[dict]:
        P = self.probabilities(beats)
        results = []
        for i, p in enumerate(P):
            cls = int(np.argmax(p))
            code = CODES[cls]
            item = {
                "beat": i + 1,
                "code": code,
                "prediction": NAMES[code],
                "confidence": float(p[cls]),
                "probabilities": {c: float(v) for c, v in zip(CODES, p)},
                "explanation": self.explain(beats[i], cls) if explain and i < MAX_EXPLAINED_BEATS else None,
            }
            results.append(item)
        return results

    def score(self, beats: np.ndarray, labels: Optional[np.ndarray] = None) -> tuple:
        """Label every beat. Returns ``(table, summary)``; the table has one row per beat, ready to save as CSV."""
        results = self.predict_beats(beats)
        rows = []
        for r in results:
            row = {
                "beat": r["beat"],
                "prediction": r["code"],
                "prediction_name": r["prediction"],
                "confidence": round(r["confidence"], 4),
                **{f"p_{c}": round(v, 4) for c, v in r["probabilities"].items()},
                "key_region_ms": (f"{r['explanation']['start_ms']}-{r['explanation']['end_ms']}" if r["explanation"] else ""),
            }
            rows.append(row)
        table = pd.DataFrame(rows)
        counts = table["prediction"].value_counts().reindex(CODES, fill_value=0).to_dict()
        summary = {
            "beats": len(table),
            "counts": {c: int(n) for c, n in counts.items()},
            "abnormal": int(len(table) - counts["N"]),
            "model": {
                "architecture": self.model.info.get("architecture"),
                "version": self.model.info.get("model_version"),
                "round_id": self.model.info.get("round_id"),
                "trained_on_simulated_data": self.model.info.get("local_data_simulated"),
            },
            "disclaimer": DISCLAIMER,
        }
        if labels is not None:
            known = labels < len(CODES)  # label 4 (Q, unclassifiable) is outside the model's four classes
            table["reference"] = [CODES[int(v)] if v < len(CODES) else "Q" for v in labels]
            table["agrees"] = np.where(known, table["prediction"] == table["reference"], "")
            if known.any():
                agree = float((table.loc[known, "prediction"] == table.loc[known, "reference"]).mean())
                summary["reference_beats"] = int(known.sum())
                summary["agreement"] = round(agree, 4)
        return table, summary


def describe_summary(summary: dict) -> str:
    counts = summary["counts"]
    parts = [f"{counts[c]} {NAMES[c].lower()}{'s' if counts[c] != 1 else ''} ({c})" for c in CODES if counts[c]]
    return f"{summary['beats']} beat(s): " + ", ".join(parts)
