"""ECG beat preprocessing shared by every backend."""

from __future__ import annotations

import numpy as np

#: Samples per beat window. Matches the widely used MIT-BIH heartbeat export
#: (187 samples at 125 Hz), so real data can be dropped in without resampling.
ECG_BEAT_LENGTH = 187
ECG_SAMPLE_RATE_HZ = 125


def prepare_ecg(X: np.ndarray) -> np.ndarray:
    """Return float32 beats of shape [n, ECG_BEAT_LENGTH], clipped to a sane range."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    if X.ndim != 2 or X.shape[1] != ECG_BEAT_LENGTH:
        raise ValueError(f"ECG input must have shape [n, {ECG_BEAT_LENGTH}], got {X.shape}")
    return np.clip(X, -5.0, 5.0)


#: Per-beat rhythm inputs after the 187 samples: pre-RR, post-RR and local average RR (10 intervals around the
#: beat), each divided by the record's average RR, as in de Chazal et al. (2004) and experiment 1f.
RR_FEATURES = ("pre_rr", "post_rr", "local_rr")
RR_INPUTS = len(RR_FEATURES)
#: Model input width: beat samples followed by the RR inputs.
ECG_INPUT_LENGTH = ECG_BEAT_LENGTH + RR_INPUTS
#: Used when a source has no RR intervals (for example the Kaggle heartbeat export): a perfectly regular rhythm.
RR_NEUTRAL = 1.0
RR_RANGE = (0.2, 4.0)


def ecg_input(signals: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
    """Model rows ``[n, 190]``: the beats followed by their RR inputs (neutral 1.0 when ``rr`` is None)."""
    beats = prepare_ecg(signals)
    if rr is None:
        rr = np.full((len(beats), RR_INPUTS), RR_NEUTRAL, dtype=np.float32)
    rr = np.asarray(rr, dtype=np.float32).reshape(-1, RR_INPUTS)
    if len(rr) != len(beats):
        raise ValueError(f"{len(beats)} beats but {len(rr)} rows of RR intervals")
    if not np.all(np.isfinite(rr)):
        raise ValueError("RR intervals contain NaN or infinite values")
    return np.concatenate([beats, np.clip(rr, *RR_RANGE)], axis=1).astype(np.float32)


def prepare_ecg_input(X: np.ndarray) -> np.ndarray:
    """Model rows from either beats alone ``[n, 187]`` (rhythm assumed regular) or beats + RR ``[n, 190]``."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    if X.ndim == 2 and X.shape[1] == ECG_INPUT_LENGTH:
        return ecg_input(X[:, :ECG_BEAT_LENGTH], X[:, ECG_BEAT_LENGTH:])
    if X.ndim == 2 and X.shape[1] == ECG_BEAT_LENGTH:
        return ecg_input(X)
    raise ValueError(f"ECG input must have shape [n, {ECG_BEAT_LENGTH}] or [n, {ECG_INPUT_LENGTH}], got {X.shape}")
