"""Simulated diabetes screening records (the eight PIMA-style measurements).

SIMULATED DATA. Generated from a hand-written model loosely shaped like the PIMA Indians Diabetes Database:
a latent metabolic risk raises glucose, insulin, BMI and skin fold, and the outcome depends mostly on glucose,
BMI, age and family history. It contains no real patient information and must not be used to draw clinical
conclusions. For real records, prepare the PIMA file with ``scripts/prepare_real_data.py diabetes --pima``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.diabetes.features import DIABETES_FEATURE_NAMES


@dataclass(frozen=True)
class DiabetesProfile:
    """How one hospital's patients differ from the others (non-IID)."""

    age_mean: float = 34.0
    bmi_mean: float = 32.0
    risk_shift: float = 0.0  # mean latent metabolic risk; a diabetes clinic sees higher-risk patients
    #: Share of insulin and skin-fold values left unmeasured (empty cells), as is common in screening data.
    unmeasured: float = 0.3


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def simulate_diabetes(n: int, rng: np.random.Generator, profile: DiabetesProfile = DiabetesProfile()):
    age = np.clip(rng.gamma(2.2, (profile.age_mean - 21) / 2.2, n) + 21, 21, 85)
    pregnancies = np.clip(np.round(rng.poisson(np.clip((age - 18) / 6.0, 0.2, 8.0))), 0, 17)
    risk = rng.normal(profile.risk_shift, 1.0, n)
    pedigree = np.clip(rng.lognormal(np.log(0.38) + 0.15 * risk, 0.55, n), 0.08, 2.4)

    bmi = np.clip(profile.bmi_mean + 2.2 * risk + rng.normal(0, 5.5, n), 18, 67)
    glucose = np.clip(117 + 14 * risk + 0.25 * (age - 33) + rng.normal(0, 22, n), 56, 199)
    diastolic_bp = np.clip(70 + 0.25 * (age - 33) + 0.35 * (bmi - 32) + rng.normal(0, 11, n), 30, 122)
    skin = np.clip(29 + 0.9 * (bmi - 32) + rng.normal(0, 7, n), 7, 99)
    insulin = np.clip(np.exp(4.85 + 0.25 * risk + 0.006 * (glucose - 120) + rng.normal(0, 0.55, n)), 14, 846)

    logit = (
        -1.0
        + 0.048 * (glucose - 121)
        + 0.10 * (bmi - 32)
        + 0.035 * (age - 33)
        + 1.2 * (pedigree - 0.47)
        + 0.07 * (pregnancies - 3.8)
    )
    y = rng.binomial(1, _sigmoid(logit)).astype(np.int64)

    columns = {
        "pregnancies": pregnancies,
        "glucose": np.round(glucose),
        "diastolic_bp": np.round(diastolic_bp),
        "skin_thickness": np.round(skin),
        "insulin": np.round(insulin),
        "bmi": np.round(bmi, 1),
        "pedigree": np.round(pedigree, 3),
        "age": np.round(age),
    }
    X = np.column_stack([columns[name] for name in DIABETES_FEATURE_NAMES]).astype(np.float32)
    # Screening data often lacks the two costlier measurements; leave some empty for the node to fill.
    for name in ("insulin", "skin_thickness"):
        column = DIABETES_FEATURE_NAMES.index(name)
        X[rng.random(n) < profile.unmeasured, column] = np.nan
    return X, y
