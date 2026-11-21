"""Pairwise-masked secure aggregation: masks cancel in the sum, a single upload reveals nothing useful."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.privacy.secure_aggregation import (
    KeyPair,
    SecureAggregationError,
    check_masked_update,
    check_public_key,
    mask_update,
    pair_secret,
    plausible_average,
    unmask_average,
)


def _weights(rng, offset=0.0):
    return {
        "conv.weight": rng.normal(offset, 0.3, (4, 3, 5)).astype(np.float32),
        "head.bias": rng.normal(offset, 0.1, (4,)).astype(np.float32),
        "bn.num_batches_tracked": np.array(12, dtype=np.int64),
    }


def _round(hospitals, rng):
    keys = {h: KeyPair.generate() for h in hospitals}
    public = {h: k.public_b64 for h, k in keys.items()}
    updates = {h: (_weights(rng, offset=i * 0.05), 50 + 40 * i) for i, h in enumerate(hospitals)}
    masked = {h: mask_update(w, n, "ecg-r0001", h, keys[h], public) for h, (w, n) in updates.items()}
    return keys, public, updates, masked


def test_both_sides_derive_the_same_pair_secret():
    a, b, c = KeyPair.generate(), KeyPair.generate(), KeyPair.generate()
    assert pair_secret(a, b.public_b64, "r1", "hospital_a", "hospital_b") == pair_secret(
        b, a.public_b64, "r1", "hospital_b", "hospital_a"
    )
    assert pair_secret(a, b.public_b64, "r1", "hospital_a", "hospital_b") != pair_secret(
        a, c.public_b64, "r1", "hospital_a", "hospital_c"
    )
    assert pair_secret(a, b.public_b64, "r1", "hospital_a", "hospital_b") != pair_secret(
        a, b.public_b64, "r2", "hospital_a", "hospital_b"
    )  # fresh masks every round


def test_masks_cancel_and_the_sum_is_the_fedavg_model():
    rng = np.random.default_rng(0)
    hospitals = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
    _, _, updates, masked = _round(hospitals, rng)
    total = sum(n for _, n in updates.values())
    expected = {
        k: sum(n * np.asarray(w[k], dtype=np.float64) for w, n in updates.values()) / total
        for k in updates["hospital_a"][0]
    }
    template = updates["hospital_a"][0]
    for upload in masked.values():
        check_masked_update(upload, template)
    averaged = unmask_average(masked.values(), total, template)
    for k in expected:
        if k == "bn.num_batches_tracked":
            assert averaged[k] == np.rint(expected[k]) and averaged[k].dtype == np.int64
        else:
            assert averaged[k].dtype == np.float32
            assert np.allclose(averaged[k], expected[k], atol=1e-6)
    assert plausible_average(averaged, template) == []


def test_a_single_upload_looks_like_noise():
    rng = np.random.default_rng(1)
    _, _, updates, masked = _round(["hospital_a", "hospital_b", "hospital_c"], rng)
    weights, n = updates["hospital_a"]
    upload = masked["hospital_a"]["conv.weight"].view(np.int64).astype(np.float64) / 2**24 / n
    # Unmasked, the values are around 0 with SD 0.3; masked, they span the whole 64-bit range.
    assert np.abs(upload).mean() > 1e6
    assert abs(np.corrcoef(upload.ravel(), weights["conv.weight"].ravel())[0, 1]) < 0.5


def test_a_missing_hospital_leaves_garbage_that_is_detected():
    rng = np.random.default_rng(2)
    _, _, updates, masked = _round(["hospital_a", "hospital_b", "hospital_c"], rng)
    total = sum(n for h, (_, n) in updates.items() if h != "hospital_c")
    template = updates["hospital_a"][0]
    partial = unmask_average([masked["hospital_a"], masked["hospital_b"]], total, template)
    assert plausible_average(partial, template)  # problems reported


def test_rejects_bad_inputs():
    rng = np.random.default_rng(3)
    _, public, updates, masked = _round(["hospital_a", "hospital_b"], rng)
    template = updates["hospital_a"][0]
    with pytest.raises(SecureAggregationError):
        check_masked_update({**masked["hospital_a"], "head.bias": np.zeros(4, np.float32)}, template)
    with pytest.raises(SecureAggregationError):
        check_masked_update({k: v for k, v in masked["hospital_a"].items() if k != "head.bias"}, template)
    with pytest.raises(SecureAggregationError):
        check_public_key("not-a-key")
    keys = KeyPair.generate()
    with pytest.raises(SecureAggregationError, match="at least two"):
        mask_update(template, 10, "r", "hospital_a", keys, {"hospital_a": keys.public_b64})
    bad = {**template, "head.bias": np.array([np.nan, 0, 0, 0], np.float32)}
    with pytest.raises(SecureAggregationError, match="NaN"):
        mask_update(bad, 10, "r", "hospital_a", keys, public)
