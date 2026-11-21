"""Rounds with secure aggregation: the coordinator only ever forms the sum of the hospitals' updates."""

import numpy as np
import pytest

from analytics.privacy.secure_aggregation import KeyPair, mask_update
from app.federated.fedavg import ClientUpdate, federated_average
from app.utils.errors import ConflictError
from backend_test_helpers import run_round
from ml.common.weights import weights_from_bytes, weights_to_bytes

HOSPITALS = ("hospital_a", "hospital_b", "hospital_c", "hospital_d")


def _start(ctx, **kwargs):
    return ctx.coordinator.start_round("ehr", ctx.hospitals.list_raw(), "tester", secure_aggregation=True, **kwargs)


def test_masked_uploads_give_exactly_the_fedavg_model(ctx):
    round_id = _start(ctx, auto_aggregate=False)["id"]
    keys = {}
    for h in HOSPITALS:
        info = ctx.coordinator.join(round_id, h)
        assert info["secure_aggregation"]["enabled"] and info["secure_aggregation"]["hospitals"] == list(HOSPITALS)
        keys[h] = KeyPair.generate()
        answer = ctx.coordinator.register_secagg_key(round_id, h, keys[h].public_b64)
    assert answer["ready"] and set(answer["public_keys"]) == set(HOSPITALS)
    public = answer["public_keys"]
    base = weights_from_bytes(ctx.coordinator.global_model_bytes(round_id, "hospital_a", "base"))
    rng = np.random.default_rng(0)
    plain = {}
    for i, h in enumerate(HOSPITALS):
        w = {k: (v + rng.normal(0, 0.05, v.shape)).astype(v.dtype) if np.issubdtype(v.dtype, np.floating) else v for k, v in base.items()}
        n = 40 + 30 * i
        plain[h] = (w, n)
        masked = mask_update(w, n, round_id, h, keys[h], public)
        result = ctx.coordinator.receive_update(round_id, h, weights_to_bytes(masked), n)
        assert result["status"] == "VALIDATED" and result["update_norm"] is None
    # What the coordinator stored for one hospital is noise, not that hospital's weights.
    stored = weights_from_bytes(ctx.coordinator._update_file(round_id, "hospital_a").read_bytes())
    some_key = next(k for k in base if np.issubdtype(base[k].dtype, np.floating) and base[k].size > 4)
    assert stored[some_key].dtype == np.uint64
    ctx.coordinator.aggregate(round_id, "tester")
    averaged = weights_from_bytes(ctx.coordinator._aggregated_file(round_id).read_bytes())
    expected = federated_average([ClientUpdate(h, w, n) for h, (w, n) in plain.items()])
    for k in expected:
        assert np.allclose(averaged[k], expected[k], atol=1e-5), k
    round_row = ctx.coordinator.get_round(round_id)
    assert round_row["config"]["aggregation"]["algorithm"] == "FedAvg (secure aggregation)"
    listing = ctx.audit.list_entries(round_id=round_id, limit=200)
    events = [e["action"] for e in listing.get("entries", listing.get("items", []))]
    assert "UPDATE_SCREENING_SKIPPED" in events and "SECAGG_KEY_RECEIVED" in events


def test_full_round_with_in_process_hospitals_completes(ctx):
    result = run_round(ctx, "diabetes", secure_aggregation=True)
    assert result["status"] == "COMPLETED", result.get("error")
    updates = ctx.coordinator.updates(result["id"])
    assert len(updates) == 4 and all(u["update_norm"] is None for u in updates)
    assert result["global_metrics"]


def test_a_missing_hospital_fails_the_round(ctx):
    round_id = _start(ctx, auto_aggregate=False)["id"]
    for h in HOSPITALS:
        ctx.coordinator.join(round_id, h)
        ctx.coordinator.register_secagg_key(round_id, h, KeyPair.generate().public_b64)
    ctx.coordinator.report_failure(round_id, "hospital_d", "training", "laptop switched off")
    row = ctx.coordinator.get_round(round_id)
    assert row["status"] == "FAILED" and "every selected hospital" in row["error"]


def test_keys_are_released_only_when_every_hospital_has_sent_one(ctx):
    round_id = _start(ctx)["id"]
    ctx.coordinator.join(round_id, "hospital_a")
    answer = ctx.coordinator.register_secagg_key(round_id, "hospital_a", KeyPair.generate().public_b64)
    assert not answer["ready"] and answer["waiting_for"] == 3 and answer["public_keys"] == {}
    with pytest.raises(ConflictError, match="already sent"):
        ctx.coordinator.register_secagg_key(round_id, "hospital_a", KeyPair.generate().public_b64)
    with pytest.raises(ConflictError, match="must join"):
        ctx.coordinator.register_secagg_key(round_id, "hospital_b", KeyPair.generate().public_b64)


def test_needs_three_hospitals_and_rejects_unmasked_uploads(ctx):
    two = [h for h in ctx.hospitals.list_raw() if h["id"] in ("hospital_a", "hospital_b")]
    with pytest.raises(ConflictError, match="at least 3"):
        ctx.coordinator.start_round("ehr", two, "tester", secure_aggregation=True)
    round_id = _start(ctx, auto_aggregate=False)["id"]
    for h in HOSPITALS:
        ctx.coordinator.join(round_id, h)
    base = weights_from_bytes(ctx.coordinator.global_model_bytes(round_id, "hospital_a", "base"))
    from app.utils.errors import ValidationFailed

    with pytest.raises(ValidationFailed, match="uint64"):
        ctx.coordinator.receive_update(round_id, "hospital_a", weights_to_bytes(base), 50)  # plain weights refused


def test_api_round_with_secure_aggregation(client, coordinator):
    from backend_test_helpers import wait_for_round

    response = client.post(
        "/api/federated/rounds/start", json={"modality": "diabetes", "secure_aggregation": True}, headers=coordinator
    )
    assert response.status_code == 201, response.text
    body = wait_for_round(client, coordinator, response.json()["id"], timeout=180)
    assert body["status"] == "COMPLETED", body.get("error")
    # The key endpoints are for hospital nodes only.
    assert client.get(f"/api/federated/rounds/{body['id']}/secagg-keys", params={"hospital_id": "hospital_a"}).status_code in (401, 403)
