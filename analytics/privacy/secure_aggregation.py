"""Secure aggregation with pairwise masks (after Bonawitz et al., CCS 2017, without the dropout recovery).

Goal: the coordinator learns only the *sum* of the hospitals' sample-weighted updates, never one hospital's update.

How it works, for hospitals 1..K in one round:

1. When it joins, each hospital creates a fresh X25519 key pair for this round and sends the public key.
2. Once every selected hospital has joined, each hospital fetches the others' public keys. Hospitals i and j derive
   the same 256-bit secret from Diffie-Hellman; the coordinator, which sees only public keys, cannot.
3. Each hospital encodes ``num_samples * weights`` as fixed-point integers in the ring of 2^64 and adds, for every
   other hospital j, a pseudorandom mask from ChaCha20 keyed by their shared secret: ``+mask`` if its id sorts
   before j, ``-mask`` otherwise. On its own the upload looks like uniform noise.
4. The coordinator adds all uploads (mod 2^64). Every mask appears once with + and once with -, so they cancel
   exactly and what is left is ``sum_k n_k * W_k``. Dividing by ``N = sum_k n_k`` gives the FedAvg model.

Limits, stated plainly:

* **Every selected hospital must submit.** If one drops out after the keys are exchanged, its masks do not cancel
  and the round cannot be aggregated. (Bonawitz et al. recover with secret-shared seeds; this prototype fails the
  round instead.)
* **Honest-but-curious coordinator.** The coordinator relays the public keys. One that swaps keys (an active
  attacker) could unmask updates; signed keys would prevent that.
* **What the sum still reveals.** With exactly two hospitals, each can subtract its own update from the result
  and recover the other's. Use at least three for the protection to mean anything; combine with differential
  privacy to protect what the aggregate itself reveals.
* **No per-update screening.** The coordinator cannot inspect a single update, so update-norm checks and the
  Isolation Forest screen cannot run. Only the shapes and the final average are checked.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

Weights = Dict[str, np.ndarray]

#: Fixed-point scale: values are stored as round(x * 2^24). Resolution 6e-8 per unit before dividing by N.
FRACTION_BITS = 24
SCALE = float(2**FRACTION_BITS)
#: |n_k * w| must stay below this so the sum cannot wrap around the signed 64-bit range.
MAX_ABS_VALUE = 2.0**62 / SCALE / 64
MASKED_DTYPE = np.uint64
#: With two hospitals each could subtract its own update from the sum; require at least three.
MIN_HOSPITALS = 3
METHOD = "Pairwise-masked secure aggregation (X25519 + HKDF-SHA256 + ChaCha20, after Bonawitz et al. 2017)"


class SecureAggregationError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- keys
@dataclass
class KeyPair:
    private: X25519PrivateKey

    @classmethod
    def generate(cls) -> "KeyPair":
        return cls(X25519PrivateKey.generate())

    @property
    def public_b64(self) -> str:
        raw = self.private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode()


def check_public_key(value: str) -> str:
    """A base64 X25519 public key (32 bytes), or SecureAggregationError."""
    try:
        raw = base64.b64decode(value, validate=True)
        X25519PublicKey.from_public_bytes(raw)
    except Exception as exc:
        raise SecureAggregationError("The secure-aggregation public key must be 32 bytes of base64 (X25519)") from exc
    return value


def pair_secret(keys: KeyPair, peer_public_b64: str, round_id: str, a: str, b: str) -> bytes:
    """The 32-byte secret hospitals ``a`` and ``b`` share for this round (same on both sides)."""
    peer = X25519PublicKey.from_public_bytes(base64.b64decode(check_public_key(peer_public_b64)))
    shared = keys.private.exchange(peer)
    low, high = sorted([a, b])
    info = f"medifed-secagg|{round_id}|{low}|{high}".encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)


def _mask_stream(secret: bytes, key: str, count: int) -> np.ndarray:
    """``count`` pseudorandom uint64 values for one tensor, from ChaCha20 keyed by the pair secret."""
    nonce = hashlib.sha256(key.encode()).digest()[:16]
    encryptor = Cipher(algorithms.ChaCha20(secret, nonce), mode=None).encryptor()
    stream = encryptor.update(b"\0" * (8 * count))
    return np.frombuffer(stream, dtype="<u8").astype(MASKED_DTYPE, copy=False)


# --------------------------------------------------------------------------------------------- encode
def _encode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise SecureAggregationError("The update contains NaN or infinite values")
    if values.size and float(np.max(np.abs(values))) > MAX_ABS_VALUE:
        raise SecureAggregationError("The update has values too large for fixed-point secure aggregation")
    return np.rint(values * SCALE).astype(np.int64).view(MASKED_DTYPE)


def mask_update(
    weights: Mapping[str, np.ndarray],
    num_samples: int,
    round_id: str,
    own_id: str,
    keys: KeyPair,
    peer_public_keys: Mapping[str, str],
) -> Weights:
    """The masked upload: fixed-point ``num_samples * weights`` plus the pairwise masks with every other hospital."""
    peers = {h: k for h, k in peer_public_keys.items() if h != own_id}
    if not peers:
        raise SecureAggregationError("Secure aggregation needs at least two hospitals")
    secrets = {h: pair_secret(keys, k, round_id, own_id, h) for h, k in peers.items()}
    masked: Weights = {}
    for name in sorted(weights):
        tensor = np.asarray(weights[name])
        encoded = _encode(tensor.astype(np.float64) * float(num_samples)).reshape(-1)
        for peer_id, secret in secrets.items():
            mask = _mask_stream(secret, name, encoded.size)
            # uint64 arithmetic wraps modulo 2^64, which is exactly the ring we need.
            encoded = encoded + mask if own_id < peer_id else encoded - mask
        masked[name] = encoded.reshape(tensor.shape)
    return masked


# --------------------------------------------------------------------------------------------- server side
def check_masked_update(masked: Mapping[str, np.ndarray], template: Mapping[str, np.ndarray]) -> None:
    """What the coordinator can check without seeing the update: same tensors, same shapes, the masked type."""
    if set(masked) != set(template):
        missing, extra = sorted(set(template) - set(masked)), sorted(set(masked) - set(template))
        raise SecureAggregationError(f"Masked update tensors do not match the model (missing {missing}, extra {extra})")
    for name, reference in template.items():
        array = np.asarray(masked[name])
        if array.dtype != MASKED_DTYPE:
            raise SecureAggregationError(f"Masked tensor '{name}' must be uint64, got {array.dtype}")
        if array.shape != np.asarray(reference).shape:
            raise SecureAggregationError(f"Masked tensor '{name}' has shape {array.shape}, expected {np.shape(reference)}")


def unmask_average(masked_updates: Iterable[Mapping[str, np.ndarray]], total_samples: int, template: Weights) -> Weights:
    """Sum the masked uploads (the masks cancel) and divide by the total sample count: the FedAvg model."""
    masked_updates = list(masked_updates)
    if len(masked_updates) < 2:
        raise SecureAggregationError("Secure aggregation needs at least two updates")
    if total_samples <= 0:
        raise SecureAggregationError("The total number of samples must be positive")
    averaged: Weights = {}
    for name, reference in template.items():
        total = np.zeros(np.asarray(reference).shape, dtype=MASKED_DTYPE)
        for update in masked_updates:
            total = total + np.asarray(update[name], dtype=MASKED_DTYPE)
        values = total.view(np.int64).astype(np.float64) / SCALE / float(total_samples)
        reference = np.asarray(reference)
        if np.issubdtype(reference.dtype, np.floating):
            averaged[name] = values.astype(reference.dtype)
        else:
            averaged[name] = np.rint(values).astype(reference.dtype)
    return averaged


def plausible_average(averaged: Weights, base: Weights, max_relative_change: float = 100.0) -> List[str]:
    """Problems with an unmasked average (a missing hospital leaves random-looking values). Empty when it is fine."""
    problems = []
    for name, value in averaged.items():
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            problems.append(f"'{name}' has non-finite values")
            continue
        reference = np.asarray(base[name], dtype=np.float64)
        scale = max(float(np.linalg.norm(reference)), 1.0)
        if float(np.linalg.norm(value - reference)) > max_relative_change * scale:
            problems.append(f"'{name}' moved {np.linalg.norm(value - reference):.3g} from the base model")
    return problems
