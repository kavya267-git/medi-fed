# Security and privacy model

This page lists what leaves a hospital, how each piece is protected, and the limits that remain. MediFed is a
research prototype; none of this makes it a certified medical or security product.

## What leaves a hospital

| Sent to the coordinator | Never sent |
| --- | --- |
| Model weights after local training (masked under secure aggregation) | Patient records: rows, ECG beats, X-ray images |
| Number of training records | Values typed into **Predict**, and its results |
| Training loss and local validation scores | The hospital's data files |
| Confusion-matrix counts on held-out data | |
| Average feature importance (EHR and diabetes only; no individual values) | |
| Data summary: record counts and which columns exist | |
| Online status (heartbeat) | |

Prediction with the global model runs entirely on the hospital's computer (`ml/prediction/`).

## Protections

| Risk | Protection | Where |
| --- | --- | --- |
| Token or weights read on the network | The agent and nodes require `https://` (certificate verified) unless the server is on this computer or the local network; `MEDIFED_ALLOW_INSECURE_HTTP=1` is for development only | `ml/training/http_transport.py` (`check_server_url`) |
| Coordinator inspects one hospital's update | **Secure aggregation** (optional per round): each hospital adds pairwise masks that cancel only in the sum, so the coordinator sees only the combined update | `analytics/privacy/secure_aggregation.py` |
| What the combined model reveals about a patient | **Differential privacy** (optional per round): updates are clipped and Gaussian noise is added; the cost is tracked with zCDP | `analytics/privacy/dp.py` |
| Stolen node token | Tokens are stored hashed; an admin can revoke a hospital at once and the hospital's operator can issue a new token, which replaces the old one | `backend/app/services/hospital_service.py` |
| A faulty or tampered update | Shape and architecture checks on every update; Isolation Forest screening flags unusual updates (not available under secure aggregation) | `backend/app/federated/validation.py`, `screening.py` |
| Mixing incompatible models | Architecture signatures: updates from another architecture are rejected | `ml/common/weights.py` |
| Records changed after the fact | Hash-chained audit log: editing an old entry breaks the chain | `backend/app/services/audit_service.py` |
| Re-identification from small counts | Weekly surveillance counts of 1–4 are suppressed | `analytics/privacy/suppression.py` |
| Seeing more than a role needs | Seven roles with server-side permission checks | `docs/permissions.md` |

## Secure aggregation in detail

1. When a hospital joins a secure round, it creates a fresh X25519 key pair and sends only the public key.
2. When every selected hospital has sent its key, each one fetches the others' keys. Each pair derives a shared
   secret (Diffie–Hellman + HKDF-SHA256) that the coordinator cannot compute.
3. Each hospital encodes `samples × weights` as fixed-point numbers modulo 2^64 and adds a ChaCha20 mask for every
   other hospital: `+mask` towards hospitals whose id sorts after its own, `−mask` towards the others.
4. The coordinator adds the uploads. Every mask appears once with + and once with −, so they cancel exactly, and
   dividing by the total number of samples gives the same FedAvg model as without masking (tested to 1e-5).

Limits:

- **Every selected hospital must send its update.** If one drops out, its masks cannot be removed and the round
  fails. Bonawitz et al. (2017) recover with secret-shared seeds; this prototype does not.
- **At least three hospitals.** With two, each could subtract its own update from the sum.
- **Honest-but-curious coordinator.** The coordinator relays the public keys; a coordinator that swapped them
  could unmask updates. Signed keys would close this.
- **No per-update screening.** The coordinator cannot see a single update, so update norms and the Isolation
  Forest screen are not available in these rounds. The unmasked average is still checked for finite, plausible
  values.

## Remaining limits

- Differential privacy is off by default, and the tested setting gives a weak guarantee (ε ≈ 31 at σ = 1.0).
- Data files on a hospital's computer are not encrypted by MediFed; that is the hospital's own device security.
- The coordinator learns which hospitals took part, how many records each used and their local scores.
