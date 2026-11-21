"""A hospital's federated-learning client.

One ``HospitalNode`` owns one hospital's data directory. During a round it:

1. joins the round and learns the training configuration;
2. downloads the current global weights;
3. trains locally on its own data;
4. optionally clips and noises the update (differential privacy prototype);
5. sends back *only* model weights, a sample count and summary metrics;
6. later downloads the aggregated model, evaluates it on its local held-out
   split and returns a confusion matrix (plus, for EHR and diabetes, aggregate feature
   importance).

Raw records never leave this class: nothing passed to the transport contains
patient-level data.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from analytics.explainability.service import local_global_importance
from analytics.privacy.dp import DPConfig, privatize_update, update_norm
from ml.common.datasets import local_data_summary
from ml.common.tabular import is_tabular
from ml.common.types import TrainConfig
from ml.common.weights import architecture_signature, weights_from_bytes, weights_to_bytes
from ml.ehr.imputation import MissingnessConfig
from ml.registry import get_adapter
from ml.training.local_trainer import evaluate_on_local_data, load_training_splits, train_on_local_data


class CoordinatorTransport(Protocol):
    """How a node reaches the coordinator (in-process in simulation, HTTP otherwise)."""

    def join_round(self, round_id: str, hospital_id: str, data_summary: dict) -> dict: ...

    def fetch_global_model(self, round_id: str, hospital_id: str, which: str) -> bytes: ...

    def submit_update(
        self,
        round_id: str,
        hospital_id: str,
        payload: bytes,
        num_samples: int,
        metrics: dict,
        dp_report: dict,
        architecture_signature: str,
    ) -> dict: ...

    def submit_evaluation(
        self, round_id: str, hospital_id: str, evaluation: dict, explanation: Optional[dict]
    ) -> dict: ...

    def report_failure(self, round_id: str, hospital_id: str, stage: str, message: str) -> None: ...

    # Secure aggregation only (rounds with ``secure_aggregation.enabled``):
    def register_secagg_key(self, round_id: str, hospital_id: str, public_key: str) -> dict: ...

    def secagg_keys(self, round_id: str, hospital_id: str) -> dict: ...


class HospitalNode:
    def __init__(
        self, hospital_id: str, data_dir: Path, transport: CoordinatorTransport, backend: Optional[str] = None
    ):
        self.hospital_id = hospital_id
        self.data_dir = Path(data_dir)
        self.transport = transport
        self.backend = backend
        #: Size of the last update sent and the last evaluation computed, for the node's console.
        self.last_update_bytes: Optional[int] = None
        #: Norm of the last update before clipping. A local diagnostic (the experiment harness reads it); with DP
        #: enabled it is never sent to the coordinator.
        self.last_update_norm: Optional[float] = None
        self.last_evaluation = None
        #: When set, the aggregated model downloaded for evaluation is kept here (bytes) so the node can save it.
        self.keep_result_model = False
        self.last_result_model: Optional[bytes] = None
        #: This round's secure-aggregation key pairs; the private keys never leave this object.
        self._secagg_keys: dict = {}
        #: How long to wait for every hospital's public key before giving up (seconds).
        self.secagg_wait_seconds = 900.0

    # ------------------------------------------------------------- metadata
    def data_summary(self) -> dict:
        """Counts and provenance only."""
        return local_data_summary(self.data_dir)

    # ---------------------------------------------------------------- round
    def join(self, round_id: str) -> dict:
        info = self.transport.join_round(round_id, self.hospital_id, self.data_summary())
        if (info.get("secure_aggregation") or {}).get("enabled"):
            from analytics.privacy.secure_aggregation import KeyPair

            keys = KeyPair.generate()  # fresh for every round
            self._secagg_keys[round_id] = keys
            self.transport.register_secagg_key(round_id, self.hospital_id, keys.public_b64)
        return info

    def _peer_keys(self, round_id: str) -> dict:
        """Wait until every selected hospital has sent its public key, then return them all."""
        import time

        waited, delay = 0.0, 0.5
        while True:
            answer = self.transport.secagg_keys(round_id, self.hospital_id)
            if answer.get("ready"):
                return answer["public_keys"]
            if answer.get("status") not in (None, "WAITING_FOR_CLIENTS", "TRAINING", "UPDATES_RECEIVING"):
                raise RuntimeError(f"Round {round_id} is {answer.get('status')}; secure aggregation cannot continue")
            if waited >= self.secagg_wait_seconds:
                raise RuntimeError(
                    f"Still waiting for {answer.get('waiting_for')} hospital(s) to send their secure-aggregation key"
                )
            time.sleep(delay)
            waited += delay
            delay = min(delay * 1.5, 5.0)

    def train_and_submit(self, round_id: str, round_info: dict) -> dict:
        modality = round_info["modality"]
        adapter = get_adapter(modality, self.backend)
        global_weights = weights_from_bytes(self.transport.fetch_global_model(round_id, self.hospital_id, "base"))

        training = round_info["training"]
        seed = int(training.get("seed", 0)) + zlib.crc32(f"{self.hospital_id}:{round_id}".encode()) % 100_000
        config = TrainConfig(
            local_epochs=int(training["local_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            seed=seed,
            # Optional; rounds started from the dashboard do not set them.
            class_weighting=str(training.get("class_weighting", "none")),
            proximal_mu=float(training.get("proximal_mu", 0.0)),
        )
        missingness = MissingnessConfig.from_dict(training.get("missingness"))
        output = train_on_local_data(self.data_dir, modality, global_weights, config, self.backend, missingness)

        self.last_update_norm = update_norm(global_weights, output.weights)
        dp_config = DPConfig.from_dict(round_info.get("differential_privacy"))
        private_weights, dp_report = privatize_update(
            global_weights, output.weights, dp_config, np.random.default_rng(seed + 1)
        )
        metrics = {
            "training_history": output.history,
            "local_validation": output.local_validation,
            "duration_seconds": output.duration_seconds,
            "dataset_source": output.dataset_source,
            "backend": adapter.backend,
            "architecture": adapter.architecture,
        }
        if output.missing_values is not None:
            metrics["missing_values"] = output.missing_values
        signature = architecture_signature(private_weights, f"{adapter.backend}:{adapter.architecture}")
        if (round_info.get("secure_aggregation") or {}).get("enabled"):
            from analytics.privacy.secure_aggregation import mask_update

            keys = self._secagg_keys.get(round_id)
            if keys is None:
                raise RuntimeError("This node has no secure-aggregation key for the round; it must join first")
            # Only the masked upload leaves the hospital; the coordinator can recover the sum, not this update.
            masked = mask_update(
                private_weights, output.num_samples, round_id, self.hospital_id, keys, self._peer_keys(round_id)
            )
            payload = weights_to_bytes(masked)
            metrics["secure_aggregation"] = True
        else:
            payload = weights_to_bytes(private_weights)
        self.last_update_bytes = len(payload)
        return self.transport.submit_update(
            round_id,
            self.hospital_id,
            payload,
            output.num_samples,
            metrics,
            dp_report,
            signature,
        )

    def evaluate_and_report(self, round_id: str, round_info: dict) -> dict:
        modality = round_info["modality"]
        adapter = get_adapter(modality, self.backend)
        payload = self.transport.fetch_global_model(round_id, self.hospital_id, "result")
        self.last_result_model = payload if self.keep_result_model else None
        weights = weights_from_bytes(payload)
        missingness = MissingnessConfig.from_dict(round_info.get("training", {}).get("missingness"))
        evaluation = evaluate_on_local_data(self.data_dir, modality, weights, self.backend, missingness=missingness)
        self.last_evaluation = evaluation

        explanation = None
        if is_tabular(modality) and round_info.get("explainability", {}).get("global_importance", True):
            train, validation, _ = load_training_splits(self.data_dir, modality, missingness)
            explanation = local_global_importance(
                adapter,
                weights,
                validation.X,
                train.X,
                seed=zlib.crc32(f"{self.hospital_id}:{round_id}:explain".encode()) % 100_000,
            )
        return self.transport.submit_evaluation(round_id, self.hospital_id, evaluation.to_dict(), explanation)

    def run_round(self, round_id: str) -> None:
        """Convenience for a stand-alone node: join, train, submit. Evaluation is a second call."""
        stage = "join"
        try:
            info = self.join(round_id)
            stage = "training"
            self.train_and_submit(round_id, info)
        except Exception as exc:
            self.transport.report_failure(round_id, self.hospital_id, stage, str(exc))
            raise
