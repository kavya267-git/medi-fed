"""HTTP transport: lets a hospital node on another machine reach the coordinator API."""

from __future__ import annotations

import ipaddress
import json
import os
import time
from typing import Optional
from urllib.parse import urlsplit

import requests

#: Set to 1 only for development, to allow plain http:// to any address.
ALLOW_INSECURE_ENV = "MEDIFED_ALLOW_INSECURE_HTTP"
_LOCAL_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal")


class InsecureServerError(ValueError):
    """The coordinator address would send the node token and model updates unencrypted over the internet."""


def is_local_address(host: str) -> bool:
    """Loopback, private-network and link-local IPs, and machine names on the local network."""
    host = (host or "").strip("[]").lower()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(_LOCAL_SUFFIXES)
    return ip.is_loopback or ip.is_private or ip.is_link_local


def check_server_url(url: str) -> str:
    """The address, normalised, if it is safe to send a node token to; otherwise ``InsecureServerError``.

    ``https://`` is always accepted (and its certificate is verified). Plain ``http://`` is accepted only on this
    computer or the local network, where the demo coordinator runs, unless MEDIFED_ALLOW_INSECURE_HTTP=1.
    """
    url = (url or "").strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return url
    if parts.scheme == "http" and parts.hostname:
        if is_local_address(parts.hostname) or os.environ.get(ALLOW_INSECURE_ENV) == "1":
            return url
        raise InsecureServerError(
            f"{parts.hostname} is not on the local network, so the address must start with https:// to keep the "
            "hospital token and model updates encrypted in transit."
        )
    raise InsecureServerError("The server address must start with https:// (or http:// on a local network).")


class CoordinatorError(RuntimeError):
    """The coordinator answered with an error status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"Coordinator returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class HttpTransport:
    def __init__(
        self, base_url: str, node_token: str, timeout: float = 120.0, session: Optional[requests.Session] = None
    ):
        self.base_url = check_server_url(base_url)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.headers = {"X-Hospital-Token": node_token}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/federated{path}"

    @staticmethod
    def _check(response) -> None:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise CoordinatorError(response.status_code, str(detail))

    def join_round(self, round_id: str, hospital_id: str, data_summary: dict) -> dict:
        response = self.session.post(
            self._url(f"/rounds/{round_id}/participate"),
            json={"hospital_id": hospital_id, "data_summary": data_summary},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.json()

    def register_secagg_key(self, round_id: str, hospital_id: str, public_key: str) -> dict:
        response = self.session.post(
            self._url(f"/rounds/{round_id}/secagg-key"),
            json={"hospital_id": hospital_id, "public_key": public_key},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.json()

    def secagg_keys(self, round_id: str, hospital_id: str) -> dict:
        response = self.session.get(
            self._url(f"/rounds/{round_id}/secagg-keys"),
            params={"hospital_id": hospital_id},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.json()

    def fetch_global_model(self, round_id: str, hospital_id: str, which: str) -> bytes:
        response = self.session.get(
            self._url(f"/rounds/{round_id}/global-model"),
            params={"hospital_id": hospital_id, "which": which},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.content

    def submit_update(
        self,
        round_id: str,
        hospital_id: str,
        payload: bytes,
        num_samples: int,
        metrics: dict,
        dp_report: dict,
        architecture_signature: str,
    ) -> dict:
        response = self.session.post(
            self._url(f"/rounds/{round_id}/updates"),
            data={
                "hospital_id": hospital_id,
                "num_samples": str(num_samples),
                "metrics": json.dumps(metrics),
                "dp_report": json.dumps(dp_report),
                "architecture_signature": architecture_signature,
            },
            files={"weights": ("update.npz", payload, "application/octet-stream")},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.json()

    def submit_evaluation(self, round_id: str, hospital_id: str, evaluation: dict, explanation: Optional[dict]) -> dict:
        response = self.session.post(
            self._url(f"/rounds/{round_id}/evaluations"),
            json={"hospital_id": hospital_id, "evaluation": evaluation, "explanation": explanation},
            headers=self.headers,
            timeout=self.timeout,
        )
        self._check(response)
        return response.json()

    def report_failure(self, round_id: str, hospital_id: str, stage: str, message: str) -> None:
        try:
            self.session.post(
                self._url(f"/rounds/{round_id}/failures"),
                json={"hospital_id": hospital_id, "stage": stage, "message": message[:500]},
                headers=self.headers,
                timeout=self.timeout,
            )
        except requests.RequestException:
            pass  # reporting a failure must never mask the original error

    def system_config(self) -> dict:
        """The coordinator's public configuration (mode, model backends and signatures)."""
        response = self.session.get(f"{self.base_url}/api/system/config", timeout=self.timeout)
        self._check(response)
        return response.json()

    def whoami(self) -> dict:
        """The registered hospital this transport's token belongs to (id, name, declared modalities)."""
        response = self.session.get(self._url("/node/whoami"), headers=self.headers, timeout=min(self.timeout, 30.0))
        self._check(response)
        return response.json()

    def profile(self, hospital_id: str) -> dict:
        """The hospital's name and the modalities it declared when it registered."""
        response = self.session.get(
            self._url("/node/profile"), params={"hospital_id": hospital_id}, headers=self.headers, timeout=min(self.timeout, 30.0)
        )
        self._check(response)
        return response.json()

    def report_data_check(self, hospital_id: str, modalities: dict) -> dict:
        """The local data check: pass / fail and counts per modality. No records."""
        response = self.session.post(
            self._url("/node/data-check"),
            json={"hospital_id": hospital_id, "modalities": modalities},
            headers=self.headers,
            timeout=min(self.timeout, 30.0),
        )
        self._check(response)
        return response.json()

    def heartbeat(self, beat: dict) -> dict:
        """Tell the coordinator this node is up. ``beat`` holds machine facts and record counts only."""
        response = self.session.post(
            self._url("/node/heartbeat"), json=beat, headers=self.headers, timeout=min(self.timeout, 15.0)
        )
        self._check(response)
        return response.json()

    def pending(self, hospital_id: str) -> dict:
        """Open rounds this hospital is invited to, and whether its operator lets it take part."""
        response = self.session.get(
            self._url("/node/rounds"), params={"hospital_id": hospital_id}, headers=self.headers, timeout=self.timeout
        )
        self._check(response)
        return response.json()

    def pending_rounds(self, hospital_id: str) -> list:
        """Ids of open rounds this hospital has been invited to and not yet joined."""
        return self.pending(hospital_id)["round_ids"]

    def wait_for_status(
        self, round_id: str, hospital_id: str, statuses, poll_seconds: float = 2.0, max_wait: float = 3600.0
    ) -> dict:
        """Poll the round until it reaches one of ``statuses`` (or fails)."""
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            response = self.session.get(
                self._url(f"/rounds/{round_id}/node-status"),
                params={"hospital_id": hospital_id},
                headers=self.headers,
                timeout=self.timeout,
            )
            self._check(response)
            payload = response.json()
            if payload["status"] in statuses or payload["status"] == "FAILED":
                return payload
            time.sleep(poll_seconds)
        raise TimeoutError(f"Round {round_id} did not reach {statuses} within {max_wait:.0f}s")
