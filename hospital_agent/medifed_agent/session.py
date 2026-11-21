"""What the agent does, independent of the window: identify the hospital, check data, connect, take part.

The data folder is read only on this machine. What reaches the server is what the node has always sent:
model weights, record counts and summary metrics, plus (for a registered hospital) the local data check's
pass / fail and counts per modality.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Every MediFed coordinator in this release trains the PyTorch models; the agent must train the same ones.
os.environ.setdefault("ML_BACKEND", "pytorch")

from ml.common.data_check import check_folder  # noqa: E402
from ml.common.datasets import local_data_summary  # noqa: E402
from ml.common.types import MODALITIES  # noqa: E402
from ml.training.http_transport import CoordinatorError, HttpTransport  # noqa: E402
from ml.training.node_runner import NodeRunner, NodeStartupError  # noqa: E402

from .settings import config_dir  # noqa: E402

BUILT_IN = {f"hospital_{x}" for x in "abcd"}


class AgentError(RuntimeError):
    """Something the operator can fix; the message says what."""


def _friendly(exc: Exception) -> str:
    if isinstance(exc, CoordinatorError):
        if exc.status_code == 403:
            return "The server did not accept this node token. Generate a new one on your hospital's 'My hospital' page."
        return f"The server answered with an error ({exc.status_code}): {exc.detail}"
    import requests

    if isinstance(exc, requests.exceptions.SSLError):
        return (
            "The server's HTTPS certificate could not be verified, so the agent did not connect. Check the address; "
            "if your network inspects HTTPS traffic, ask IT to allow this server."
        )
    if isinstance(exc, requests.RequestException):
        return f"Cannot reach the server ({exc.__class__.__name__}). Check the address and your internet connection."
    return str(exc)


class AgentSession:
    def __init__(self, log: Callable[[str], None]):
        self.log = log
        self.profile: Optional[dict] = None
        self.runner: Optional[NodeRunner] = None
        self._thread: Optional[threading.Thread] = None
        self.take_part = True

    # ---------------------------------------------------------- identity
    def identify(self, server: str, token: str, hospital_id: Optional[str] = None) -> dict:
        """The hospital this token belongs to. Built-in demo hospitals (hospital_a-d) need their id as well."""
        server = server.strip().rstrip("/")
        if not server.startswith(("https://", "http://")):
            raise AgentError("The server address must start with https:// (or http:// on a local network).")
        if not token.strip():
            raise AgentError("Paste your hospital's node token first.")
        transport = HttpTransport(server, token.strip(), timeout=30.0)
        try:
            profile = transport.profile(hospital_id) if hospital_id else transport.whoami()
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            raise AgentError(_friendly(exc)) from exc
        self.profile = profile
        return profile

    # ---------------------------------------------------------- data
    def declared(self) -> List[str]:
        return list((self.profile or {}).get("declared_modalities") or MODALITIES)

    def check_data(self, folder: Path) -> Dict[str, dict]:
        """The local check, on this machine only. Nothing is sent."""
        folder = Path(folder)
        if not folder.is_dir():
            raise AgentError(f"{folder} is not a folder.")
        present = set(local_data_summary(folder))
        modalities = self.declared() if self.profile else sorted(present) or list(MODALITIES)
        return check_folder(folder, modalities)

    def make_sample_data(self) -> Path:
        """SIMULATED records for this hospital in the agent's own folder (fictional, labelled simulated)."""
        from ml.simulation.hospitals import generate_hospital_data, get_simulated_hospital, sample_hospital

        if not self.profile:
            raise AgentError("Check the token first, so the sample data matches your hospital.")
        hospital_id = self.profile["hospital_id"]
        root = config_dir() / "sample-data"
        hospital = (
            get_simulated_hospital(hospital_id)
            if hospital_id in BUILT_IN
            else sample_hospital(hospital_id, self.profile.get("name") or hospital_id)
        )
        generate_hospital_data(root, hospital, modalities=self.declared())
        return root / hospital_id

    # ---------------------------------------------------------- rounds
    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self, server: str, token: str, folder: Path, on_end: Callable[[Optional[str]], None]) -> None:
        if self.connected:
            return
        if not self.profile:
            raise AgentError("Check the token first.")
        runner = NodeRunner(
            self.profile["hospital_id"],
            server.strip().rstrip("/"),
            token.strip(),
            Path(folder),
            log=self.log,
            name=self.profile.get("name"),
            models_dir=config_dir() / "models",
        )
        runner.local_participation = self.take_part
        self.runner = runner

        def work() -> None:
            error = None
            try:
                runner.run()
            except NodeStartupError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = _friendly(exc)
            on_end(error)

        self._thread = threading.Thread(target=work, name="medifed-node", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        if self.runner is not None:
            self.runner.stop()

    # ---------------------------------------------------------- importing an export
    @property
    def mappings_dir(self) -> Path:
        return config_dir() / "mappings"

    def importable_modalities(self) -> List[str]:
        """Tabular kinds of data this hospital can import from its own export (declared ones, if known)."""
        from ml.common.tabular import TABULAR_MODALITIES

        declared = self.declared() if self.profile else list(TABULAR_MODALITIES)
        return [m for m in TABULAR_MODALITIES if m in declared]

    def import_export(self, frame, mapping, folder: Optional[Path], source_name: str) -> dict:
        """Convert the hospital's export with its column matching, write the data folder, check it, keep the
        matching for next time. On this machine only."""
        from ml.common.column_mapping import import_export, save_mapping
        from ml.common.data_check import check_modality

        if folder is None:
            folder = config_dir() / "my-data" / ((self.profile or {}).get("hospital_id") or "hospital")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            result = import_export(frame, mapping, folder, source_name=source_name)
        except ValueError as exc:
            raise AgentError(str(exc)) from exc
        save_mapping(self.mappings_dir, mapping)
        manifest = result["manifest"]
        return {
            "folder": folder,
            "check": check_modality(folder, mapping.modality),
            "train": manifest["splits"]["train"],
            "val": manifest["splits"]["val"],
            "not_recorded": result["report"]["filled_with_reference"],
        }

    def score_frame(self, modality: str, folder: Optional[Path], frame):
        """Score new patients. A file with the hospital's own headings is translated with its saved matching."""
        import pandas as pd

        from ml.common.column_mapping import apply_mapping, load_mapping
        from ml.common.tabular import tabular_schema

        predictor = self.predictor(modality, folder)
        names = tabular_schema(modality).feature_names
        if not set(names) <= set(map(str, frame.columns)):
            mapping = load_mapping(self.mappings_dir, modality)
            present = set(map(str, frame.columns))
            if mapping is not None:
                used = [c for c in mapping.features.values() if c]
                # A measurement missing from this file is treated as not measured (estimated from local records),
                # but most of the matched columns must be there, or this is a different layout.
                if sum(c in present for c in used) < max(1, len(used) // 2 + 1):
                    mapping = None
                else:
                    mapping.features = {k: (c if c in present else None) for k, c in mapping.features.items()}
            if mapping is None:
                raise AgentError(
                    "This file's columns do not match the model's. Use 'Import data…' once with an export that has "
                    "the same headings; the agent remembers the matching and uses it here."
                )
            table, _ = apply_mapping(frame, mapping, with_label=False, fill_not_recorded=False)
            scored = predictor.predict_frame(table)
            extra = [c for c in frame.columns if c not in set(filter(None, mapping.features.values()))]
            return pd.concat([frame[extra].reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
        return predictor.predict_frame(frame)

    def evaluate_frame(self, modality: str, folder: Optional[Path], frame, mapping) -> dict:
        """Test the saved global model on the hospital's own labelled records: accuracy, F1, sensitivity,
        specificity, ROC-AUC and the confusion matrix. On this machine only; the matching is not saved, so an
        evaluation file never changes how the next export is imported."""
        from ml.prediction.evaluate import evaluate_labelled

        predictor = self.predictor(modality, folder)
        try:
            return evaluate_labelled(predictor, frame, mapping)
        except ValueError as exc:
            raise AgentError(str(exc)) from exc

    # ---------------------------------------------------------- prediction
    @property
    def models_dir(self) -> Path:
        return config_dir() / "models"

    def saved_models(self) -> Dict[str, dict]:
        """Global models kept on this computer that can predict for new patients: ``{modality: info}``."""
        from ml.prediction.local import available_models

        return available_models(self.models_dir)

    def ecg_model_info(self) -> Optional[dict]:
        """The saved global ECG model's info, or None when there is none on this computer yet."""
        from ml.prediction.ecg import ecg_model_available

        return ecg_model_available(self.models_dir)

    def ecg_predictor(self):
        """A heartbeat labeller for the saved global ECG model. Runs on this machine only."""
        from ml.prediction.ecg import ECGPredictor, load_ecg_model
        from ml.prediction.local import PredictionUnavailable

        info = self.ecg_model_info() or {}
        key = ("ecg", info.get("sha256"))
        cached = getattr(self, "_ecg_predictor", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            predictor = ECGPredictor(load_ecg_model(self.models_dir))
        except PredictionUnavailable as exc:
            raise AgentError(str(exc)) from exc
        self._ecg_predictor = (key, predictor)
        return predictor

    def predictor(self, modality: str, folder: Optional[Path]):
        """A predictor for the saved global model, filling gaps from this hospital's own training records.

        Rebuilt when the saved model or the data folder changes. Runs on this machine only."""
        from ml.prediction.local import LocalPredictor, PredictionUnavailable, load_local_model

        info = self.saved_models().get(modality) or {}
        key = (modality, info.get("sha256"), str(folder) if folder else None)
        cached = getattr(self, "_predictor", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            predictor = LocalPredictor(load_local_model(self.models_dir, modality), folder)
        except PredictionUnavailable as exc:
            raise AgentError(str(exc)) from exc
        self._predictor = (key, predictor)
        return predictor

    def set_take_part(self, value: bool) -> None:
        self.take_part = value
        if self.runner is not None:
            self.runner.local_participation = value
