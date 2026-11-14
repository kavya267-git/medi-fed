"""The agent's "Predict" window: the global model, used on new patients, on this computer.

The patient's values, the prediction and its explanation stay on this machine; nothing here talks to the server.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Optional

from ml.common.tabular import tabular_schema
from ml.prediction.local import DISCLAIMER, format_value

LABELS = {"ehr": "EHR adverse outcome", "diabetes": "Diabetes"}
RISK = "#a3261b"
CALM = "#2a7a57"


def model_title(modality: str, info: dict) -> str:
    version = info.get("model_version")
    shown = f"v{version}" if version is not None else "version unknown"
    return f"{LABELS.get(modality, modality)} · global model {shown} ({info.get('architecture') or 'model'})"


class PredictWindow:
    def __init__(self, parent: tk.Tk, session, data_dir: Callable[[], str], run_in_background, log: Callable[[str], None]):
        self.session = session
        self.data_dir = data_dir
        self.run_in_background = run_in_background
        self.log = log
        self.entries: Dict[str, tk.StringVar] = {}

        self.top = tk.Toplevel(parent)
        self.top.title("Predict with the global model")
        self.top.minsize(720, 640)
        self.top.geometry("780x820")
        outer = ttk.Frame(self.top, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Predict for a new patient", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Uses the latest global model saved on this computer. The values you enter and the result stay "
            "here; nothing is sent to the server.",
            style="Muted.TLabel",
            wraplength=680,
        ).pack(anchor="w", pady=(2, 6))
        tk.Label(outer, text=DISCLAIMER, fg="#8a5a00", wraplength=680, justify="left", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        chooser = ttk.Frame(outer)
        chooser.pack(fill="x")
        ttk.Label(chooser, text="Model").pack(side="left")
        self.models = session.saved_models()
        self.choice = tk.StringVar()
        self.titles = {model_title(m, info): m for m, info in self.models.items()}
        self.combo = ttk.Combobox(chooser, textvariable=self.choice, values=list(self.titles), state="readonly", width=60)
        self.combo.pack(side="left", padx=8)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._build_form())
        self.model_note = ttk.Label(outer, text="", style="Muted.TLabel", wraplength=680)
        self.model_note.pack(anchor="w", pady=(4, 8))

        self.form = ttk.Frame(outer)
        self.form.pack(fill="x")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=10)
        self.predict_button = ttk.Button(buttons, text="Predict", command=self.predict)
        self.predict_button.pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=8)
        ttk.Button(buttons, text="Score a CSV file…", command=self.score_file).pack(side="left")

        self.headline = tk.Label(outer, text="", font=("Segoe UI", 14, "bold"), anchor="w", justify="left")
        self.headline.pack(fill="x")
        self.details = tk.Text(outer, height=12, wrap="word", font=("Segoe UI", 10), state="disabled", relief="flat")
        self.details.pack(fill="both", expand=True, pady=(6, 0))

        if self.titles:
            self.combo.current(0)
            self._build_form()
        else:
            self.model_note.configure(
                text="No global model for EHR or diabetes on this computer yet. Connect, take part in a round, and "
                "the agent keeps the model the round produces."
            )
            self.predict_button.configure(state="disabled")

    # ------------------------------------------------------------ form
    @property
    def modality(self) -> Optional[str]:
        return self.titles.get(self.choice.get())

    def _build_form(self) -> None:
        for child in self.form.winfo_children():
            child.destroy()
        self.entries = {}
        modality = self.modality
        if modality is None:
            return
        info = self.models[modality]
        notes = []
        if info.get("local_f1") is not None:
            notes.append(f"F1 on your held-out records when saved: {info['local_f1']:.2f}")
        if info.get("local_data_simulated"):
            notes.append("your training data for this model was SIMULATED")
        if info.get("downloaded_at"):
            notes.append(f"saved {info['downloaded_at']}")
        self.model_note.configure(text=" · ".join(notes))
        schema = tabular_schema(modality)
        columns = 2
        for index, feature in enumerate(schema.features):
            row, column = divmod(index, columns)
            ttk.Label(self.form, text=f"{feature.label} ({feature.unit})").grid(
                row=row, column=column * 2, sticky="w", padx=(0 if column == 0 else 18, 6), pady=3
            )
            variable = tk.StringVar()
            ttk.Entry(self.form, textvariable=variable, width=12).grid(row=row, column=column * 2 + 1, sticky="w", pady=3)
            self.entries[feature.name] = variable
        hint = (
            "Leave a box empty if it was not measured: the agent estimates it from your own training records."
            if self.data_dir()
            else "Fill every box (choose your data folder to let the agent estimate missing values)."
        )
        ttk.Label(self.form, text=hint, style="Muted.TLabel", wraplength=680).grid(
            row=(len(schema.features) + 1) // columns + 1, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )
        self._show("", "", None)

    def clear(self) -> None:
        for variable in self.entries.values():
            variable.set("")
        self._show("", "", None)

    def _show(self, headline: str, body: str, colour: Optional[str]) -> None:
        self.headline.configure(text=headline, fg=colour or "#1b2430")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", body)
        self.details.configure(state="disabled")

    # ------------------------------------------------------------ predict
    def predict(self) -> None:
        modality = self.modality
        if modality is None:
            return
        values = {name: variable.get().strip() or None for name, variable in self.entries.items()}
        folder = self.data_dir()
        self.predict_button.configure(state="disabled")
        self._show("Predicting…", "", None)

        def work():
            return self.session.predictor(modality, Path(folder) if folder else None).predict_one(values)

        def done(result, error):
            self.predict_button.configure(state="normal")
            if error:
                self._show("Cannot predict", str(error), RISK)
                return
            self._render(result)
            self.log(f"prediction made on this computer ({LABELS.get(modality, modality)}); nothing was sent")

        self.run_in_background(work, done)

    def _render(self, result: dict) -> None:
        percent = round(100 * result["probability"])
        threshold = round(100 * result["threshold"])
        headline = f"{result['prediction']}: {percent}% (threshold {threshold}%)"
        lines = []
        raising = [f for f in result["factors"] if f["direction"] == "raises"][:3]
        lowering = [f for f in result["factors"] if f["direction"] == "lowers"][:3]
        def bullets(factors):
            return [
                f"   • {f['label']}: {format_value(f['value'], f['unit'])}" + (" (estimated)" if f.get("estimated") else "")
                for f in factors
            ]

        if raising:
            lines += ["Raised the risk most:"] + bullets(raising)
        if lowering:
            lines += ["Lowered the risk most:"] + bullets(lowering)
        estimated = [i for i in result["inputs"] if i["estimated"]]
        if estimated:
            lines.append(
                "\nEstimated from your training records (not entered): "
                + ", ".join(f"{i['label']} ≈ {format_value(i['value'], i['unit'])}" for i in estimated)
            )
        if result.get("explanation_method"):
            lines.append(f"\nHow the reasons were found: {result['explanation_method']}.")
        model = result["model"]
        lines.append(f"Model: {model.get('architecture')} v{model.get('version')} from {model.get('round_id')}.")
        lines.append(f"\n{result['disclaimer']}")
        self._show(headline, "\n".join(lines), RISK if result["positive"] else CALM)

    def score_file(self) -> None:
        modality = self.modality
        if modality is None:
            return
        chosen = filedialog.askopenfilename(
            title="Choose a CSV of new patients", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not chosen:
            return
        folder = self.data_dir()
        source = Path(chosen)
        target = source.with_name(f"{source.stem}_predictions.csv")

        def work():
            import pandas as pd

            frame = pd.read_csv(source)
            scored = self.session.score_frame(modality, Path(folder) if folder else None, frame)
            scored.to_csv(target, index=False)
            return scored

        def done(scored, error):
            if error:
                messagebox.showerror("Score a CSV file", str(error), parent=self.top)
                return
            flagged = int(scored["probability"].ge(0.5).sum())
            problems = int((scored["problem"] != "").sum())
            text = f"{len(scored)} patient(s) scored; {flagged} at or above the threshold."
            if problems:
                text += f" {problems} row(s) could not be scored (see the 'problem' column)."
            self._show("File scored", f"{text}\n\nResults written to:\n{target}\n\n{DISCLAIMER}", None)
            self.log(f"scored {len(scored)} patient(s) from {source.name} on this computer; results in {target.name}")

        self._show("Scoring the file…", "", None)
        self.run_in_background(work, done)
