"""The agent's "Evaluate with labelled file" window: how well the global model does on this hospital's own past
patients, whose true answers are known.

It reuses the import window's column matching (the person checks which of their columns feeds each input and
which column holds the true answer), scores every row with the saved global model, compares with the true
answers and shows accuracy, F1, sensitivity, specificity, ROC-AUC and the confusion matrix. The file, the
predictions and the scores stay on this computer; nothing is sent to the server, and the matching used here is
not saved over the import matching.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Optional

from ml.prediction.local import DISCLAIMER

from .import_window import ImportWindow

LABELS = {"ehr": "EHR adverse outcome", "diabetes": "Diabetes"}
LOCAL_ONLY = "Tested on your own records; nothing leaves this computer."
POSITIVE_WORDS = {"ehr": ("adverse outcome", "no adverse outcome"), "diabetes": ("diabetes", "no diabetes")}


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


class EvaluateWindow(ImportWindow):
    WINDOW_TITLE = "Evaluate the global model"
    HEADING = "Evaluate with a labelled file"
    INTRO = (
        "Pick a CSV of past patients whose true answer you already know. Check which of your columns holds each "
        "measurement, and choose the column with the true answer. The agent predicts every row with the saved global "
        "model and compares. " + LOCAL_ONLY
    )
    ACTION = "Evaluate"
    LABEL_TITLE = "True answer (column)"
    FILE_BUTTON = "Choose labelled file…"
    ASK_PATIENT_ID = False

    def modality_options(self) -> List[str]:
        saved = self.session.saved_models()
        return [m for m in self.session.importable_modalities() if m in saved]

    def __init__(self, parent, session, data_dir, run_in_background, log):
        self.result_window: Optional[tk.Toplevel] = None
        self.last_result: Optional[dict] = None
        super().__init__(parent, session, data_dir, lambda _folder: None, run_in_background, log, lambda _r: None)
        if not self.modality_titles:
            self._show(
                "No global EHR or diabetes model on this computer yet. Connect and take part in a round first; the "
                "agent keeps the model each round produces."
            )

    def do_import(self) -> None:  # the "Evaluate" button
        mapping = self.current_mapping()
        if not mapping.label:
            self._show("Choose the column that holds the true answer.")
            return
        frame, source, folder = self.frame, self.source, self.data_dir() or None
        self.import_button.configure(state="disabled")
        self._show("Predicting every row and comparing with the true answers…")

        def work():
            from pathlib import Path

            return self.session.evaluate_frame(mapping.modality, Path(folder) if folder else None, frame, mapping)

        def done(result, error):
            self.import_button.configure(state="normal")
            if error:
                self._show(f"Not evaluated: {error}")
                return
            self.last_result = result
            self._show(
                f"{result['n']} records evaluated: accuracy {_pct(result['accuracy'])}, macro-F1 {_num(result['macro_f1'])}. "
                f"See the result window.\n{LOCAL_ONLY}"
            )
            self.log(f"evaluated the global {LABELS.get(mapping.modality, mapping.modality)} model on "
                     f"{result['n']} labelled records from {source.name} on this computer; nothing was sent")
            self.show_result(result)

        self.run_in_background(work, done)

    # ------------------------------------------------------------ result
    def show_result(self, result: dict) -> None:
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        top = tk.Toplevel(self.top)
        self.result_window = top
        top.title("Evaluation result")
        top.resizable(False, False)
        outer = ttk.Frame(top, padding=16)
        outer.pack(fill="both", expand=True)
        modality = result["modality"]
        yes, no = POSITIVE_WORDS.get(modality, ("yes", "no"))
        model = result["model"]
        ttk.Label(outer, text=f"Global {LABELS.get(modality, modality)} model on your records",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=f"{model.get('architecture')} v{model.get('version')} ({model.get('round_id')}) · {result['n']} records: "
            f"{result['positives']} with {yes}, {result['negatives']} with {no} · threshold {result['threshold']:.0%}",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        table = ttk.Frame(outer)
        table.pack(anchor="w")
        positives, negatives = result["positives"], result["negatives"]
        c = result["confusion"]
        rows = [
            ("Accuracy", _pct(result["accuracy"]), "share of all records the model got right"),
            ("F1 (macro)", _num(result["macro_f1"]), "average of the F1 for each answer"),
            (f"F1 for '{yes}'", _num(result["f1_positive"]), "balance of finding cases and not raising false alarms"),
            ("Sensitivity", _pct(result["sensitivity"]), f"found {c['tp']} of the {positives} records with {yes}"),
            ("Specificity", _pct(result["specificity"]), f"cleared {c['tn']} of the {negatives} records with {no}"),
            ("ROC-AUC", _num(result["roc_auc"]), "ranking quality, threshold-free (0.5 = chance)"),
        ]
        for i, (name, value, hint) in enumerate(rows):
            ttk.Label(table, text=name, font=("Segoe UI", 10, "bold")).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(table, text=value, font=("Segoe UI", 10)).grid(row=i, column=1, sticky="e", padx=(14, 14))
            ttk.Label(table, text=hint, style="Muted.TLabel").grid(row=i, column=2, sticky="w")

        ttk.Label(outer, text="Confusion matrix", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))
        grid = tk.Frame(outer)
        grid.pack(anchor="w")
        cells = [
            ["", f"Predicted {yes}", f"Predicted {no}"],
            [f"Truly {yes}", str(c["tp"]), str(c["fn"])],
            [f"Truly {no}", str(c["fp"]), str(c["tn"])],
        ]
        good = {(1, 1), (2, 2)}
        for r, line in enumerate(cells):
            for col, text in enumerate(line):
                header = r == 0 or col == 0
                colour = "#eef2fb" if header else ("#e3f1ea" if (r, col) in good else "#fbe9e7")
                tk.Label(grid, text=text, width=18 if col == 0 else 21, bg=colour, relief="solid", borderwidth=1,
                         font=("Segoe UI", 10, "bold" if header else "normal"), padx=6, pady=4).grid(row=r, column=col)

        notes = []
        if result.get("small_sample"):
            notes.append(f"Only {result['n']} records: these numbers can change a lot by chance; test on more records "
                         "before relying on them.")
        if result.get("rows_with_estimated_values"):
            notes.append(f"{result['rows_with_estimated_values']} record(s) had values missing; they were estimated "
                         f"({result['gaps_filled']}).")
        report = result.get("report") or {}
        if report.get("dropped_without_label"):
            notes.append(f"{report['dropped_without_label']} row(s) without a true answer were left out.")
        if report.get("out_of_bounds_treated_as_missing"):
            notes.append("Values outside the plausible range were treated as not measured: "
                         + ", ".join(f"{k} ({v})" for k, v in report["out_of_bounds_treated_as_missing"].items()))
        if model.get("trained_on_simulated_data"):
            notes.append("Your own training data for this model was SIMULATED.")
        tk.Label(outer, text=LOCAL_ONLY, fg="#2a7a57", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 0))
        for note in notes:
            ttk.Label(outer, text=note, style="Muted.TLabel", wraplength=640).pack(anchor="w", pady=(2, 0))
        tk.Label(outer, text=DISCLAIMER, fg="#8a5a00", wraplength=640, justify="left",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Button(outer, text="Close", command=top.destroy).pack(anchor="e", pady=(10, 0))
