"""The agent's "Import data" window: match the hospital's own export to the model's inputs, on this computer.

The person picks their CSV export, checks the suggested column for every input (or marks it "not recorded"),
confirms how text values and units are translated, and imports. The agent writes the training and held-out
files into the data folder, runs the local data check, and saves the matching for the next export. Nothing is
sent to the server.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

from ml.common.column_mapping import ColumnMapping, headings_signature, inspect_export, load_mapping, suggest_mapping
from ml.common.tabular import tabular_schema

NOT_RECORDED = "— not recorded —"
NONE = "— none —"
LABELS = {"ehr": "EHR (adverse outcome)", "diabetes": "Diabetes"}


class ImportWindow:
    def __init__(
        self,
        parent: tk.Tk,
        session,
        data_dir: Callable[[], str],
        set_data_dir: Callable[[str], None],
        run_in_background,
        log: Callable[[str], None],
        on_checked: Callable[[dict], None],
    ):
        self.session = session
        self.data_dir = data_dir
        self.set_data_dir = set_data_dir
        self.run_in_background = run_in_background
        self.log = log
        self.on_checked = on_checked
        self.frame = None
        self.source: Optional[Path] = None
        self.feature_vars: Dict[str, tk.StringVar] = {}
        self.sample_labels: Dict[str, ttk.Label] = {}
        self.choice_vars: Dict[str, tk.BooleanVar] = {}
        self.zero_missing = tk.BooleanVar(value=False)

        self.top = tk.Toplevel(parent)
        self.top.title("Import your hospital's data")
        self.top.minsize(900, 700)
        self.top.geometry("1080x880")
        outer = ttk.Frame(self.top, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Import your hospital's export", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Pick your CSV export and tell the agent which of your columns holds each measurement. Suggestions "
            "are pre-selected; check every row. The file is read and converted on this computer only.",
            style="Muted.TLabel",
            wraplength=1020,
        ).pack(anchor="w", pady=(2, 10))

        top_row = ttk.Frame(outer)
        top_row.pack(fill="x")
        ttk.Label(top_row, text="Kind of data").pack(side="left")
        self.modality = tk.StringVar()
        options = session.importable_modalities()
        self.modality_titles = {LABELS.get(m, m): m for m in options}
        self.modality_box = ttk.Combobox(
            top_row, textvariable=self.modality, values=list(self.modality_titles), state="readonly", width=26
        )
        self.modality_box.pack(side="left", padx=8)
        self.modality_box.bind("<<ComboboxSelected>>", lambda _e: self._suggest())
        ttk.Button(top_row, text="Choose export file…", command=self.choose_file).pack(side="left", padx=8)
        self.file_note = ttk.Label(top_row, text="No file chosen", style="Muted.TLabel")
        self.file_note.pack(side="left", padx=8)
        if options:
            self.modality_box.current(0)

        self.grid = ttk.Frame(outer)
        self.grid.pack(fill="x", pady=(12, 6))
        self.extra = ttk.Frame(outer)
        self.extra.pack(fill="x", pady=(4, 6))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=8)
        self.import_button = ttk.Button(buttons, text="Import into my data folder", command=self.do_import, state="disabled")
        self.import_button.pack(side="left")
        self.result = tk.Text(outer, height=7, wrap="word", font=("Segoe UI", 10), state="disabled", relief="flat")
        self.result.pack(fill="both", expand=True)

    @property
    def selected(self) -> Optional[str]:
        return self.modality_titles.get(self.modality.get())

    # ------------------------------------------------------------ file
    def choose_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose your export (CSV)", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], parent=self.top
        )
        if not chosen:
            return
        import pandas as pd

        try:
            frame = pd.read_csv(chosen)
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            messagebox.showerror("Import", f"Cannot read {Path(chosen).name}: {exc}", parent=self.top)
            return
        if frame.empty:
            messagebox.showerror("Import", "The file has no rows.", parent=self.top)
            return
        self.frame, self.source = frame, Path(chosen)
        self.file_note.configure(text=f"{self.source.name}: {len(frame)} rows, {len(frame.columns)} columns")
        self._suggest()

    def _suggest(self) -> None:
        modality = self.selected
        if self.frame is None or modality is None:
            return
        columns = [str(c) for c in self.frame.columns]
        saved = load_mapping(self.session.mappings_dir, modality, columns)
        if saved is not None:
            saved.reasons = {name: "your saved matching" for name, column in saved.features.items() if column}
        mapping = saved or suggest_mapping(modality, columns, self.frame)
        self._build_grid(modality, columns, mapping)

    # ------------------------------------------------------------ mapping grid
    def _build_grid(self, modality: str, columns: List[str], mapping: ColumnMapping) -> None:
        for child in self.grid.winfo_children():
            child.destroy()
        self.feature_vars, self.sample_labels, self.reason_labels = {}, {}, {}
        self.reasons = dict(mapping.reasons)
        schema = tabular_schema(modality)
        choices = [NOT_RECORDED] + columns
        ttk.Label(self.grid, text="Model input", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self.grid, text="Your column", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w")
        ttk.Label(self.grid, text="How it was matched", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="w")
        ttk.Label(self.grid, text="First values in your file", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=3, sticky="w", padx=(10, 0)
        )
        for row, feature in enumerate(schema.features, start=1):
            ttk.Label(self.grid, text=f"{feature.label} ({feature.unit})").grid(row=row, column=0, sticky="w", pady=2)
            variable = tk.StringVar(value=mapping.features.get(feature.name) or NOT_RECORDED)
            box = ttk.Combobox(self.grid, textvariable=variable, values=choices, state="readonly", width=26)
            box.grid(row=row, column=1, sticky="w", padx=8, pady=2)
            box.bind("<<ComboboxSelected>>", lambda _e, n=feature.name: self._picked(n))
            self.feature_vars[feature.name] = variable
            reason = tk.Label(self.grid, text="", anchor="w", font=("Segoe UI", 9))
            reason.grid(row=row, column=2, sticky="w")
            self.reason_labels[feature.name] = reason
            sample = ttk.Label(self.grid, text="", style="Muted.TLabel")
            sample.grid(row=row, column=3, sticky="w", padx=(10, 0))
            self.sample_labels[feature.name] = sample
        row = len(schema.features) + 1
        ttk.Separator(self.grid).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(self.grid, text="Outcome (label)").grid(row=row + 1, column=0, sticky="w")
        self.label_var = tk.StringVar(value=mapping.label or NONE)
        label_box = ttk.Combobox(self.grid, textvariable=self.label_var, values=[NONE] + columns, state="readonly", width=30)
        label_box.grid(row=row + 1, column=1, sticky="w", padx=8, pady=2)
        label_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh())
        ttk.Label(self.grid, text=schema_label_hint(modality), style="Muted.TLabel").grid(
            row=row + 1, column=2, columnspan=2, sticky="w"
        )
        ttk.Label(self.grid, text="Patient ID (optional)").grid(row=row + 2, column=0, sticky="w")
        self.group_var = tk.StringVar(value=mapping.group or NONE)
        ttk.Combobox(self.grid, textvariable=self.group_var, values=[NONE] + columns, state="readonly", width=30).grid(
            row=row + 2, column=1, sticky="w", padx=8, pady=2
        )
        ttk.Label(self.grid, text="keeps a patient's rows together when splitting", style="Muted.TLabel").grid(
            row=row + 2, column=2, columnspan=2, sticky="w"
        )
        self.saved_choices = {"label": set(mapping.label_positive), "male": set(mapping.male_values),
                              "conversions": set(mapping.conversions)}
        self.zero_missing.set(mapping.zero_means_missing)
        from_saved = "your saved matching" in mapping.reasons.values()
        if modality == "diabetes" and not from_saved and not mapping.zero_means_missing:
            # A 0 glucose, blood pressure or BMI is impossible: the export uses 0 for "not measured".
            probe = inspect_export(self.frame, mapping)
            self.zero_missing.set(bool(probe["impossible_zeros"]))
        self._refresh(first=True)

    def current_mapping(self) -> ColumnMapping:
        modality = self.selected
        features = {name: (None if var.get() == NOT_RECORDED else var.get()) for name, var in self.feature_vars.items()}
        label = None if self.label_var.get() == NONE else self.label_var.get()
        group = None if self.group_var.get() == NONE else self.group_var.get()
        def picked(prefix: str) -> List[str]:
            return [k.split(":", 1)[1] for k, v in self.choice_vars.items() if k.startswith(prefix) and v.get()]

        return ColumnMapping(
            modality,
            features,
            label=label,
            group=group,
            label_positive=picked("label:"),
            male_values=picked("male:"),
            conversions=picked("conv:"),
            zero_means_missing=bool(self.zero_missing.get()) and modality == "diabetes",
            headings=headings_signature(self.frame.columns),
        )

    def _picked(self, name: str) -> None:
        self.reasons[name] = "chosen by you"
        self._refresh()

    def _refresh(self, first: bool = False) -> None:
        if self.frame is None:
            return
        for name, variable in self.feature_vars.items():
            column = variable.get()
            if column == NOT_RECORDED:
                self.reason_labels[name].configure(text="not recorded", fg="#5d6b7c")
                self.sample_labels[name].configure(text="the shared model relies on your other inputs")
            else:
                reason = self.reasons.get(name, "chosen by you")
                colour = "#8a5a00" if "check" in reason else "#2a7a57"
                self.reason_labels[name].configure(text=reason, fg=colour)
                values = self.frame[column].dropna().astype(str).head(4).tolist()
                self.sample_labels[name].configure(text=", ".join(values) or "(empty)")
        previous = {k: v.get() for k, v in self.choice_vars.items()}
        mapping = self.current_mapping()
        report = inspect_export(self.frame, mapping)
        for name, filled in report.get("mostly_empty", {}).items():
            self.reason_labels[name].configure(text=f"only {filled:.0%} filled: treated as not recorded", fg="#8a5a00")
        for child in self.extra.winfo_children():
            child.destroy()
        self.choice_vars = {}

        def boxes(title: str, prefix: str, values: List[str], default) -> None:
            if not values:
                return
            ttk.Label(self.extra, text=title, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            line = ttk.Frame(self.extra)
            line.pack(fill="x")
            for value in values:
                key = f"{prefix}{value}"
                state = previous.get(key, default(value))
                variable = tk.BooleanVar(value=state)
                ttk.Checkbutton(line, text=value, variable=variable).pack(side="left", padx=(0, 10))
                self.choice_vars[key] = variable

        saved = self.saved_choices
        if mapping.label:
            boxes("Which outcome values mean YES (1)?", "label:", report["label_values"],
                  lambda v: v in saved["label"] or v.strip().lower() in ("1", "yes", "y", "true", "died", "positive",
                                                                          "tested_positive", "diabetic"))
        if mapping.modality == "ehr":
            boxes("Which values of the sex column mean MALE (1)?", "male:", report["sex_values"],
                  lambda v: v in saved["male"] or v.strip().lower() in ("m", "male", "1"))
        if report["conversions"]:
            ttk.Label(self.extra, text="Units that look different from the model's", font=("Segoe UI", 9, "bold")).pack(
                anchor="w", pady=(6, 0)
            )
            for item in report["conversions"]:
                key = f"conv:{item['key']}"
                variable = tk.BooleanVar(value=previous.get(key, True))
                text = (f"'{item['column']}' has a typical value of {item['median']:g}: convert {item['description']}")
                ttk.Checkbutton(self.extra, text=text, variable=variable).pack(anchor="w")
                self.choice_vars[key] = variable
        if mapping.modality == "diabetes":
            zeros = report.get("impossible_zeros", [])
            hint = f" - found zeros in: {', '.join(zeros)}" if zeros else ""
            ttk.Checkbutton(
                self.extra,
                text="A 0 in glucose, blood pressure, skin fold, insulin or BMI means 'not measured' (as in the PIMA "
                f"file){hint}",
                variable=self.zero_missing,
            ).pack(anchor="w", pady=(6, 0))
        missing = set(mapping.not_recorded()) | set(report.get("mostly_empty", {}))
        if missing:
            total = len(mapping.features)
            ttk.Label(
                self.extra,
                text=(
                    f"Your hospital records {total - len(missing)} of the model's {total} inputs. That is fine: the "
                    "shared model learned those inputs at other hospitals and predicts from the ones you have. Inputs "
                    "you do not record are filled with a typical value and contribute nothing at your hospital."
                ),
                style="Muted.TLabel",
                wraplength=1020,
            ).pack(anchor="w", pady=(6, 0))
        self.import_button.configure(state="normal" if mapping.label else "disabled")

    # ------------------------------------------------------------ import
    def do_import(self) -> None:
        mapping = self.current_mapping()
        frame, source = self.frame, self.source
        self.import_button.configure(state="disabled")
        self._show("Importing and checking…")

        def work():
            return self.session.import_export(frame, mapping, self.data_dir() or None, source.name)

        def done(result, error):
            self.import_button.configure(state="normal")
            if error:
                self._show(f"Not imported: {error}")
                return
            folder, check = result["folder"], result["check"]
            self.set_data_dir(str(folder))
            lines = [
                f"Imported {result['train']} training and {result['val']} held-out records into {folder}.",
                "Local data check: " + ("PASSED" if check["passed"] else "FAILED"),
            ]
            lines += [f"  • {p}" for p in check["problems"]]
            if result["not_recorded"]:
                lines.append("Not recorded (filled with reference values): " + ", ".join(result["not_recorded"]))
            lines.append("Your column matching is saved; the next export with the same headings is pre-filled.")
            self._show("\n".join(lines))
            self.log(f"imported {source.name} as {mapping.modality} on this computer; check "
                     f"{'passed' if check['passed'] else 'failed'}")
            self.on_checked({mapping.modality: check})

        self.run_in_background(work, done)

    def _show(self, text: str) -> None:
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("end", text)
        self.result.configure(state="disabled")


def schema_label_hint(modality: str) -> str:
    return "1 = diabetes, 0 = no diabetes" if modality == "diabetes" else "1 = adverse outcome, 0 = none"

