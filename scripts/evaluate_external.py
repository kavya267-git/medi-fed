"""External validation: score the saved global model on labelled patients from a hospital that never took part.

Usage (repo root):
    python scripts/evaluate_external.py --modality diabetes --csv data/external/frankfurt_diabetes.csv \
        --label Outcome [--model <global_latest.npz>]

What it does, in order:
1. Loads the global model like the agent does (``global_latest.npz`` + ``.json``; SHA-256 and architecture
   signature checked). Default: the model the source agent saved in ``MEDIFED_AGENT_HOME`` (or the agent's
   normal settings folder).
2. Matches the file's columns to the model's inputs with ``ml.common.column_mapping`` (common names, unit
   conversion such as glucose mmol/L -> mg/dL, impossible zeros -> not measured) and prints the matching.
3. Overlap check (diabetes): drops every row whose eight measurements equal a record of the PIMA file the
   model was trained on, and reports how many. Exact repeated rows are reported and, for the main result,
   counted once.
4. Fills gaps by KNN imputation fitted on the evaluated records' measurements (never their answers); the model
   standardises with its fixed reference values, as in training. Threshold 0.5.
5. Baselines on the same rows: (a) a model trained only on ONE PIMA hospital's training split - the smallest and
   the largest - for the same number of rounds x local epochs as the global model, three seeds; (b) always
   predicting the majority class of the training data.
6. Writes ``<out>.json`` and a bar chart ``<out>.png`` (default ``docs/results/external_diabetes``).

Research prototype, not a medical device; these numbers are not clinical accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hospital_agent")]
os.environ.setdefault("ML_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common.column_mapping import CONVERSIONS, inspect_export, suggest_mapping  # noqa: E402
from ml.common.tabular import tabular_schema  # noqa: E402
from ml.common.types import TrainConfig  # noqa: E402
from ml.prediction.evaluate import (  # noqa: E402
    binary_metrics,
    bootstrap_ci,
    duplicate_mask,
    fill_gaps,
    labelled_table,
    overlap_mask,
)
from ml.prediction.local import load_model_file  # noqa: E402
from ml.registry import get_adapter  # noqa: E402
from ml.training.local_trainer import load_training_splits, train_on_local_data  # noqa: E402

METRICS = ("accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc")
METRIC_TITLES = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "sensitivity": "Sensitivity",
                 "specificity": "Specificity", "roc_auc": "ROC-AUC"}


def default_model(modality: str) -> Path:
    from medifed_agent.settings import config_dir

    return config_dir() / "models" / modality / "global_latest.npz"


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def print_mapping(mapping, report) -> None:
    schema = tabular_schema(mapping.modality)
    print("\nColumn matching (model input <- file column, why):")
    for feature in schema.features:
        column = mapping.features.get(feature.name)
        why = mapping.reasons.get(feature.name, "")
        print(f"  {feature.label + ' (' + feature.unit + ')':38s} <- {column or '— not recorded —':28s} {why}")
    print(f"  {'True answer (label)':38s} <- {mapping.label}   values meaning yes: {mapping.label_positive or ['1']}")
    print(f"  0 means 'not measured' in {', '.join(report['impossible_zeros']) or '-'}: {mapping.zero_means_missing}")
    print(f"  unit conversions applied: {[CONVERSIONS[k][3] for k in mapping.conversions] or 'none'}")


def own_only_models(modality, hospitals_root: Path, rounds: int, seeds, log=print) -> dict:
    """For the smallest and the largest hospital: weights trained only on its own training split."""
    sizes = {}
    for folder in sorted(p for p in hospitals_root.iterdir() if (p / modality / "manifest.json").exists()):
        sizes[folder.name] = len(load_training_splits(folder, modality)[0])
    if len(sizes) < 2:
        raise SystemExit(f"Need at least two hospital folders with {modality} data under {hospitals_root}")
    smallest, largest = min(sizes, key=sizes.get), max(sizes, key=sizes.get)
    adapter = get_adapter(modality, "pytorch")
    out = {}
    for role, hospital in (("smallest", smallest), ("largest", largest)):
        runs = []
        for seed in seeds:
            weights = adapter.init_weights(seed)
            for r in range(rounds):
                config = TrainConfig(2, 32, 0.003, seed * 1000 + r, class_weighting="balanced")
                weights = train_on_local_data(hospitals_root / hospital, modality, weights, config, "pytorch").weights
            runs.append(weights)
        log(f"  own-only model trained: {hospital} ({role}, {sizes[hospital]} training records), seeds {list(seeds)}")
        out[role] = {"hospital": hospital, "train_records": sizes[hospital], "weights": runs}
    return out, sizes


def majority_class(hospitals_root: Path, modality: str) -> int:
    labels = np.concatenate([
        load_training_splits(p, modality)[0].y
        for p in hospitals_root.iterdir() if (p / modality / "manifest.json").exists()
    ])
    return int(np.bincount(labels.astype(int)).argmax())


def summarise_runs(runs: list) -> dict:
    """Mean, min and max of each metric over seeds."""
    out = {}
    for metric in METRICS + ("f1_positive",):
        values = [r[metric] for r in runs if r[metric] is not None]
        out[metric] = {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))} \
            if values else None
    return out


def evaluate_set(name, X_raw, y, modality, model, own, majority, ci=False) -> dict:
    filled, how = fill_gaps(X_raw, modality)
    adapter = model.adapter
    p_global = adapter.predict_proba(model.weights, filled)[:, 1]
    result = {"rows": int(len(y)), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
              "rows_with_estimated_values": int(np.isnan(X_raw).any(axis=1).sum()), "gaps_filled": how}
    result["global"] = binary_metrics(y, p_global)
    if ci:
        result["global_ci95"] = {m: bootstrap_ci(y, p_global, m) for m in METRICS}
    for role, item in own.items():
        runs = [binary_metrics(y, adapter.predict_proba(w, filled)[:, 1]) for w in item["weights"]]
        result[f"own_{role}"] = {"hospital": item["hospital"], "train_records": item["train_records"],
                                 "per_seed": runs, "summary": summarise_runs(runs)}
    result["majority"] = binary_metrics(y, np.full(len(y), float(majority)))
    result["majority"]["predicts"] = majority
    print(f"\n== {name}: {len(y)} rows ({int(y.sum())} positive, {len(y) - int(y.sum())} negative)")
    head = f"  {'model':34s}" + "".join(f"{METRIC_TITLES[m]:>13s}" for m in METRICS)
    print(head)

    def line(title, metrics):
        cells = "".join(f"{'-' if metrics[m] is None else format(metrics[m], '.3f'):>13s}" for m in METRICS)
        print(f"  {title:34s}{cells}")

    line("Global FedAvg model", result["global"])
    for role in own:
        item = result[f"own_{role}"]
        line(f"Own-only {item['hospital']} ({item['train_records']} train), mean",
             {m: (item["summary"][m] or {}).get("mean") for m in METRICS})
    line("Majority class (always 'no')" if majority == 0 else "Majority class", result["majority"])
    c = result["global"]["confusion"]
    print(f"  global confusion matrix: TN {c['tn']}  FP {c['fp']}  FN {c['fn']}  TP {c['tp']}")
    return result


def plot(out: dict, path: Path) -> None:
    """Left: what the file is made of. Right: the scores, on whichever set is large enough to show (with the
    whole-file scores clearly marked as not an external test when the clean set is too small)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlap, rows = out["overlap"], out["data"]["rows"]
    trained = overlap["copies_of_hospital_training_split_records"]
    held = overlap["copies_of_held_out_records_only"]
    parts = []
    if trained is not None:
        other = overlap["rows_matching_training_records"] - trained - (held or 0)
        parts += [("copies of PIMA records the\nglobal model trained on", trained, "#a3261b"),
                  ("copies of PIMA held-out records", held or 0, "#d9822b")]
        if other:
            parts.append(("copies of other PIMA records", other, "#e8b04a"))
    else:
        parts.append(("copies of PIMA records", overlap["rows_matching_training_records"], "#a3261b"))
    parts += [("repeats of the rows below", overlap["unseen_rows"] - overlap["unseen_distinct_rows"], "#9aa3b2"),
              ("distinct rows not in PIMA", overlap["unseen_distinct_rows"], "#24489a")]
    main_name, whole_name = list(out["results"])
    main = out["results"][main_name]
    shown_name = whole_name if main["too_small_to_conclude"] else main_name
    shown = out["results"][shown_name]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, gridspec_kw={"width_ratios": [1, 2.2]})
    bottom = 0
    for label, count, colour in parts:
        left.bar(0, count, 0.5, bottom=bottom, color=colour, label=f"{label}: {count}")
        bottom += count
    left.set_xlim(-0.6, 0.6)
    left.set_xticks([])
    left.set_ylabel("rows in the file")
    left.set_title(f"What the {rows}-row file contains", fontsize=10)
    left.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    left.spines[["top", "right", "bottom"]].set_visible(False)

    series = [("Global FedAvg model", {m: shown["global"][m] for m in METRICS}, None, "#24489a")]
    for role, colour in (("smallest", "#c98a1b"), ("largest", "#2a7a57")):
        item = shown[f"own_{role}"]
        summary = item["summary"]
        series.append((f"Own-only: {item['hospital']} ({item['train_records']} train)",
                       {m: (summary[m] or {}).get("mean") for m in METRICS},
                       {m: (summary[m]["mean"] - summary[m]["min"], summary[m]["max"] - summary[m]["mean"])
                        if summary[m] else (0, 0) for m in METRICS},
                       colour))
    series.append(("Majority class", {m: shown["majority"][m] for m in METRICS}, None, "#9aa3b2"))
    width = 0.8 / len(series)
    x = np.arange(len(METRICS))
    for i, (label, values, errors, colour) in enumerate(series):
        heights = [values[m] if values[m] is not None else 0 for m in METRICS]
        err = None
        if errors:
            err = np.array([[errors[m][0] for m in METRICS], [errors[m][1] for m in METRICS]])
        right.bar(x + (i - (len(series) - 1) / 2) * width, heights, width, label=label, color=colour,
                  yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
    right.set_xticks(x, [METRIC_TITLES[m] for m in METRICS])
    right.set_ylim(0, 1)
    right.set_ylabel("score (0-1)")
    if main["too_small_to_conclude"]:
        title = (f"Scores on the whole file - NOT an external test: {overlap['rows_matching_training_records']} of "
                 f"{rows} rows are PIMA records\n(only {main['rows']} distinct rows are new: too few to report)")
    else:
        title = f"Scores on {shown['rows']} patients not in the training cohort"
    right.set_title(title, fontsize=9, color="#a3261b" if main["too_small_to_conclude"] else "#1b2430")
    right.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", alpha=0.3)
    fig.text(0.01, 0.005, "Own-only bars: mean of 3 seeds, whiskers min-max. Majority class: ROC-AUC 0.5 by definition. "
             "Research prototype; not clinical accuracy.", fontsize=6, color="#5b6478")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modality", default="diabetes", choices=("diabetes",))
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--label", required=True, help="the column holding the true answer")
    parser.add_argument("--positive", nargs="*", default=[], help="text values of the label that mean yes")
    parser.add_argument("--model", type=Path, help="global_latest.npz (default: the agent's saved model)")
    parser.add_argument("--training-reference", type=Path, default=ROOT / "data" / "raw" / "pima" / "diabetes.arff",
                        help="the PIMA ARFF file, for the overlap check (MD5-checked)")
    parser.add_argument("--hospitals", type=Path, default=ROOT / "data" / "hospitals",
                        help="prepared PIMA hospital folders, for the own-only baselines")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, help="rounds for the own-only models (default: the global model's version)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "external_diabetes")
    args = parser.parse_args()
    modality = args.modality
    seeds = [int(s) for s in args.seeds.split(",")]

    model_path = args.model or default_model(modality)
    model = load_model_file(model_path, modality, "pytorch")
    info = model.info
    print(f"Global model: {model_path}\n  {info.get('architecture')} v{info.get('model_version')} from "
          f"{info.get('round_id')}; SHA-256 {info.get('sha256', '')[:16]}… verified; architecture signature verified")

    frame = pd.read_csv(args.csv)
    print(f"\nExternal file: {args.csv} ({len(frame)} rows, {len(frame.columns)} columns), MD5 {md5(args.csv)}")
    mapping = suggest_mapping(modality, [str(c) for c in frame.columns], frame)
    mapping.label = args.label
    mapping.label_positive = list(args.positive)
    report = inspect_export(frame, mapping)
    mapping.zero_means_missing = bool(report["impossible_zeros"])
    mapping.conversions = [item["key"] for item in report["conversions"]]
    print_mapping(mapping, report)
    missing_inputs = mapping.not_recorded()
    if missing_inputs:
        print(f"  not in the file (estimated for every row): {', '.join(missing_inputs)}")

    # The file as the model's inputs: 0 = not measured -> gap, units converted. Overlap is checked on these values,
    # with the training cohort read the same way (read_pima_arff also turns its zeros into gaps), so a 0 in one
    # file and a gap in the other count as equal.
    X_all, y_all, table_report = labelled_table(frame, mapping)
    if len(y_all) != len(frame):
        raise SystemExit("rows without a label: not expected for this file")
    from ml.common.real_data import PIMA_MD5, fetch_pima, read_pima_arff

    reference_X, reference_y = read_pima_arff(fetch_pima(args.training_reference))
    reference_X = np.asarray(reference_X, dtype=np.float64)
    in_training = overlap_mask(X_all, reference_X)
    same_label = int(sum(
        bool((overlap_mask(reference_X, X_all[i:i + 1]) & (np.asarray(reference_y) == y_all[i])).any())
        for i in np.where(in_training)[0]
    ))
    distinct_matched = int((~duplicate_mask(X_all[in_training], np.zeros(int(in_training.sum())))).sum())
    trained_on = held_out = None
    split_files = {s: [p / modality / f"{s}.csv" for p in sorted(args.hospitals.iterdir())
                       if (p / modality / f"{s}.csv").exists()] for s in ("train", "val")}
    if split_files["train"]:
        names = tabular_schema(modality).feature_names
        split_X = {s: np.concatenate([pd.read_csv(f)[names].to_numpy(dtype=np.float64) for f in fs])
                   for s, fs in split_files.items() if fs}
        trained_on = int(overlap_mask(X_all, split_X["train"]).sum())
        held_out = int((overlap_mask(X_all, split_X["val"]) & ~overlap_mask(X_all, split_X["train"])).sum()) \
            if "val" in split_X else None
    repeated = duplicate_mask(X_all, y_all)
    unseen = ~in_training
    unique_unseen = np.zeros(len(frame), dtype=bool)
    unseen_rows = np.where(unseen)[0]
    unique_unseen[unseen_rows[~duplicate_mask(X_all[unseen_rows], y_all[unseen_rows])]] = True
    # How close the rest is to the training cohort: most measurements equal to one PIMA record.
    filled_ref = np.nan_to_num(np.round(reference_X, 3), nan=-1.0)
    closeness = [int((filled_ref == np.nan_to_num(np.round(row, 3), nan=-1.0)).sum(axis=1).max())
                 for row in X_all[unique_unseen]]
    print("\nOverlap check against the training cohort (PIMA, MD5 " + PIMA_MD5 + "):")
    print(f"  rows whose 8 measurements equal a PIMA record: {int(in_training.sum())} of {len(frame)} "
          f"({distinct_matched} distinct PIMA records), {same_label} of them with the same outcome too -> EXCLUDED")
    if trained_on is not None:
        print(f"    of these, copies of records in the hospitals' training splits: {trained_on}; "
              f"of held-out records only: {held_out}")
    print(f"  exact repeated rows in the whole file: {int(repeated.sum())}")
    print(f"  left after excluding the overlap: {int(unseen.sum())} rows, of which {int(unique_unseen.sum())} distinct")
    print(f"  those distinct rows share at most this many of 8 values with one PIMA record: {sorted(closeness)}")
    print(f"  gaps after matching (0 = not measured, out-of-range values): {table_report['values_missing']} values; "
          f"out of range -> missing: {table_report['out_of_bounds_treated_as_missing'] or 'none'}")

    rounds = args.rounds or int(info.get("model_version") or 8)
    print(f"\nTraining own-only baselines ({rounds} rounds x 2 local epochs, as the global rounds) ...")
    own, sizes = own_only_models(modality, args.hospitals, rounds, seeds)
    majority = majority_class(args.hospitals, modality)

    main_name = "main: not in the training cohort, distinct"
    whole_name = "whole file incl. PIMA copies (NOT an external test)"
    results = {
        main_name: evaluate_set(main_name, X_all[unique_unseen], y_all[unique_unseen], modality, model, own, majority,
                                ci=True),
        whole_name: evaluate_set(whole_name, X_all, y_all, modality, model, own, majority),
    }
    main_result = results[main_name]
    too_small = main_result["rows"] < 100
    main_result["too_small_to_conclude"] = too_small
    if too_small:
        print(f"\nOnly {main_result['rows']} records are not copies of the training cohort: far too few to say anything "
              "about performance on a new population. No external-validation claim can be made from this file.")
    ci = main_result["global_ci95"]
    print("Global model, main set, 95% bootstrap intervals: " + ", ".join(
        f"{METRIC_TITLES[m]} {main_result['global'][m]:.3f} " + (f"[{ci[m][0]:.3f}-{ci[m][1]:.3f}]" if ci[m] else "[n/a]")
        for m in METRICS if main_result["global"][m] is not None))

    out = {
        "what": "External validation of the global diabetes model on a cohort that took no part in training",
        "verdict": ("NOT POSSIBLE with this file: almost every row is a record of the training cohort (PIMA)"
                    if too_small else "external test set large enough to report"),
        "data": {
            "file": (args.csv.relative_to(ROOT) if args.csv.is_absolute() and ROOT in args.csv.parents else args.csv).as_posix(),
            "md5": md5(args.csv), "rows": int(len(frame)), "simulated": False,
            "positives": int(y_all.sum()), "negatives": int(len(y_all) - y_all.sum()),
        },
        "model": {k: info.get(k) for k in ("architecture", "model_version", "round_id", "sha256", "backend")},
        "mapping": {"features": mapping.features, "label": mapping.label, "zero_means_missing": mapping.zero_means_missing,
                    "conversions": mapping.conversions, "reasons": mapping.reasons},
        "overlap": {"training_reference": "PIMA Indians Diabetes (OpenML 37)", "reference_md5": PIMA_MD5,
                    "compared_on": "the model's inputs, 0 = not measured as a gap in both files",
                    "rows_matching_training_records": int(in_training.sum()),
                    "distinct_training_records_matched": distinct_matched,
                    "matching_with_same_outcome": same_label,
                    "copies_of_hospital_training_split_records": trained_on,
                    "copies_of_held_out_records_only": held_out,
                    "repeated_rows_in_file": int(repeated.sum()),
                    "unseen_rows": int(unseen.sum()), "unseen_distinct_rows": int(unique_unseen.sum()),
                    "unseen_distinct_max_values_shared_with_one_pima_record": sorted(closeness)},
        "baselines": {"own_only": {"rounds": rounds, "local_epochs": 2, "seeds": seeds, "hospital_train_records": sizes},
                      "majority_class_predicts": majority},
        "threshold": 0.5,
        "results": results,
        "note": "Research prototype, not a medical device. These numbers are not clinical accuracy.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                         encoding="utf-8")
    png_path = args.out.with_suffix(".png")
    plot(out, png_path)
    print(f"\nWrote {json_path} and {png_path}")


if __name__ == "__main__":
    main()
