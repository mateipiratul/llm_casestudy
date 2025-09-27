"""Generate historian-alignment plots for model outputs.

The script compares each individual model response against historian-provided
grades, then aggregates summary plots/CSVs under `analysis_reports/`.
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .parser_utils import clean_response_parse_visual, parse_yesno_pv
except Exception:  # pragma: no cover - fallback for standalone execution
    from parser_utils import clean_response_parse_visual, parse_yesno_pv

# --- Configuration ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTDIR = BASE_DIR / "analysis_reports"
CSV_OUTDIR = OUTDIR / "csv"
PLOT_OUTDIR = OUTDIR / "historian"
HISTORIAN_DATA_PATH = BASE_DIR / "historian_grades.json"

# Run file collections: all runs (temps 1.0 and 0.6) plus temp=1.0 only.
RUN_SETS = {
    "all": {
        "files": [
            "results/1_test_results.json",
            "results/2_test_results.json",
            "results/3_test_results.json",
            "results/4_test_results.json",
            "results/5_test_results.json",
        ],
        "suffix": "",
        "label": "all temperatures",
    },
    "temp1": {
        "files": [
            "results/1_test_results.json",
            "results/2_test_results.json",
            "results/3_test_results.json",
            "results/4_test_results.json",
        ],
        "suffix": "_temp1",
        "label": "temperature 1 only",
    },
}

# Historian grades are on a 1-10 scale. Questions without a grade are skipped.
def load_historian_grades(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    grades: Dict[str, float] = {}
    for entry in data.get("questions", []):
        qid = entry.get("question_id")
        grade = entry.get("grade")
        if qid and grade is not None:
            grades[qid] = float(grade)
    return grades


HISTORIAN_GRADES: Dict[str, float] = load_historian_grades(HISTORIAN_DATA_PATH)
YES_THRESHOLD = 5.0

LANG_ORDER = ["ro", "en", "hu", "ru"]
LANG_LABELS = {"ro": "RO", "en": "EN", "hu": "HU", "ru": "RU"}

sns.set_theme(style="whitegrid", context="paper")
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

# --- Utilities -------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_scale_records(run_files: Iterable[str]) -> pd.DataFrame:
    """Return per-response scale records with historian alignment metrics."""
    rows: List[Dict[str, object]] = []
    for path in run_files:
        path_obj = BASE_DIR / path
        if not path_obj.exists():
            print(f"Warning: run file not found, skipping: {path}")
            continue
        with open(path_obj, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for record in payload.get("results", []):
            system_prompt = record.get("system_prompt_id", "")
            if not isinstance(system_prompt, str) or "scale" not in system_prompt.lower():
                continue
            qid = record.get("question_id")
            lang = record.get("question_language")
            model = record.get("model")
            grade = HISTORIAN_GRADES.get(qid)
            if grade is None or not (qid and lang and model):
                continue
            try:
                value = float(str(record.get("response", "")).strip())
            except Exception:
                continue
            if not (1.0 <= value <= 10.0):
                continue
            abs_error = abs(value - grade)
            accuracy = max(0.0, 1.0 - abs_error / 9.0)
            rows.append({
                "model": model,
                "question_id": qid,
                "language": lang,
                "run_file": path,
                "historian_grade": grade,
                "response_value": value,
                "abs_error": abs_error,
                "accuracy": accuracy,
                "accuracy_pct": accuracy * 100.0,
            })
    if not rows:
        raise RuntimeError("No historian-aligned scale rows could be built; check source data.")
    df = pd.DataFrame(rows)
    df["language_label"] = df["language"].map(LANG_LABELS).fillna(df["language"])
    return df


def load_yesno_records(run_files: Iterable[str]) -> pd.DataFrame:
    """Return per-response yes/no records with historian alignment metrics."""
    rows: List[Dict[str, object]] = []
    for path in run_files:
        path_obj = BASE_DIR / path
        if not path_obj.exists():
            print(f"Warning: run file not found, skipping: {path}")
            continue
        with open(path_obj, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for record in payload.get("results", []):
            system_prompt = record.get("system_prompt_id", "")
            if not isinstance(system_prompt, str) or "yesno" not in system_prompt.lower():
                continue
            qid = record.get("question_id")
            lang = record.get("question_language")
            model = record.get("model")
            grade = HISTORIAN_GRADES.get(qid)
            if grade is None or not (qid and lang and model):
                continue
            raw = str(record.get("response", ""))
            cleaned = clean_response_parse_visual(raw)
            yn = parse_yesno_pv(cleaned, lang)
            if yn is None:
                continue
            expected_yes = bool(grade > YES_THRESHOLD)
            is_yes = bool(yn)
            correct = (is_yes == expected_yes)
            accuracy = 1.0 if correct else 0.0
            rows.append({
                "model": model,
                "question_id": qid,
                "language": lang,
                "run_file": path,
                "historian_grade": grade,
                "response_value": bool(yn),
                "expected_yes": expected_yes,
                "is_threshold_case": grade == YES_THRESHOLD,
                "is_correct": correct,
                "accuracy": accuracy,
                "accuracy_pct": accuracy * 100.0,
            })
    if not rows:
        raise RuntimeError("No historian-aligned yes/no rows could be built; check source data.")
    df = pd.DataFrame(rows)
    df["language_label"] = df["language"].map(LANG_LABELS).fillna(df["language"])
    return df


def save_bar_plot(series: pd.Series, title: str, xlabel: str, outfile: str, horizontal: bool = True) -> None:
    series = series.dropna()
    if series.empty:
        return
    series = series.sort_values(ascending=not horizontal)
    plt.figure(figsize=(11, max(4.5, 0.4 * len(series))))
    if horizontal:
        ax = sns.barplot(x=series.values, y=series.index, orient="h", palette="viridis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        for i, v in enumerate(series.values):
            ax.text(v + max(1.0, 0.02 * (series.max() or 1)), i, f"{v:.1f}", va="center", fontsize=8)
    else:
        ax = sns.barplot(x=series.index, y=series.values, palette="viridis")
        ax.set_ylabel(xlabel)
        ax.set_xlabel("")
        for i, v in enumerate(series.values):
            ax.text(i, v + max(1.0, 0.02 * (series.max() or 1)), f"{v:.1f}", ha="center", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def save_heatmap(pivot: pd.DataFrame, title: str, outfile: str) -> None:
    if pivot.empty:
        return
    plt.figure(figsize=(12, max(5, 0.45 * pivot.shape[0])))
    ax = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# --- Main entry point ------------------------------------------------------

def generate_outputs(run_files: Iterable[str], suffix: str) -> None:
    df = load_scale_records(run_files)

    detailed_path = CSV_OUTDIR / f"historian_alignment_detailed{suffix}.csv"
    df.to_csv(detailed_path, index=False)

    model_summary = df.groupby("model")["accuracy_pct"].mean().sort_values(ascending=False)
    model_summary.to_csv(CSV_OUTDIR / f"historian_accuracy_by_model{suffix}.csv")

    lang_summary = (
        df.groupby("language_label")["accuracy_pct"].mean()
        .reindex([LANG_LABELS.get(l, l) for l in LANG_ORDER])
        .dropna()
    )
    lang_summary.to_csv(CSV_OUTDIR / f"historian_accuracy_by_language{suffix}.csv")

    question_summary = df.groupby("question_id")["accuracy_pct"].mean().sort_values(ascending=False)
    question_summary.to_csv(CSV_OUTDIR / f"historian_accuracy_by_question{suffix}.csv")

    plot_suffix = suffix if suffix else ""


    save_bar_plot(
        model_summary,
        "Historian Alignment by Model",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_model_accuracy{plot_suffix}.png",
        horizontal=True,
    )

    save_bar_plot(
        lang_summary,
        "Average Accuracy by Language",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_language_accuracy{plot_suffix}.png",
        horizontal=False,
    )

    save_bar_plot(
        question_summary,
        "Historian Alignment by Question",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_question_accuracy{plot_suffix}.png",
        horizontal=True,
    )

    heatmap_data = (
        df.groupby(["model", "language_label"])["accuracy_pct"].mean().unstack("language_label")
    )
    ordered_cols = [lbl for lbl in [LANG_LABELS.get(l, l) for l in LANG_ORDER] if lbl in heatmap_data.columns]
    heatmap_data = heatmap_data[ordered_cols]
    save_heatmap(
        heatmap_data,
        "Historian Alignment Heatmap (Model x Language)",
        PLOT_OUTDIR / f"historian_model_language_heatmap{plot_suffix}.png",
    )

    # Yes/No alignment
    yn_df = load_yesno_records(run_files)

    yn_detailed_path = CSV_OUTDIR / f"historian_yesno_alignment_detailed{suffix}.csv"
    yn_df.to_csv(yn_detailed_path, index=False)

    yn_model_summary = yn_df.groupby("model")["accuracy_pct"].mean().sort_values(ascending=False)
    yn_model_summary.to_csv(CSV_OUTDIR / f"historian_yesno_accuracy_by_model{suffix}.csv")

    yn_lang_summary = (
        yn_df.groupby("language_label")["accuracy_pct"].mean()
        .reindex([LANG_LABELS.get(l, l) for l in LANG_ORDER])
        .dropna()
    )
    yn_lang_summary.to_csv(CSV_OUTDIR / f"historian_yesno_accuracy_by_language{suffix}.csv")

    yn_question_summary = (
        yn_df.groupby("question_id")["accuracy_pct"].mean().sort_values(ascending=False)
    )
    yn_question_summary.to_csv(CSV_OUTDIR / f"historian_yesno_accuracy_by_question{suffix}.csv")

    save_bar_plot(
        yn_model_summary,
        "Historian Yes/No Alignment by Model",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_yesno_model_accuracy{plot_suffix}.png",
        horizontal=True,
    )

    save_bar_plot(
        yn_lang_summary,
        "Yes/No Accuracy by Language",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_yesno_language_accuracy{plot_suffix}.png",
        horizontal=False,
    )

    save_bar_plot(
        yn_question_summary,
        "Yes/No Accuracy by Question",
        "Accuracy (%)",
        PLOT_OUTDIR / f"historian_yesno_question_accuracy{plot_suffix}.png",
        horizontal=True,
    )

    yn_heatmap = (
        yn_df.groupby(["model", "language_label"])["accuracy_pct"].mean().unstack("language_label")
    )
    ordered_cols = [lbl for lbl in [LANG_LABELS.get(l, l) for l in LANG_ORDER] if lbl in yn_heatmap.columns]
    yn_heatmap = yn_heatmap[ordered_cols]
    save_heatmap(
        yn_heatmap,
        "Historian Yes/No Alignment Heatmap",
        PLOT_OUTDIR / f"historian_yesno_model_language_heatmap{plot_suffix}.png",
    )


def main() -> None:
    ensure_dir(OUTDIR)
    ensure_dir(CSV_OUTDIR)
    ensure_dir(PLOT_OUTDIR)

    for key, config in RUN_SETS.items():
        suffix = config["suffix"]
        files = config["files"]
        generate_outputs(files, suffix)
        label = config.get("label", key)
        print(f"Saved historian-alignment outputs ({label}) to {OUTDIR} and {CSV_OUTDIR}")


if __name__ == "__main__":
    main()
