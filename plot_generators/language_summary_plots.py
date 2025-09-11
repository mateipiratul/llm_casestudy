import json
import os
import re
from itertools import product
from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_FILES = [
    'results/1_test_results.json',
    'results/2_test_results.json',
    'results/3_test_results.json',
    'results/4_test_results.json',
    'results/5_test_results.json'
]
OUT_DIR = os.path.join("analysis_reports", "simple")

LANG_ORDER = ["ro", "en", "hu", "ru"]
LANG_LABEL = {"ro": "RO", "en": "EN", "hu": "HU", "ru": "RU"}
LANG_COLORS = {"ro": "#0077b6", "en": "#009e73", "hu": "#d55e00", "ru": "#cc79a7"}
A4_LANDSCAPE = (11.7, 8.3)
FIG_DPI = 200
SCALE_STABILITY_TOL = 1.0  

YES_WORDS = {"en": ["yes"], "ro": ["da"], "hu": ["igen"], "ru": ["да"]}
NO_WORDS = {"en": ["no"], "ro": ["nu"], "hu": ["nem"], "ru": ["нет"]}


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clean_response(text):
    if not isinstance(text, str):
        return text
    s = text.lstrip()
    if s.startswith("<think>"):
        end = s.find("</think>")
        if end != -1:
            return s[end + len("</think>") :].strip()
    return text.strip()


def shorten_model_name(full_name: str) -> str:
    return full_name.split("/", 1)[1] if "/" in full_name else full_name


def parse_yesno_value(response: str, lang: str) -> Optional[int]:
    if not isinstance(response, str) or lang not in YES_WORDS:
        return None
    cleaned = re.sub(r"[.!?,]", "", response).lower().strip()
    if cleaned in YES_WORDS[lang]:
        return 1
    if cleaned in NO_WORDS[lang]:
        return 0
    return None


def load_results(paths: List[str]) -> pd.DataFrame:
    """Loads and concatenates data from a list of JSON files."""
    all_dfs = []
    for file_path in paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data.get("results", []))
            if not df.empty:
                all_dfs.append(df)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing '{file_path}': {e}")
    
    if not all_dfs:
        raise ValueError("No valid data loaded from any of the provided files.")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    full_df["response"] = full_df["response"].apply(clean_response)
    full_df["model"] = full_df["model"].apply(shorten_model_name)
    return full_df


# -----------------------------
# 1) 
# -----------------------------
def plot_lang_model_heatmaps(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    yn = df[df["system_prompt_id"].str.startswith("yesno")].copy()
    yn["y"] = yn.apply(lambda r: parse_yesno_value(r.get("response"), r.get("question_language")), axis=1)
    yn = yn.dropna(subset=["y"]).copy()
    if not yn.empty:
        agg = yn.groupby(["question_language", "model"]).agg(p_yes=("y", "mean")).reset_index()
        pivot = agg.pivot_table(index="question_language", columns="model", values="p_yes")
        pivot = pivot.reindex(index=[l for l in LANG_ORDER if l in pivot.index])
        plt.figure(figsize=A4_LANDSCAPE)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, linecolor="#efefef")
        plt.title("Percent YES by language (rows) and model (cols)")
        plt.xlabel("Model")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap_yes_rate_lang_x_model.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close()

    sc = df[df["system_prompt_id"].str.startswith("scale")].copy()
    sc["score"] = pd.to_numeric(sc["response"], errors="coerce")
    sc = sc.dropna(subset=["score"]).copy()
    if not sc.empty:
        agg = sc.groupby(["question_language", "model"]).agg(mean_score=("score", "mean")).reset_index()
        pivot = agg.pivot_table(index="question_language", columns="model", values="mean_score")
        pivot = pivot.reindex(index=[l for l in LANG_ORDER if l in pivot.index])
        plt.figure(figsize=A4_LANDSCAPE)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, linecolor="#efefef")
        plt.title("Mean scale score (1–10) by language (rows) and model (cols)")
        plt.xlabel("Model")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap_scale_mean_lang_x_model.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close()


# -----------------------------
# 2) 
# -----------------------------
def compute_stability_yesno(df: pd.DataFrame) -> pd.DataFrame:
    yn = df[df["system_prompt_id"].str.startswith("yesno")].copy()
    yn["y"] = yn.apply(lambda r: parse_yesno_value(r.get("response"), r.get("question_language")), axis=1)
    yn = yn.dropna(subset=["y"]).copy()
    rows = []
    for m, gm in yn.groupby("model"):
        pivot = gm.pivot_table(index="question_id", columns="question_language", values="y", aggfunc="first")
        valid_mask = pivot.notna().sum(axis=1) >= 2
        pivot_v = pivot[valid_mask]
        total = pivot_v.shape[0]
        if total == 0:
            rows.append({"model": m, "stability": np.nan, "total_questions": 0})
            continue
        equal_rows = pivot_v.apply(lambda row: len(set(row.dropna().astype(int))) == 1, axis=1)
        stable = equal_rows.sum()
        rows.append({"model": m, "stability": stable / total, "total_questions": int(total)})
    return pd.DataFrame(rows)


def compute_stability_scale(df: pd.DataFrame, tol: float = SCALE_STABILITY_TOL) -> pd.DataFrame:
    sc = df[df["system_prompt_id"].str.startswith("scale")].copy()
    sc["score"] = pd.to_numeric(sc["response"], errors="coerce")
    sc = sc.dropna(subset=["score"]).copy()
    rows = []
    for m, gm in sc.groupby("model"):
        pivot = gm.pivot_table(index="question_id", columns="question_language", values="score", aggfunc="mean")
        valid_mask = pivot.notna().sum(axis=1) >= 2
        pivot_v = pivot[valid_mask]
        total = pivot_v.shape[0]
        if total == 0:
            rows.append({"model": m, "stability": np.nan, "total_questions": 0})
            continue
        # stable if max-min <= tol
        diffs = pivot_v.max(axis=1) - pivot_v.min(axis=1)
        stable = (diffs <= tol).sum()
        rows.append({"model": m, "stability": stable / total, "total_questions": int(total)})
    return pd.DataFrame(rows)


def plot_stability_bars(stab_df: pd.DataFrame, out_path: str, title: str):
    if stab_df.empty:
        return
    df = stab_df.sort_values("stability", ascending=False)
    plt.figure(figsize=A4_LANDSCAPE)
    sns.barplot(data=df, x="stability", y="model", color="#577590")
    plt.xlim(0, 1)
    plt.xlabel("Stability across languages (fraction of questions)")
    plt.ylabel("Model")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# -----------------------------
# 3) 
# -----------------------------
def plot_sensitive_questions(df: pd.DataFrame, out_dir: str, top_k: int = 10):
    ensure_dir(out_dir)
    yn = df[df["system_prompt_id"].str.startswith("yesno")].copy()
    yn["y"] = yn.apply(lambda r: parse_yesno_value(r.get("response"), r.get("question_language")), axis=1)
    yn = yn.dropna(subset=["y"]).copy()
    if not yn.empty:
        agg = yn.groupby(["question_id", "question_language"]).agg(p_yes=("y", "mean")).reset_index()
        pivot = agg.pivot_table(index="question_id", columns="question_language", values="p_yes", aggfunc="mean")
        pivot = pivot.reindex(columns=[l for l in LANG_ORDER if l in pivot.columns])
        max_gap = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
        top = max_gap.head(top_k)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top.values, y=top.index, color="#f94144")
        plt.xlabel("Max difference across languages (percent YES)")
        plt.ylabel("Question ID")
        plt.title("Top questions by cross-language difference — Yes/No")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_sensitive_yesno.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close()

    sc = df[df["system_prompt_id"].str.startswith("scale")].copy()
    sc["score"] = pd.to_numeric(sc["response"], errors="coerce")
    sc = sc.dropna(subset=["score"]).copy()
    if not sc.empty:
        agg = sc.groupby(["question_id", "question_language"]).agg(mean_score=("score", "mean")).reset_index()
        pivot = agg.pivot_table(index="question_id", columns="question_language", values="mean_score", aggfunc="mean")
        pivot = pivot.reindex(columns=[l for l in LANG_ORDER if l in pivot.columns])
        max_gap = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
        top = max_gap.head(top_k)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top.values, y=top.index, color="#f9844a")
        plt.xlabel("Max difference across languages (scale points)")
        plt.ylabel("Question ID")
        plt.title("Top questions by cross-language difference — Scale (1–10)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_sensitive_scale.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close()


# -----------------------------
# 4) 
# -----------------------------
def plot_flip_matrix(df: pd.DataFrame, out_path: str):
    yn = df[df["system_prompt_id"].str.startswith("yesno")].copy()
    yn["y"] = yn.apply(lambda r: parse_yesno_value(r.get("response"), r.get("question_language")), axis=1)  
    yn = yn.dropna(subset=["y"]).copy()
    if yn.empty:
        return
    
    flips = pd.DataFrame(index=LANG_ORDER, columns=LANG_ORDER, dtype=float)
    counts = pd.DataFrame(index=LANG_ORDER, columns=LANG_ORDER, dtype=float)
    flips[:] = np.nan
    counts[:] = 0
    for (m, q), g in yn.groupby(["model", "question_id"]):
        pivot = g.pivot_table(index="question_id", columns="question_language", values="y", aggfunc="first")
        row = pivot.iloc[0]
        for la, lb in product(LANG_ORDER, LANG_ORDER):
            if la == lb:
                continue
            va = row.get(la)
            vb = row.get(lb)
            if pd.notna(va) and pd.notna(vb):
                counts.loc[la, lb] += 1
                if int(va) == 1 and int(vb) == 0:
                    flips.loc[la, lb] = (0 if pd.isna(flips.loc[la, lb]) else flips.loc[la, lb]) + 1
    frac = flips.copy()
    for la, lb in product(LANG_ORDER, LANG_ORDER):
        if la == lb:
            frac.loc[la, lb] = np.nan
        else:
            num = 0 if pd.isna(flips.loc[la, lb]) else flips.loc[la, lb]
            den = counts.loc[la, lb]
            frac.loc[la, lb] = (num / den) if den > 0 else np.nan

    plt.figure(figsize=(8, 7))
    sns.heatmap(frac, annot=True, fmt=".2f", cmap="Reds", linewidths=0.6, linecolor="#eeeeee")
    plt.title("Flip rate: P(Yes in A and No in B) across models/questions")
    plt.xlabel("Language B")
    plt.ylabel("Language A")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# -----------------------------
# 5) 
# -----------------------------
def plot_yesno_vs_scale_scatter(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    yn = df[df["system_prompt_id"].str.startswith("yesno")].copy()
    yn["y"] = yn.apply(lambda r: parse_yesno_value(r.get("response"), r.get("question_language")), axis=1)
    yn = yn.dropna(subset=["y"]).copy()
    sc = df[df["system_prompt_id"].str.startswith("scale")].copy()
    sc["score"] = pd.to_numeric(sc["response"], errors="coerce")
    sc = sc.dropna(subset=["score"]).copy()
    ynj = yn[["model", "question_id", "question_language", "y"]].rename(columns={"question_language": "language"})
    scj = sc[["model", "question_id", "question_language", "score"]].rename(columns={"question_language": "language"})
    merged = pd.merge(ynj, scj, on=["model", "question_id", "language"], how="inner")
    if merged.empty:
        return
    merged["accord"] = ((merged["y"].astype(int) == 1) & (merged["score"] >= 6)) | ((merged["y"].astype(int) == 0) & (merged["score"] <= 5))
    accord = merged.groupby(["language", "model"]).agg(accord_rate=("accord", "mean"), n=("accord", "size")).reset_index()
    pivot = accord.pivot_table(index="language", columns="model", values="accord_rate")
    pivot = pivot.reindex(index=[l for l in LANG_ORDER if l in pivot.index])
    plt.figure(figsize=A4_LANDSCAPE)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="GnBu", linewidths=0.5, linecolor="#efefef", vmin=0, vmax=1)
    plt.title("Accord Rate: (YES=1 & Scale≥6) or (YES=0 & Scale≤5)")
    plt.xlabel("Model")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_yesno_scale_accord_rate.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    accord.to_csv(os.path.join(out_dir, "yesno_scale_accord_rate.csv"), index=False)

    merged["YES"] = merged["y"].astype(int)
    try:
        g = sns.catplot(
            data=merged,
            x="YES",
            y="score",
            col="language",
            kind="violin",
            col_wrap=4,
            sharey=True,
            palette={0: "#f94144", 1: "#43aa8b"},
        )
        g.set_axis_labels("YES (0/1)", "Scale score (1–10)")
        g.fig.suptitle("Scale distributions by YES (faceted by language)", y=1.02)
        for ax in g.axes.flatten():
            ax.set_ylim(1, 10)
        g.savefig(os.path.join(out_dir, "violin_yesno_vs_scale_by_language.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close(g.fig)
    except Exception:
        plt.figure(figsize=(9, 6))
        sns.violinplot(data=merged, x="YES", y="score", inner="quartile")
        plt.ylim(1, 10)
        plt.xlabel("YES (0/1)")
        plt.ylabel("Scale score (1–10)")
        plt.title("Scale distributions by YES")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "violin_yesno_vs_scale_by_language.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close()

    try:
        g2 = sns.catplot(
            data=merged,
            x="YES",
            y="score",
            col="model",
            kind="violin",
            col_wrap=3,
            sharey=True,
            palette={0: "#f94144", 1: "#43aa8b"},
        )
        g2.set_axis_labels("YES (0/1)", "Scale score (1–10)")
        g2.fig.suptitle("Scale distributions by YES (faceted by model)", y=1.02)
        for ax in g2.axes.flatten():
            ax.set_ylim(1, 10)
        g2.savefig(os.path.join(out_dir, "violin_yesno_vs_scale_by_model.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close(g2.fig)
    except Exception:
        pass

    def band(score: float) -> str:
        if score <= 3:
            return "Low (1–3)"
        if score <= 7:
            return "Mid (4–7)"
        return "High (8–10)"

    merged["band"] = merged["score"].apply(band)
    dist = (
        merged.groupby(["YES", "band"]).size().reset_index(name="n")
    )
    totals = dist.groupby("YES")["n"].transform("sum")
    dist["prop"] = dist["n"] / totals
    order_bands = ["Low (1–3)", "Mid (4–7)", "High (8–10)"]
    pivot_bar = dist.pivot_table(index="YES", columns="band", values="prop").reindex(columns=order_bands).fillna(0)

    plt.figure(figsize=(8, 5))
    bottoms = np.zeros(pivot_bar.shape[0])
    xlabels = ["NO", "YES"] if set(pivot_bar.index) == {0, 1} else [str(i) for i in pivot_bar.index]
    x = np.arange(pivot_bar.shape[0])
    colors = ["#f94144", "#f9c74f", "#43aa8b"]
    for i, col in enumerate(pivot_bar.columns):
        vals = pivot_bar[col].values
        plt.bar(x, vals, bottom=bottoms, color=colors[i], edgecolor="white", label=col)
        bottoms += vals
    plt.xticks(x, xlabels)
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.title("Scale distribution by YES (100% stacked: Low/Mid/High)")
    plt.legend(title="Band", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stacked100_yesno_scale_low_mid_high.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    merged["score_int"] = merged["score"].round().clip(1, 10).astype(int)
    freq = merged.groupby(["YES", "score_int"]).size().reset_index(name="n")
    row_tot = freq.groupby("YES")["n"].transform("sum")
    freq["prop"] = freq["n"] / row_tot
    pivot_hm = freq.pivot_table(index="YES", columns="score_int", values="prop").reindex(columns=list(range(1, 11))).fillna(0)
    plt.figure(figsize=(10, 3.5))
    sns.heatmap(pivot_hm, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    plt.xlabel("Scale score")
    plt.ylabel("YES (0/1)")
    plt.title("Scale score frequency by YES (per-row normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_yesno_scale_score_frequency.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(results_files: List[str] = RESULTS_FILES):
    ensure_dir(OUT_DIR)
    print(f"Loading {len(results_files)} result file(s) ...")
    df = load_results(results_files)

    # 1) Heatmaps
    plot_lang_model_heatmaps(df, OUT_DIR)

    # 2) Stability bars
    stab_yn = compute_stability_yesno(df)
    if not stab_yn.empty:
        stab_yn.to_csv(os.path.join(OUT_DIR, "stability_yesno.csv"), index=False)
        plot_stability_bars(stab_yn, os.path.join(OUT_DIR, "bar_stability_yesno.png"), "Stability across languages — Yes/No")
    stab_sc = compute_stability_scale(df, tol=SCALE_STABILITY_TOL)
    if not stab_sc.empty:
        stab_sc.to_csv(os.path.join(OUT_DIR, "stability_scale.csv"), index=False)
        plot_stability_bars(stab_sc, os.path.join(OUT_DIR, "bar_stability_scale.png"), f"Stability across languages — Scale (tol={SCALE_STABILITY_TOL:g})")

    # 3) Sensitive questions
    plot_sensitive_questions(df, os.path.join(OUT_DIR, "sensitive"), top_k=10)

    # 4) Flips matrix
    plot_flip_matrix(df, os.path.join(OUT_DIR, "heatmap_flips_yesno.png"))

    # 5) Scatter Yes/No vs Scale
    plot_yesno_vs_scale_scatter(df, OUT_DIR)

    print(f"Done. Plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()