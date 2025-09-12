import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re
try:
    from .parser_utils import (
        clean_response_parse_visual,
        parse_yesno_pv,
        parse_scale_pv,
        infer_prompt_type,
    )
except Exception:
    from parser_utils import (
        clean_response_parse_visual,
        parse_yesno_pv,
        parse_scale_pv,
        infer_prompt_type,
    )


# --- Config ---
OUTDIR = 'analysis_reports'
CSV_OUTDIR = os.path.join(OUTDIR, 'csv')
RUN_FILES_1_4 = [
    'results/1_test_results.json',
    'results/2_test_results.json',
    'results/3_test_results.json',
    'results/4_test_results.json',
]
RUN_FILE_5 = 'results/5_test_results.json'

FIG_DPI = 200
FIGSIZE_WIDE = (12, 7)
FIGSIZE_TALL = (9, 12)

sns.set_theme(style='whitegrid', context='paper')
mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 8,
})

LANG_ORDER = ['ro', 'en', 'hu', 'ru']
LANG_NAMES = {'ro': 'RO', 'en': 'EN', 'hu': 'HU', 'ru': 'RU'}
LANG_COLORS = {
    'ro': '#d62728',  # red
    'en': '#1f77b4',  # blue
    'hu': '#2ca02c',  # green
    'ru': '#9467bd',  # purple
}


# --- Helpers: parsing ---
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


PARSE_FAIL_ROWS: List[Dict] = []


def _clean_text(resp: str) -> str:
    # Keep for compatibility; use parser_utils cleaner
    cleaned = clean_response_parse_visual(resp)
    return cleaned or ''


def _normalize_ru_homoglyphs(text: str) -> str:
    return text


def parse_yesno(resp: str, lang: str) -> Optional[bool]:
    return parse_yesno_pv(resp, lang)


def parse_scale(resp: str) -> Optional[float]:
    return parse_scale_pv(resp)


def load_runs(paths: List[str]) -> pd.DataFrame:
    rows = []
    for run_idx, path in enumerate(paths, start=1):
        if not os.path.exists(path):
            print(f"Warning: file not found, skipping: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for r in data.get('results', []):
            model = r.get('model')
            qid = r.get('question_id')
            lang = r.get('question_language')
            sys_id = r.get('system_prompt_id')
            ptype = infer_prompt_type(sys_id)
            raw = (r.get('response') or '').strip()
            resp = _clean_text(raw)
            yn = parse_yesno(resp, lang) if ptype == 'yesno' else None
            sc = parse_scale(resp) if ptype == 'scale' else None
            if ptype == 'yesno' and yn is None:
                PARSE_FAIL_ROWS.append({'run': run_idx, 'ptype': ptype, 'model': model, 'qid': qid, 'lang': lang, 'system_prompt_id': sys_id, 'raw': raw, 'cleaned': resp})
            if ptype == 'scale' and sc is None:
                PARSE_FAIL_ROWS.append({'run': run_idx, 'ptype': ptype, 'model': model, 'qid': qid, 'lang': lang, 'system_prompt_id': sys_id, 'raw': raw, 'cleaned': resp})
            rows.append({
                'model': model,
                'qid': qid,
                'lang': lang,
                'prompt_type': ptype,
                'yesno': yn,
                'scale': sc,
                'run': run_idx,
            })
    return pd.DataFrame(rows)


# --- Metric utilities ---
def mode_binary(values: List[Optional[bool]]) -> Optional[bool]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    y = sum(1 for v in vals if v)
    n = len(vals) - y
    if y > n:
        return True
    if n > y:
        return False
    return None


def compute_model_consistency(df14: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, g in df14.groupby('model'):
        # YES/NO
        yn_items = g[g['prompt_type']=='yesno'].groupby(['qid','lang'])
        total_items = 0
        perfect = 0
        flip_rates = []
        for (_, _), gg in yn_items:
            runs = gg.set_index('run').sort_index()['yesno']
            if set([1,2,3,4]).issubset(set(runs.index)):
                vals = runs.loc[[1,2,3,4]].tolist()
                if None in vals:
                    continue
                total_items += 1
                if len(set(vals)) == 1:
                    perfect += 1
                # flip rate vs first run
                base = vals[0]
                diffs = [int(v != base) for v in vals[1:]]
                flip_rates.append(sum(diffs) / 3.0)
        yn_perfect = (perfect / total_items) if total_items else np.nan
        yn_flip_avg = (np.mean(flip_rates) if flip_rates else np.nan)

        # SCALE
        sc_items = g[g['prompt_type']=='scale'].groupby(['qid','lang'])
        sc_total = 0
        sc_high = 0
        sc_stds = []
        for (_, _), gg in sc_items:
            runs = gg.set_index('run').sort_index()['scale']
            if set([1,2,3,4]).issubset(set(runs.index)):
                vals = runs.loc[[1,2,3,4]].tolist()
                if any(v is None for v in vals):
                    continue
                sc_total += 1
                sd = float(np.std(vals, ddof=0))
                sc_stds.append(sd)
                if sd <= 1.0:
                    sc_high += 1
        sc_high_rate = (sc_high / sc_total) if sc_total else np.nan
        sc_std_avg = (np.mean(sc_stds) if sc_stds else np.nan)

        out.append({
            'model': model,
            'yn_perfect_consistency_rate': yn_perfect,
            'yn_avg_flip_rate': yn_flip_avg,
            'sc_high_consistency_rate': sc_high_rate,
            'sc_avg_std': sc_std_avg,
        })
    return pd.DataFrame(out)


def plot_barh(series: pd.Series, title: str, xlabel: str, outfile: str, xlim: Optional[Tuple[float,float]] = None, sort_desc: bool = True):
    s = series.dropna().sort_values(ascending=not sort_desc)
    plt.figure(figsize=(12, max(5, 0.45*len(s))))
    ax = sns.barplot(x=s.values, y=s.index, orient='h', color='#4c78a8')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    if xlim is not None:
        ax.set_xlim(*xlim)
    for i, v in enumerate(s.values):
        ax.text(v + (0.005 if (xlim and xlim[1] <= 1.0) else 0.01), i, f"{v:.2f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile), dpi=FIG_DPI)
    plt.close()


def compute_yesno_perfect_components(df14: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute overall perfect rate per model and language-wise contributions.
    Returns:
      - overall_df: model, total_items, total_perfect, overall_rate
      - lang_df: model, lang, items, perfect, rate
    """
    rows_lang = []
    rows_overall = []
    for model, g in df14[df14['prompt_type']=='yesno'].groupby('model'):
        total_items = 0
        total_perfect = 0
        # per-language
        for lang, gl in g.groupby('lang'):
            perfect = 0
            items = 0
            for qid, gg in gl.groupby('qid'):
                runs = gg.set_index('run').sort_index()['yesno']
                if set([1,2,3,4]).issubset(set(runs.index)):
                    vals = runs.loc[[1,2,3,4]].tolist()
                    if None in vals:
                        continue
                    items += 1
                    if len(set(vals)) == 1:
                        perfect += 1
            rate = (perfect / items) if items else np.nan
            rows_lang.append({'model': model, 'lang': lang, 'items': items, 'perfect': perfect, 'rate': rate})
            total_items += items
            total_perfect += perfect
        overall_rate = (total_perfect / total_items) if total_items else np.nan
        rows_overall.append({'model': model, 'total_items': total_items, 'total_perfect': total_perfect, 'overall_rate': overall_rate})
    return pd.DataFrame(rows_overall), pd.DataFrame(rows_lang)


def plot_A1_stacked_by_language(overall_df: pd.DataFrame, lang_df: pd.DataFrame, outfile: str):
    """A1 stacked per language: vertical bars; height = overall perfect rate; segments = language share."""
    overall_df = overall_df.set_index('model')
    lang_pivot = lang_df.pivot_table(index='model', columns='lang', values='perfect', aggfunc='sum').fillna(0.0)
    for l in LANG_ORDER:
        if l not in lang_pivot.columns:
            lang_pivot[l] = 0.0
    lang_pivot = lang_pivot[LANG_ORDER]
    shares = lang_pivot.div(lang_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    seg = shares.mul(overall_df['overall_rate'], axis=0).fillna(0.0)
    seg = seg.loc[overall_df['overall_rate'].sort_values(ascending=False).index]

    plt.figure(figsize=(max(14, 0.8*len(seg)), 8))
    ax = plt.gca()
    x = np.arange(len(seg))
    bottom = np.zeros(len(seg))
    for lang in LANG_ORDER:
        vals = seg[lang].values
        ax.bar(x, vals, bottom=bottom, color=LANG_COLORS.get(lang, '#999'), edgecolor='white', label=LANG_NAMES.get(lang, lang).upper(), width=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(seg.index, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Yes/No Perfect Consistency (0–1)')
    ax.set_title('Yes/No Perfect Consistency by Model (segmented by language)\nStability = identical Yes/No across runs 1–4')
    overall_sorted = overall_df.loc[seg.index, 'overall_rate']
    for i, v in enumerate(overall_sorted.values):
        ax.text(i, min(1.0, v + 0.015), f"{v:.2f}", va='bottom', ha='center', fontsize=8)
    ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_A1_grouped_by_language(lang_df: pd.DataFrame, outfile: str):
    """A1 grouped per language: 4 separate bars per model, one for each language."""
    # Calculate per-language rates for each model
    lang_rates = lang_df.copy()
    lang_rates['rate'] = lang_rates['perfect'] / lang_rates['items']
    lang_rates = lang_rates.fillna(0.0)
    
    # Pivot to get model x language matrix
    pivot = lang_rates.pivot_table(index='model', columns='lang', values='rate', aggfunc='first').fillna(0.0)
    
    # Ensure all languages are present
    for l in LANG_ORDER:
        if l not in pivot.columns:
            pivot[l] = 0.0
    pivot = pivot[LANG_ORDER]
    
    # Sort models by overall performance (average across languages)
    model_avg = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[model_avg.index]
    
    # Create grouped bar plot
    n_models = len(pivot)
    n_langs = len(LANG_ORDER)
    bar_width = 0.22
    x = np.arange(n_models)
    
    plt.figure(figsize=(max(14, 1.0*n_models*n_langs), 20))
    ax = plt.gca()
    
    # Plot bars for each language
    for i, lang in enumerate(LANG_ORDER):
        offset = (i - 1.5) * bar_width
        vals = pivot[lang].values
        bars = ax.bar(x + offset, vals, bar_width, 
                     color=LANG_COLORS.get(lang, '#999'), 
                     label=LANG_NAMES.get(lang, lang).upper(),
                     edgecolor='white', linewidth=1.0)
        
        # Add value labels on bars
        for j, v in enumerate(vals):
            if v > 0.02:  # Only show labels for bars with meaningful height
                ax.text(x[j] + offset, v + 0.02, f"{v:.2f}", 
                       ha='center', va='bottom', fontsize=18, rotation=0, weight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Yes/No Perfect Consistency (0–1)', fontsize=24)
    ax.set_title('A1b: Yes/No Perfect Consistency by Model and Language (grouped bars)\nStability = identical Yes/No across runs 1–4', fontsize=26, pad=30)
    ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=20, title_fontsize=22)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def compute_temperature_effects(df14: pd.DataFrame, df5: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build baseline per (model,qid,lang) from runs 1–4
    base_yes = df14[df14['prompt_type']=='yesno'].groupby(['model','qid','lang'])['yesno'].apply(mode_binary).reset_index(name='base_yes')
    base_scale = df14[df14['prompt_type']=='scale'].dropna(subset=['scale']).groupby(['model','qid','lang'])['scale'].mean().reset_index(name='base_scale')

    # Run 5 values
    r5_yes = df5[df5['prompt_type']=='yesno'].dropna(subset=['yesno']).groupby(['model','qid','lang'])['yesno'].first().reset_index(name='r5_yes')
    r5_scale = df5[df5['prompt_type']=='scale'].dropna(subset=['scale']).groupby(['model','qid','lang'])['scale'].first().reset_index(name='r5_scale')

    # Merge
    yes_merge = pd.merge(base_yes, r5_yes, on=['model','qid','lang'], how='inner')
    yes_merge['agree'] = (yes_merge['base_yes'] == yes_merge['r5_yes']).astype(float)
    sc_merge = pd.merge(base_scale, r5_scale, on=['model','qid','lang'], how='inner')
    sc_merge['mad'] = (sc_merge['base_scale'] - sc_merge['r5_scale']).abs()

    # Aggregations
    yes_by_model = yes_merge.dropna(subset=['base_yes']).groupby('model')['agree'].mean().to_frame('yes_agree_rate')
    yes_by_lang = yes_merge.dropna(subset=['base_yes']).groupby('lang')['agree'].mean().to_frame('yes_agree_rate')
    sc_by_model = sc_merge.groupby('model')['mad'].mean().to_frame('scale_mad')
    sc_by_lang = sc_merge.groupby('lang')['mad'].mean().to_frame('scale_mad')
    return yes_by_model, yes_by_lang, sc_by_model, sc_by_lang


def compute_language_consistency(df14: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang, g in df14.groupby('lang'):
        # YES/NO
        yn_items = g[g['prompt_type']=='yesno'].groupby(['model','qid'])
        total_items = 0
        perfect = 0
        flip_rates = []
        for (_, _), gg in yn_items:
            runs = gg.set_index('run').sort_index()['yesno']
            if set([1,2,3,4]).issubset(set(runs.index)):
                vals = runs.loc[[1,2,3,4]].tolist()
                if None in vals:
                    continue
                total_items += 1
                if len(set(vals)) == 1:
                    perfect += 1
                base = vals[0]
                diffs = [int(v != base) for v in vals[1:]]
                flip_rates.append(sum(diffs)/3.0)
        yn_perfect = (perfect / total_items) if total_items else np.nan
        yn_flip_avg = (np.mean(flip_rates) if flip_rates else np.nan)

        # SCALE (more permissive: use items with at least 2 parsed runs)
        sc_items = g[g['prompt_type']=='scale'].groupby(['model','qid'])
        sc_total = 0
        sc_high = 0
        sc_stds = []
        for (_, _), gg in sc_items:
            runs = gg.set_index('run').sort_index()['scale']
            # Take available parsed runs among 1..4
            vals = [runs.get(r) for r in [1,2,3,4] if runs.get(r) is not None]
            if len(vals) < 2:
                continue
            sc_total += 1
            sd = float(np.std(vals, ddof=0))
            sc_stds.append(sd)
            if sd <= 1.0:
                sc_high += 1
        sc_high_rate = (sc_high / sc_total) if sc_total else np.nan
        sc_std_avg = (np.mean(sc_stds) if sc_stds else np.nan)

        rows.append({
            'lang': lang,
            'yn_perfect_consistency_rate': yn_perfect,
            'yn_avg_flip_rate': yn_flip_avg,
            'sc_high_consistency_rate': sc_high_rate,
            'sc_avg_std': sc_std_avg,
        })
    return pd.DataFrame(rows)


def compute_scale_std_items(df14: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item std across runs 1–4 for scale, by (model,qid,lang)."""
    rows = []
    g = df14[df14['prompt_type']=='scale']
    for (model, qid, lang), gg in g.groupby(['model','qid','lang']):
        runs = gg.set_index('run').sort_index()['scale']
        if set([1,2,3,4]).issubset(set(runs.index)):
            vals = runs.loc[[1,2,3,4]].tolist()
            if any(v is None for v in vals):
                continue
            sd = float(np.std(vals, ddof=0))
            rows.append({'model': model, 'qid': qid, 'lang': lang, 'std': sd})
    return pd.DataFrame(rows)


def pairwise_yes_agreement(df14: pd.DataFrame) -> pd.DataFrame:
    # Use modal yes/no per (model,qid,lang) across runs 1-4
    base = df14[df14['prompt_type']=='yesno'].groupby(['model','qid','lang'])['yesno'].apply(mode_binary).reset_index(name='mode')
    # Pivot to compare across models on same items
    items = base.dropna(subset=['mode'])
    models = sorted(items['model'].unique().tolist())
    # Create dict: model -> series indexed by (qid,lang)
    series_by_model = {}
    for m, g in items.groupby('model'):
        s = g.set_index(['qid','lang'])['mode']
        series_by_model[m] = s
    # Compute agreement matrix
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for i, mi in enumerate(models):
        si = series_by_model[mi]
        for j, mj in enumerate(models):
            sj = series_by_model[mj]
            joined = pd.merge(si.rename('a'), sj.rename('b'), left_index=True, right_index=True, how='inner')
            if joined.empty:
                val = np.nan
            else:
                val = (joined['a'] == joined['b']).mean()
            mat.loc[mi, mj] = val
    return mat


def pairwise_scale_corr(df14: pd.DataFrame) -> pd.DataFrame:
    # Use mean scale per (model,qid,lang) across runs 1-4
    base = df14[df14['prompt_type']=='scale'].dropna(subset=['scale']).groupby(['model','qid','lang'])['scale'].mean().reset_index(name='mean')
    items = base
    models = sorted(items['model'].unique().tolist())
    # Build wide matrix indexed by (qid,lang), columns=models, values=mean
    wide = items.pivot_table(index=['qid','lang'], columns='model', values='mean')
    corr = wide.corr(method='pearson', min_periods=3)
    return corr


def heatmap_with_values(df: pd.DataFrame, title: str, outfile: str, cmap='viridis', vmin=None, vmax=None, fmt='.2f', annot=False, top_xlabels=False, x_fontsize=6, y_fontsize=6):
    plt.figure(figsize=(max(12, 0.7*len(df.columns)+6), max(10, 0.7*len(df.index)+5)))
    ax = sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'shrink': 0.85})
    plt.title(title)
    if top_xlabels:
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', rotation=45, labelsize=x_fontsize)
    else:
        ax.tick_params(axis='x', rotation=45, labelsize=x_fontsize)
    ax.tick_params(axis='y', labelsize=y_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def language_agreement_per_question(df14: pd.DataFrame) -> pd.DataFrame:
    # Build per (model,qid,lang) modal yes/no
    base = df14[df14['prompt_type']=='yesno'].groupby(['model','qid','lang'])['yesno'].apply(mode_binary).reset_index(name='mode')
    # Majority answer per question across all languages and models
    maj = base.dropna(subset=['mode']).groupby(['qid'])['mode'].apply(mode_binary).reset_index(name='q_majority')
    merged = pd.merge(base, maj, on='qid', how='left')
    merged['agree_majority'] = (merged['mode'] == merged['q_majority']).astype(float)
    # For each question × language: fraction of models aligning with question majority
    agg = merged.dropna(subset=['q_majority']).groupby(['qid','lang'])['agree_majority'].mean().reset_index()
    # Pivot q × lang
    heat = agg.pivot(index='qid', columns='lang', values='agree_majority')
    return heat


def question_yesno_stability(df14: pd.DataFrame) -> pd.DataFrame:
    # Stability score across runs for each (question, model) averaged over languages
    rows = []
    g = df14[df14['prompt_type']=='yesno']
    for (qid, model), gg in g.groupby(['qid','model']):
        langs = []
        for lang, gl in gg.groupby('lang'):
            runs = gl.set_index('run').sort_index()['yesno']
            if set([1,2,3,4]).issubset(set(runs.index)):
                vals = runs.loc[[1,2,3,4]].tolist()
                if None in vals:
                    continue
                langs.append(1.0 if len(set(vals)) == 1 else 0.0)
        if langs:
            rows.append({'qid': qid, 'model': model, 'stability': float(np.mean(langs))})
    if not rows:
        return pd.DataFrame()
    mat = pd.DataFrame(rows).pivot(index='qid', columns='model', values='stability')
    return mat


def question_scale_stddevs(df14: pd.DataFrame) -> pd.DataFrame:
    # For each question × language × model: std across runs 1–4
    rows = []
    g = df14[df14['prompt_type']=='scale']
    for (qid, lang, model), gg in g.groupby(['qid','lang','model']):
        runs = gg.set_index('run').sort_index()['scale']
        if set([1,2,3,4]).issubset(set(runs.index)):
            vals = runs.loc[[1,2,3,4]].tolist()
            if any(v is None for v in vals):
                continue
            sd = float(np.std(vals, ddof=0))
            rows.append({'qid': qid, 'lang': lang, 'model': model, 'std': sd})
    return pd.DataFrame(rows)


def plot_histograms_per_model(sc_merge: pd.DataFrame, outfile_prefix: str):
    # sc_merge contains columns: model, qid, lang, base_scale, r5_scale, mad
    models = sorted(sc_merge['model'].unique().tolist())
    n = len(models)
    cols = 3
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*4, rows*3.2))
    for i, m in enumerate(models, start=1):
        ax = plt.subplot(rows, cols, i)
        sub = sc_merge[sc_merge['model']==m]
        ax.hist(sub['mad'].dropna(), bins=10, color='#4c78a8', alpha=0.85)
        ax.set_title(m)
        ax.set_xlabel('|Δ scale|')
        ax.set_ylabel('count')
    plt.suptitle('Histogram of |Temp0.6 − mean(Temp1.0)| per model (Scale)')
    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(os.path.join(OUTDIR, f'{outfile_prefix}_per_model.png'), dpi=FIG_DPI)
    plt.close()


def plot_scatter_per_model(sc_merge: pd.DataFrame, outfile_prefix: str):
    models = sorted(sc_merge['model'].unique().tolist())
    n = len(models)
    cols = 3
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*4.5, rows*4))
    for i, m in enumerate(models, start=1):
        ax = plt.subplot(rows, cols, i)
        sub = sc_merge[sc_merge['model']==m]
        x = sub['base_scale']
        y = sub['r5_scale']
        ax.scatter(x, y, s=12, alpha=0.7, color='#2ca02c')
        # trendline
        if len(sub) >= 2:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                xs = np.linspace(1, 10, 100)
                ax.plot(xs, p(xs), color='black', linewidth=1)
            except Exception:
                pass
        ax.plot([1,10],[1,10], color='gray', linestyle='--', linewidth=1)
        ax.set_title(m)
        ax.set_xlabel('mean(Temp1.0)')
        ax.set_ylabel('Temp0.6')
        ax.set_xlim(1,10)
        ax.set_ylim(1,10)
    plt.suptitle('Temp0.6 vs mean(Temp1.0) per model (Scale)')
    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(os.path.join(OUTDIR, f'{outfile_prefix}_per_model.png'), dpi=FIG_DPI)
    plt.close()


def composite_scores(model_cons: pd.DataFrame, yes_agree_by_model: pd.DataFrame, sc_mad_by_model: pd.DataFrame) -> pd.DataFrame:
    df = model_cons.set_index('model').copy()
    df = df.join(yes_agree_by_model, how='left')  # adds yes_agree_rate
    df = df.join(sc_mad_by_model, how='left')      # adds scale_mad
    # Normalize metrics to 0..1 (higher better). For flip rate and scale_mad, invert.
    def norm01(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        if s.max() == s.min():
            return pd.Series(1.0, index=s.index)  # avoid division by zero; treat as perfect
        return (s - s.min()) / (s.max() - s.min())

    df['m_yn_perfect'] = df['yn_perfect_consistency_rate']
    df['m_flip_inv'] = 1.0 - df['yn_avg_flip_rate']
    df['m_sc_high'] = df['sc_high_consistency_rate']
    # scale_mad: lower is better -> invert after normalization
    df['m_sc_mad_inv'] = 1.0 - norm01(df['scale_mad'])

    # Normalize the already-probability metrics too (in case of NaNs or edge cases)
    for c in ['m_yn_perfect','m_flip_inv','m_sc_high','yes_agree_rate']:
        if c in df:
            s = df[c].astype(float)
            s = s.clip(lower=0.0, upper=1.0)
            df[c] = s

    # Composite: average of four axes: yn perfect, inverse flip, sc high, yes temp agreement
    df['stability_index'] = df[['m_yn_perfect','m_flip_inv','m_sc_high','yes_agree_rate']].mean(axis=1, skipna=True)
    return df


def _safe_name(name: str) -> str:
    # Replace path separators and non-filename-safe chars with underscore
    name = name.replace(os.sep, '_').replace('/', '_')
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip('_') or 'model'


def radar_chart(df: pd.DataFrame, model: str, outfile_prefix: str):
    # Axes: Consistency (Yes/No), Flip Rate (inverted), Scale Stability, Temp Agreement
    row = df.loc[model]
    labels = ['YN Consistency','1-Flip Rate','Scale Stability','Temp Agreement']
    values = [
        float(row.get('m_yn_perfect', np.nan)),
        float(row.get('m_flip_inv', np.nan)),
        float(row.get('m_sc_high', np.nan)),
        float(row.get('yes_agree_rate', np.nan)),
    ]
    # Close the polygon
    vals = values + [values[0]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, color='#4c78a8', linewidth=2)
    ax.fill(angles, vals, color='#4c78a8', alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    ax.set_title(f'Model: {model}')
    plt.tight_layout()
    safe_model = _safe_name(model)
    plt.savefig(os.path.join(OUTDIR, f'{outfile_prefix}_{safe_model}.png'), dpi=FIG_DPI)
    plt.close()


def main():
    ensure_dir(OUTDIR)
    ensure_dir(CSV_OUTDIR)

    # Load data
    df14 = load_runs(RUN_FILES_1_4)
    df5 = load_runs([RUN_FILE_5])
    # Print parse warnings summary and save CSV
    if PARSE_FAIL_ROWS:
        warn_df = pd.DataFrame(PARSE_FAIL_ROWS)
        warn_path = os.path.join(CSV_OUTDIR, 'study_plots_parse_failures.csv')
        ensure_dir(CSV_OUTDIR)
        warn_df.to_csv(warn_path, index=False)
        counts = warn_df.groupby(['run','ptype']).size().reset_index(name='n')
        print('Parse warnings (study_plots):')
        for _, row in counts.iterrows():
            print(f"  run {int(row['run'])} {row['ptype']}: {int(row['n'])} failures (details: {warn_path})")
    else:
        print('No parse failures detected in study_plots input parsing.')

    # --- A. Model-Level Consistency (runs 1–4) ---
    model_cons = compute_model_consistency(df14)
    model_cons.to_csv(os.path.join(CSV_OUTDIR, 'model_consistency_metrics.csv'), index=False)

    # A1 enhanced: stacked by language, with overall annotation
    overall_df, lang_df = compute_yesno_perfect_components(df14)
    plot_A1_stacked_by_language(overall_df, lang_df, 'A1_model_yesno_perfect_consistency.png')
    plot_barh(model_cons.set_index('model')['yn_avg_flip_rate'],
              'Yes/No Average Flip Rate vs Run1 (runs 1–4)\nFraction of runs 2–4 differing from run 1',
              'flip rate (0–1, lower is better)', 'A2_model_yesno_avg_flip_rate.png', xlim=(0,1))
    plot_barh(model_cons.set_index('model')['sc_high_consistency_rate'],
              'Scale High Consistency Rate (std ≤ 1, runs 1–4)',
              'rate (0–1)', 'A3_model_scale_high_consistency.png', xlim=(0,1))
    plot_barh(model_cons.set_index('model')['sc_avg_std'],
              'Scale Average Standard Deviation (runs 1–4) — lower is more stable',
              'std dev (1–10 scale)', 'A4_model_scale_avg_std.png', sort_desc=False)

    # Additional diagnostics for scale std equality suspicion
    std_items = compute_scale_std_items(df14)
    if not std_items.empty:
        by_model = std_items.groupby('model')['std']
        summary = pd.DataFrame({
            'count_items': by_model.count(),
            'mean_std': by_model.mean(),
            'median_std': by_model.median(),
            'min_std': by_model.min(),
            'max_std': by_model.max(),
        }).sort_index()
        summary.to_csv(os.path.join(CSV_OUTDIR, 'model_scale_std_summary.csv'))
        # Print notice if any models share identical mean_std (rounded 1e-4)
        rounded = summary['mean_std'].round(4)
        dup_vals = rounded[rounded.duplicated(keep=False)]
        if not dup_vals.empty:
            print('Note: some models share identical mean std (rounded to 4 decimals):')
            for val, grp in rounded.groupby(rounded):
                if len(grp) > 1:
                    models_same = ', '.join(grp.index.tolist())
                    print(f"  {val:.4f} -> {models_same}")

    # --- B. Temperature Effect (runs 1–4 vs run 5) ---
    yes_m, yes_l, sc_m, sc_l = compute_temperature_effects(df14, df5)
    yes_m.to_csv(os.path.join(CSV_OUTDIR, 'temp_yes_agree_by_model.csv'))
    yes_l.to_csv(os.path.join(CSV_OUTDIR, 'temp_yes_agree_by_language.csv'))
    sc_m.to_csv(os.path.join(CSV_OUTDIR, 'temp_scale_mad_by_model.csv'))
    sc_l.to_csv(os.path.join(CSV_OUTDIR, 'temp_scale_mad_by_language.csv'))

    plot_barh(yes_m['yes_agree_rate'], 'Yes/No Agreement Rate by Model (Temp1.0 mean vs Temp0.6)', 'agreement rate (0–1)', 'B1_temp_yes_agree_by_model.png', xlim=(0,1))
    plot_barh(yes_l['yes_agree_rate'], 'Yes/No Agreement Rate by Language (Temp1.0 mean vs Temp0.6)', 'agreement rate (0–1)', 'B2_temp_yes_agree_by_language.png', xlim=(0,1))
    plot_barh(sc_m['scale_mad'], 'Scale Mean Absolute Difference by Model | mean(Temp1.0) − Temp0.6 |', '|Δ| (1–10 scale)', 'B3_temp_scale_mad_by_model.png')
    plot_barh(sc_l['scale_mad'], 'Scale Mean Absolute Difference by Language | mean(Temp1.0) − Temp0.6 |', '|Δ| (1–10 scale)', 'B4_temp_scale_mad_by_language.png')

    # --- C. Language-Level Consistency (runs 1–4) ---
    lang_cons = compute_language_consistency(df14)
    lang_cons.to_csv(os.path.join(CSV_OUTDIR, 'language_consistency_metrics.csv'), index=False)
    plot_barh(lang_cons.set_index('lang')['yn_perfect_consistency_rate'], 'Language: Yes/No Perfect Consistency (runs 1–4)', 'rate (0–1)', 'C1_lang_yesno_perfect_consistency.png', xlim=(0,1))
    plot_barh(lang_cons.set_index('lang')['yn_avg_flip_rate'], 'Language: Yes/No Flip Rate vs Run1 (runs 1–4)', 'flip rate (0–1, lower is better)', 'C2_lang_yesno_flip_rate.png', xlim=(0,1))
    plot_barh(lang_cons.set_index('lang')['sc_high_consistency_rate'], 'Language: Scale High Consistency (std ≤ 1, runs 1–4)', 'rate (0–1)', 'C3_lang_scale_high_consistency.png', xlim=(0,1))
    plot_barh(lang_cons.set_index('lang')['sc_avg_std'], 'Language: Scale Average Standard Deviation (runs 1–4)', 'std dev (1–10 scale)', 'C4_lang_scale_avg_std.png')

    # --- D. Cross-Model & Cross-Language Agreement ---
    yes_agree_mat = pairwise_yes_agreement(df14)
    heatmap_with_values(
        yes_agree_mat,
        'Pairwise Model Agreement (Yes/No, runs 1–4)\nCell = fraction of items with the same modal Yes/No (modal = majority across runs 1–4)',
        'D1_pairwise_yes_agreement.png',
        cmap='YlGnBu', vmin=0, vmax=1, annot=True, top_xlabels=False, x_fontsize=9, y_fontsize=9
    )

    sc_corr_mat = pairwise_scale_corr(df14)
    heatmap_with_values(
        sc_corr_mat,
        'Pairwise Model Correlation (Scale means, runs 1–4)\nPearson on per-item mean(1–10); not on Yes/No. Example 1100 vs 0011 ⇒ r ≈ −1 if used',
        'D2_pairwise_scale_correlation.png',
        cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f', annot=True, top_xlabels=False, x_fontsize=9, y_fontsize=9
    )

    lang_q_heat = language_agreement_per_question(df14)
    heatmap_with_values(
        lang_q_heat,
        'Language Agreement per Question (Yes/No)\nCell = share of models (per language) whose modal Yes/No matches the question-level majority (across langs+models)',
        'D3_lang_agreement_per_question.png',
        cmap='YlGnBu', vmin=0, vmax=1, annot=True, top_xlabels=True, x_fontsize=9, y_fontsize=9
    )

    # --- E. Question-Level Reliability ---
    q_model_stab = question_yesno_stability(df14)
    if not q_model_stab.empty:
        heatmap_with_values(q_model_stab, 'Per-Question Consistency (Yes/No stability across runs 1–4) Q×Model\nCell = avg over languages of [1 if all 4 runs identical else 0]', 'E1_question_model_yesno_stability.png', cmap='YlGnBu', vmin=0, vmax=1, annot=True, top_xlabels=False, x_fontsize=9, y_fontsize=9)
        q_model_stab.to_csv(os.path.join(CSV_OUTDIR, 'question_model_yesno_stability.csv'))
    # E2 removed per request

    # F and G sections removed per request

    print(f"All figures saved to {OUTDIR}")


if __name__ == '__main__':
    main()
