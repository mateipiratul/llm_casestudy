import os
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid', context='paper')
mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 8,
})
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

A4_LANDSCAPE = (11.7, 8.3)
FIG_DPI = 200


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_yesno(resp: str, lang: str):
    return parse_yesno_pv(resp, lang)


def parse_scale(resp: str):
    return parse_scale_pv(resp)


PARSE_FAIL_ROWS = []


def load_runs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(paths, 1):
        if not os.path.exists(path):
            print(f"Warning: file not found, skipping: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        for r in data.get('results', []):
            model = r.get('model')
            qid = r.get('question_id')
            lang = r.get('question_language')
            sys_id = r.get('system_prompt_id')
            ptype = infer_prompt_type(sys_id)
            raw = (r.get('response') or '').strip()
            resp = clean_response_parse_visual(raw)
            yn = parse_yesno(resp, lang) if ptype == 'yesno' else None
            sc = parse_scale(resp) if ptype == 'scale' else None
            if ptype == 'yesno' and yn is None:
                PARSE_FAIL_ROWS.append({'run': idx, 'ptype': ptype, 'model': model, 'qid': qid, 'lang': lang, 'system_prompt_id': sys_id, 'raw': raw, 'cleaned': resp})
            if ptype == 'scale' and sc is None:
                PARSE_FAIL_ROWS.append({'run': idx, 'ptype': ptype, 'model': model, 'qid': qid, 'lang': lang, 'system_prompt_id': sys_id, 'raw': raw, 'cleaned': resp})
            rows.append({'model': model, 'qid': qid, 'lang': lang, 'prompt_type': ptype, 'yesno': yn, 'scale': sc, 'run_id': idx})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([])


def modal_yes(series: pd.Series):
    vals = [v for v in series.tolist() if v is not None]
    if not vals:
        return None
    y = sum(1 for v in vals if v)
    n = len(vals) - y
    if y > n:
        return True
    if n > y:
        return False
    return None


def main():
    ap = argparse.ArgumentParser(description='Generate plots (production-grade) using 4 standard runs.')
    ap.add_argument('--outdir', default='analysis_reports', help='Directory for outputs')
    ap.add_argument('--topk', type=int, default=10, help='Top-K for sensitive question lists')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    csv_dir = os.path.join(args.outdir, 'csv')
    ensure_dir(csv_dir)

    # Only the 4 standard runs (exclude essay results here)
    runs = [
        'results/1_test_results.json',
        'results/2_test_results.json',
        'results/3_test_results.json',
        'results/4_test_results.json',
    ]
    df = load_runs(runs)
    # Print parse failures
    if PARSE_FAIL_ROWS:
        warn_df = pd.DataFrame(PARSE_FAIL_ROWS)
        ensure_dir(csv_dir)
        warn_path = os.path.join(csv_dir, 'language_summary_parse_failures.csv')
        warn_df.to_csv(warn_path, index=False)
        counts = warn_df.groupby(['run','ptype']).size().reset_index(name='n')
        print('Parse warnings (language_summary_plots):')
        for _, row in counts.iterrows():
            print(f"  run {int(row['run'])} {row['ptype']}: {int(row['n'])} failures (details: {warn_path})")
    else:
        print('No parse failures detected in language_summary_plots input parsing.')

    # Aggregations across runs
    mod = df[df['prompt_type']=='yesno'].groupby(['model','qid','lang'])['yesno'].apply(modal_yes).reset_index(name='modal_yes')
    med = df[df['prompt_type']=='scale'].dropna(subset=['scale']).groupby(['model','qid','lang'])['scale'].median().reset_index(name='scale_median')

    # Heatmaps: yes-rate and scale-mean by model × language
    yn_rate = mod.dropna(subset=['modal_yes']).groupby(['model','lang'])['modal_yes'].mean().reset_index(name='yes_rate')
    yn_pivot = yn_rate.pivot(index='model', columns='lang', values='yes_rate')
    # Ensure numeric dtype for heatmap
    if not yn_pivot.empty:
        yn_pivot = yn_pivot.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=A4_LANDSCAPE)
    if not yn_pivot.empty:
        sns.heatmap(yn_pivot, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Yes Rate by Model × Language (modal across runs)\nCell = fraction of questions where modal answer is Yes (modal = majority across runs 1–4)')
    plt.xlabel('Language')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'heatmap_yes_rate_lang_x_model.png'), dpi=FIG_DPI)
    plt.close()
    yn_pivot.to_csv(os.path.join(csv_dir, 'yes_rate_by_model_lang.csv'))

    sc_mean = med.groupby(['model','lang'])['scale_median'].mean().reset_index(name='scale_mean')
    sc_pivot = sc_mean.pivot(index='model', columns='lang', values='scale_mean')
    if not sc_pivot.empty:
        sc_pivot = sc_pivot.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=A4_LANDSCAPE)
    if not sc_pivot.empty:
        sns.heatmap(sc_pivot, annot=True, fmt='.2f', cmap='viridis', vmin=1, vmax=10)
    plt.title('Mean of Median Scale by Model × Language\nCell = mean across questions of per-question median scale (runs 1–4, 1–10)')
    plt.xlabel('Language')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'heatmap_scale_mean_lang_x_model.png'), dpi=FIG_DPI)
    plt.close()
    sc_pivot.to_csv(os.path.join(csv_dir, 'scale_mean_by_model_lang.csv'))

    # Stability across languages (sorted by instability) — with diagnostics
    all_models = sorted(df['model'].dropna().unique().tolist())
    all_langs = sorted(df['lang'].dropna().unique().tolist())

    # Coverage by model × language (rows with parsed values)
    cov_yes = df[df['prompt_type']=='yesno'].dropna(subset=['yesno']).groupby(['model','lang']).size().reset_index(name='yesno_rows')
    cov_sc = df[df['prompt_type']=='scale'].dropna(subset=['scale']).groupby(['model','lang']).size().reset_index(name='scale_rows')
    cov = pd.merge(cov_yes, cov_sc, on=['model','lang'], how='outer').fillna(0)
    cov.to_csv(os.path.join(csv_dir, 'coverage_by_model_language.csv'), index=False)

    # YES/NO pair diagnostics per (model,qid,run)
    yn_rows = []
    for (m,q,r), g in df[df['prompt_type']=='yesno'].groupby(['model','qid','run_id']):
        present = g.dropna(subset=['yesno']).groupby('lang')['yesno'].first()
        langs_present = set(present.index.tolist())
        n_langs = len(langs_present)
        stable = None
        if n_langs >= 2:
            stable = 1.0 if len(set(present.tolist())) == 1 else 0.0
        yn_rows.append({
            'model': m, 'qid': q, 'run_id': r, 'n_langs_present': n_langs,
            'langs_present': ','.join(sorted(langs_present)),
            'langs_missing': ','.join(sorted(set(all_langs)-langs_present)),
            'stable': stable
        })
    yes_df = pd.DataFrame(yn_rows)
    yes_df.to_csv(os.path.join(csv_dir, 'yesno_pairs_by_model.csv'), index=False)
    if not yes_df.dropna(subset=['stable']).empty:
        yes_stab = yes_df.dropna(subset=['stable']).groupby('model')['stable'].mean().reindex(all_models).fillna(0.0).sort_values(ascending=True)
    else:
        yes_stab = pd.Series([0.0]*len(all_models), index=all_models)
    plt.figure(figsize=(10, max(4, 0.3*len(yes_stab))))
    sns.barplot(x=yes_stab.values, y=yes_stab.index, orient='h', color='#1f77b4')
    plt.xlabel('Stability across languages (% of (question,run) pairs)')
    plt.ylabel('Model (sorted: most unstable at top)')
    plt.title('YES/NO stability across languages (share of (question,run) with identical answers)\nHigher = more cross-language agreement per (model,question,run)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'bar_yesno_stability_across_languages.png'), dpi=FIG_DPI)
    plt.close()
    yes_stab.to_frame('yesno_stability').to_csv(os.path.join(csv_dir, 'yesno_stability_across_languages.csv'))

    sc_rows = []
    for (m,q,r), g in df[df['prompt_type']=='scale'].groupby(['model','qid','run_id']):
        present = g.dropna(subset=['scale']).groupby('lang')['scale'].first()
        langs_present = set(present.index.tolist())
        n_langs = len(langs_present)
        stable = None
        rng = None
        if n_langs >= 2:
            vals = present.tolist()
            rng = max(vals) - min(vals)
            stable = 1.0 if rng <= 1.0 else 0.0
        sc_rows.append({
            'model': m, 'qid': q, 'run_id': r, 'n_langs_present': n_langs,
            'langs_present': ','.join(sorted(langs_present)),
            'langs_missing': ','.join(sorted(set(all_langs)-langs_present)),
            'range': rng, 'stable': stable
        })
    sc_df = pd.DataFrame(sc_rows)
    sc_df.to_csv(os.path.join(csv_dir, 'scale_pairs_by_model.csv'), index=False)
    if not sc_df.dropna(subset=['stable']).empty:
        sc_stab = sc_df.dropna(subset=['stable']).groupby('model')['stable'].mean().reindex(all_models).fillna(0.0).sort_values(ascending=True)
    else:
        sc_stab = pd.Series([0.0]*len(all_models), index=all_models)
    plt.figure(figsize=(10, max(4, 0.3*len(sc_stab))))
    sns.barplot(x=sc_stab.values, y=sc_stab.index, orient='h', color='#2ca02c')
    plt.xlabel('Stability across languages (% of (question,run) pairs)')
    plt.ylabel('Model (sorted: most unstable at top)')
    plt.title('SCALE stability across languages (std ≤ 1 across languages per (question,run))\nHigher = more cross-language consistency (lower spread)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'bar_scale_stability_across_languages.png'), dpi=FIG_DPI)
    plt.close()
    sc_stab.to_frame('scale_stability').to_csv(os.path.join(csv_dir, 'scale_stability_across_languages.csv'))

    # Sensitive questions (across models)
    # YES/NO via p*(1-p)
    mod_m = mod.dropna(subset=['modal_yes'])
    frac = mod_m.groupby(['qid','lang'])['modal_yes'].mean().reset_index(name='p_yes')
    frac['var'] = frac['p_yes']*(1-frac['p_yes'])
    yn_sens = frac.groupby('qid')['var'].mean().sort_values(ascending=False)
    top = yn_sens.head(args.topk)
    plt.figure(figsize=(10, 0.5*len(top)))
    sns.barplot(x=top.values, y=top.index, color='#d62728')
    plt.xlabel('Variability across models (avg p*(1-p) across languages)')
    plt.ylabel('Question')
    plt.title('Most sensitive questions (YES/NO) — varied across models\nHigher = more disagreement across models (avg p·(1–p))')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'top_sensitive_yesno_across_models.png'), dpi=FIG_DPI)
    plt.close()
    yn_sens.to_frame('sensitivity').to_csv(os.path.join(csv_dir, 'sensitivity_yesno_across_models.csv'))

    # SCALE via across-model range of per-run medians
    rng = med.groupby(['qid','lang'])['scale_median'].agg(lambda s: (max(s)-min(s)) if len(s) else 0.0)
    sc_sens = rng.groupby('qid').mean().sort_values(ascending=False)
    top = sc_sens.head(args.topk)
    plt.figure(figsize=(10, 0.5*len(top)))
    sns.barplot(x=top.values, y=top.index, color='#1f77b4')
    plt.xlabel('Average across-model range (median scale) across languages')
    plt.ylabel('Question')
    plt.title('Most sensitive questions (SCALE) — varied across models\nHigher = larger across-model range of medians')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'top_sensitive_scale_across_models.png'), dpi=FIG_DPI)
    plt.close()
    sc_sens.to_frame('sensitivity').to_csv(os.path.join(csv_dir, 'sensitivity_scale_across_models.csv'))

    print(f"Done. Plots written to {args.outdir}")


if __name__ == '__main__':
    main()
