import json
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import numpy as np
import re
try:
    from .parser_utils import clean_response_parse_visual, parse_yesno_pv
except Exception:
    from parser_utils import clean_response_parse_visual, parse_yesno_pv

# Configuration matching study_plots.py standards
FIG_DPI = 200
A4_LANDSCAPE_FIGSIZE = (16, 10)

sns.set_theme(style='whitegrid', context='paper')
mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 8,
})

def ensure_dir(p: str):
    """Create directory if it doesn't exist (matching study_plots.py pattern)."""
    os.makedirs(p, exist_ok=True)

RESULTS_FILES = [
    'results/1_test_results.json',
    'results/2_test_results.json',
    'results/3_test_results.json',
    'results/4_test_results.json'
]
YESNO_PLOT_FILE = 'analysis_reports/consistency_yesno_quadrant_map.png'
SCALE_PLOT_FILE = 'analysis_reports/consistency_scale_diagonal_map.png'
A4_LANDSCAPE_FIGSIZE = (16, 10)

def parse_yesno_response(row):
    response = row['response']
    lang = row['question_language']
    parsed = parse_yesno_pv(response, lang)
    if parsed is True:
        return 1
    return 0

def clean_response(response_text):
    return clean_response_parse_visual(response_text)

def shorten_model_name(full_name):
    if '/' in full_name:
        return full_name.split('/', 1)[1]
    return full_name

def load_and_prepare_data(file_paths):
    print(f"Loading results from {len(file_paths)} files...")
    all_dfs = []
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: Results file not found at '{file_path}'. Skipping.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data.get('results', []))
            df['run_id'] = i + 1
            all_dfs.append(df)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing '{file_path}': {e}")
    if not all_dfs:
        print("Error: No valid data could be loaded. Exiting.")
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    print("Applying data cleaning and transformations...")
    full_df['response'] = full_df['response'].apply(clean_response)
    full_df['model'] = full_df['model'].apply(shorten_model_name)
    return full_df

def create_yesno_heatmap(df):
    if df.empty:
        print("No 'Yes/No' data found to generate a plot.")
        return
    print("Generating 'Yes/No' consistency heatmap...")
    # Report invalid parses (None)
    parsed_vals = df.apply(lambda r: parse_yesno_pv(r['response'], r['question_language']), axis=1)
    invalid_n = parsed_vals.isna().sum()
    if invalid_n:
        print(f"Warning: {int(invalid_n)} YES/NO responses could not be parsed exactly (see cleaning rules). They count as 'No' in the visualization.")
    df['yes_count'] = parsed_vals.fillna(False).astype(int)
    pivot_df = df.pivot_table(
        index='model', columns=['question_id', 'question_language'],
        values='yes_count', aggfunc='sum'
    )
    
    col_scores = pivot_df.mean(axis=0).groupby('question_id').mean()
    sorted_qids = col_scores.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df.reindex(sorted_qids, axis=1, level='question_id')
    row_scores = pivot_df.mean(axis=1)
    sorted_models = row_scores.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df.reindex(sorted_models, axis=0)
    
    models = pivot_df.index.tolist()
    unique_qids = pivot_df.columns.get_level_values('question_id').unique().tolist()
    
    quadrant_layout = {
        'ro': {'pos': (0, 0), 'label': 'RO'}, 'en': {'pos': (0.5, 0), 'label': 'EN'},
        'hu': {'pos': (0, 0.5), 'label': 'HU'}, 'ru': {'pos': (0.5, 0.5), 'label': 'RU'}
    }
    
    cmap = plt.get_cmap('YlGnBu')
    norm = mcolors.Normalize(vmin=0, vmax=4)

    fig, ax = plt.subplots(figsize=A4_LANDSCAPE_FIGSIZE)

    for y, model in enumerate(models):
        for x, qid in enumerate(unique_qids):
            for lang, layout in quadrant_layout.items():
                try:
                    count = pivot_df.loc[model, (qid, lang)]
                    color = cmap(norm(count))
                except KeyError:
                    count = np.nan
                    color = '#e0e0e0'

                rect_x, rect_y = x + layout['pos'][0], y + layout['pos'][1]
                rect = mpatches.Rectangle((rect_x, rect_y), 0.5, 0.5, facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(rect)
                
                if not np.isnan(count):
                    ax.text(rect_x + 0.25, rect_y + 0.25, layout['label'],
                            ha='center', va='center', fontsize=6, color='black', weight='bold', alpha=0.6)

    for i in range(len(unique_qids) + 1): ax.axvline(i, color='#555555', linewidth=2)
    for i in range(len(models) + 1): ax.axhline(i, color='#555555', linewidth=2)
    ax.set_ylim(len(models), 0); ax.set_xlim(0, len(unique_qids))
    ax.set_yticks([i + 0.5 for i in range(len(models))]); ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks([i + 0.5 for i in range(len(unique_qids))]); ax.set_xticklabels(unique_qids, rotation=60, fontsize=8)
    ax.set_title('Consistency of "Yes" Responses Across 4 Runs\nCell split into RO|EN|HU|RU; color = count of "Yes" across runs (0–4). Unparsable counted as 0.', fontsize=14, pad=12)
    cbar_ax = fig.add_axes([0.9, 0.6, 0.02, 0.3])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_label('Number of "Yes" Responses', rotation=270, labelpad=10)
    
    fig.subplots_adjust(left=0.18, right=0.89, top=0.92, bottom=0.15)
    ensure_dir(os.path.dirname(YESNO_PLOT_FILE))
    plt.savefig(YESNO_PLOT_FILE, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Yes/No consistency heatmap saved to '{YESNO_PLOT_FILE}'")
    plt.close()  # Add plt.close() for memory management

def create_scale_heatmap(df):
    if df.empty:
        print("No 'Scale' data found to generate a plot.")
        return
        
    print("Generating RO/HU score divergence heatmap...")
    rn = pd.to_numeric(df['response'], errors='coerce')
    # Enforce valid range [1,10]
    rn = rn.where((rn >= 1) & (rn <= 10))
    invalid_sc = rn.isna().sum()
    if invalid_sc:
        print(f"Warning: {int(invalid_sc)} SCALE responses invalid or out of [1,10] after cleaning; excluded from medians.")
    df['response_num'] = rn
    df_ro_hu = df[df['question_language'].isin(['ro', 'hu'])].copy()
    
    pivot_data = df_ro_hu.pivot_table(
        index='model', 
        columns=['question_id', 'question_language'],
        values='response_num', 
        aggfunc="median"
    )
    
    try:
        ro_df = pivot_data.xs('ro', level='question_language', axis=1)
    except KeyError:
        print("Warning: No Romanian ('ro') data found in the scale results.")
        ro_df = pd.DataFrame(index=pivot_data.index)

    try:
        hu_df = pivot_data.xs('hu', level='question_language', axis=1)
    except KeyError:
        print("Warning: No Hungarian ('hu') data found in the scale results.")
        hu_df = pd.DataFrame(index=pivot_data.index)

    difference_df = ro_df.subtract(hu_df, fill_value=0)

    annot_df = pd.DataFrame(index=pivot_data.index, columns=difference_df.columns)
    for model in annot_df.index:
        for qid in annot_df.columns:
            ro_val = ro_df.get(qid, {}).get(model, 'N/A')
            hu_val = hu_df.get(qid, {}).get(model, 'N/A')
            ro_str = f"{ro_val:.1f}" if isinstance(ro_val, (int, float)) else "N/A"
            hu_str = f"{hu_val:.1f}" if isinstance(hu_val, (int, float)) else "N/A"
            if ro_str == "10.0":
                ro_str = "10"
            if hu_str == "10.0":
                hu_str = "10"
            annot_df.loc[model, qid] = f"{ro_str}|{hu_str}"

    col_scores = difference_df.abs().mean(axis=0)
    sorted_qids = col_scores.sort_values(ascending=False).index.tolist()
    difference_df = difference_df[sorted_qids]
    annot_df = annot_df[sorted_qids]
    
    row_scores = pivot_data.mean(axis=1)
    sorted_models = row_scores.sort_values(ascending=False).index.tolist()
    difference_df = difference_df.reindex(sorted_models)
    annot_df = annot_df.reindex(sorted_models)

    fig, ax = plt.subplots(figsize=A4_LANDSCAPE_FIGSIZE)
    
    sns.heatmap(
        difference_df, 
        ax=ax,
        annot=annot_df,
        fmt='s',
        cmap="RdBu_r",
        center=0,
        linewidths=1.5,
        linecolor='white',
        annot_kws={"size": 8, "weight": "bold"},
        cbar_kws={'label': 'Score Difference (RO median - HU median)', 'shrink': 0.75}
    )

    ax.set_title('Divergence of Median Scores (RO vs HU) Across 4 Runs\nCell text = RO|HU medians; color = RO − HU (only valid 1–10 included)', fontsize=16, pad=20)
    ax.set_xlabel('Question ID', fontsize=12, labelpad=10)
    ax.set_ylabel('Model', fontsize=12, labelpad=10)
    plt.xticks(rotation=60, ha='center', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    annot_patch = mpatches.Patch(color='grey', label='RO score | HU score')
    ax.legend(handles=[annot_patch], title='Cell Annotation Format', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    fig.subplots_adjust(left=0.17, right=0.83, top=0.9, bottom=0.25)
    ensure_dir(os.path.dirname(SCALE_PLOT_FILE))
    plt.savefig(SCALE_PLOT_FILE, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Scale divergence heatmap saved to '{SCALE_PLOT_FILE}'")
    plt.close()  # Add plt.close() for memory management

    
def main():
    df = load_and_prepare_data(RESULTS_FILES)
    if df is None:
        return
    df_yesno = df[df['system_prompt_id'].astype(str).str.startswith('yesno')].copy()
    df_scale = df[df['system_prompt_id'].astype(str).str.startswith('scale')].copy()
    create_yesno_heatmap(df_yesno)
    create_scale_heatmap(df_scale)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
