import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
import numpy as np
import re

RESULTS_FILE = '2_test_results.json'
YESNO_PLOT_FILE = 'results_yesno_quadrant_map.png'
SCALE_PLOT_FILE = 'results_scale_map.png'
A4_LANDSCAPE_FIGSIZE = (10.5, 7.5)

LANG_COLORS = {
    'ro': '#0077b6', 'en': '#009e73',
    'hu': '#d55e00', 'ru': '#cc79a7'
}
YES_WORDS = {
    'en': ['yes'], 'ro': ['da'],
    'hu': ['igen'], 'ru': ['да']
}
NO_WORDS = {
    'en': ['no'], 'ro': ['nu'],
    'hu': ['nem'], 'ru': ['нет']
}

def parse_yesno_response(row):
    response = row['response']
    lang = row['question_language']
    if not isinstance(response, str) or lang not in YES_WORDS:
        return -1
    cleaned_response = re.sub(r'[.!,?]', '', response).lower().strip()
    if cleaned_response in YES_WORDS[lang]:
        return 1
    elif cleaned_response in NO_WORDS[lang]:
        return 0
    else:
        return -1

def clean_response(response_text):
    if not isinstance(response_text, str):
        return response_text
    stripped_text = response_text.lstrip()
    if stripped_text.startswith('<think>'):
        end_tag_index = stripped_text.find('</think>')
        if end_tag_index != -1:
            cleaned_text = stripped_text[end_tag_index + len('</think>'):]
            return cleaned_text.strip()
    return response_text

# --- MODIFICATION 2: Function to shorten model names ---
def shorten_model_name(full_name):
    """Shortens the model name to only include the part after the first '/'."""
    if '/' in full_name:
        return full_name.split('/', 1)[1]
    return full_name

def load_and_prepare_data(file_path):
    print(f"Loading results from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at '{file_path}'.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data.get('results', []))
        if df.empty:
            print("Warning: The results file is empty or contains no results.")
            return None
        
        # Apply response cleaning and model name shortening
        df['response'] = df['response'].apply(clean_response)
        df['model'] = df['model'].apply(shorten_model_name)
        return df
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing the results file: {e}")
        return None

def create_yesno_heatmap(df):
    if df.empty:
        print("No 'Yes/No' data found to generate a plot.")
        return

    print("Generating 'Yes/No' response heatmap with language quadrants...")
    df['response_code'] = df.apply(parse_yesno_response, axis=1)
    
    # Create the pivot table with a multi-index column
    pivot_df = df.pivot_table(
        index='model',
        columns=['question_id', 'question_language'],
        values='response_code'
    )

    # --- MODIFICATION 1: Sort the data ---
    # To sort, we first calculate the mean score per base question ID (averaging across languages)
    # We only consider valid 'Yes' (1) and 'No' (0) answers for a fair score
    valid_responses = pivot_df[pivot_df >= 0]
    
    # Sort Columns (Questions): left-to-right from highest to lowest mean score
    col_scores = valid_responses.mean(axis=0).groupby('question_id').mean()
    sorted_qids = col_scores.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df.reindex(sorted_qids, axis=1, level='question_id')

    # Sort Rows (Models): top-to-bottom from highest to lowest mean score
    row_scores = valid_responses.mean(axis=1)
    sorted_models = row_scores.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df.reindex(sorted_models, axis=0)
    
    # Get the final sorted lists for plotting
    models = pivot_df.index.tolist()
    unique_qids = pivot_df.columns.get_level_values('question_id').unique().tolist()
    
    quadrant_layout = {
        'ro': {'pos': (0, 0.5), 'label': 'RO'}, 'en': {'pos': (0.5, 0.5), 'label': 'EN'},
        'hu': {'pos': (0, 0), 'label': 'HU'}, 'ru': {'pos': (0.5, 0), 'label': 'RU'}
    }
    status_colors = {1: '#90be6d', 0: '#f94144', -1: '#f9c74f', -2: '#e0e0e0'}

    # --- MODIFICATION 3: Set fixed A4-friendly figure size ---
    fig, ax = plt.subplots(figsize=A4_LANDSCAPE_FIGSIZE)

    for y, model in enumerate(models):
        for x, qid in enumerate(unique_qids):
            for lang, layout in quadrant_layout.items():
                try:
                    code = pivot_df.loc[model, (qid, lang)]
                    color = status_colors.get(code, status_colors[-2])
                except KeyError:
                    color = status_colors[-2]
                rect_x, rect_y = x + layout['pos'][0], y + layout['pos'][1]
                rect = mpatches.Rectangle((rect_x, rect_y), 0.5, 0.5, facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(rect)
                ax.text(rect_x + 0.25, rect_y + 0.25, layout['label'],
                        ha='center', va='center', fontsize=6, color='black', weight='bold', alpha=0.5)

    for i in range(len(unique_qids) + 1): ax.axvline(i, color='#555555', linewidth=2)
    for i in range(len(models) + 1): ax.axhline(i, color='#555555', linewidth=2)

    ax.set_ylim(len(models), 0); ax.set_xlim(0, len(unique_qids))
    ax.set_yticks([i + 0.5 for i in range(len(models))]); ax.set_yticklabels(models, fontsize=9)
    ax.set_xticks([i + 0.5 for i in range(len(unique_qids))]); ax.set_xticklabels(unique_qids, rotation=90, fontsize=9)
    ax.set_title('Model Responses: Yes/No Questions by Language', fontsize=16, pad=15)

    status_patches = [mpatches.Patch(color=c, label=l) for l, c in 
                      [('Yes', status_colors[1]), ('No', status_colors[0]), ('Uncertain', status_colors[-1]), ('Missing', status_colors[-2])]]
    ax.legend(handles=status_patches, title='Response', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(YESNO_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Yes/No quadrant heatmap saved to '{YESNO_PLOT_FILE}'")
    plt.show()

def create_scale_heatmap(df):
    if df.empty:
        print("No 'Scale' data found to generate a plot.")
        return
    print("Generating 'Scale' response heatmap...")
    df['response_num'] = pd.to_numeric(df['response'], errors='coerce')
    
    # This pivot table averages scores across languages for each model/question pair
    pivot_data = df.pivot_table(index='model', columns='question_id', values='response_num')

    # --- MODIFICATION 1: Sort the data ---
    # Sort Columns (Questions): left-to-right from highest to lowest mean score
    col_scores = pivot_data.mean(axis=0)
    sorted_cols = col_scores.sort_values(ascending=False).index.tolist()
    pivot_data = pivot_data[sorted_cols]

    # Sort Rows (Models): top-to-bottom from highest to lowest mean score
    row_scores = pivot_data.mean(axis=1)
    sorted_rows = row_scores.sort_values(ascending=False).index.tolist()
    pivot_data = pivot_data.loc[sorted_rows]

    # --- MODIFICATION 3: Set fixed A4-friendly figure size ---
    # Heads-up: If you have many models/questions, labels may become small.
    fig, ax = plt.subplots(figsize=A4_LANDSCAPE_FIGSIZE)
    
    sns.heatmap(
        pivot_data, ax=ax, annot=True, fmt=".1f", cmap="viridis",
        linewidths=1.5, linecolor='white',
        cbar_kws={'label': 'Agreement Score (1=Disagree, 10=Agree)', 'shrink': 0.6}
    )
    ax.set_title('Model Responses: Scale Questions (Average Score)', fontsize=16, pad=15)
    ax.set_xlabel('Question ID', fontsize=12, labelpad=10)
    ax.set_ylabel('Model', fontsize=12, labelpad=10)
    
    # We still need the original df for coloring the labels
    qid_to_lang = df.set_index('question_id')['question_language'].to_dict()
    for tick_label in ax.get_xticklabels():
        qid = tick_label.get_text()
        language = qid_to_lang.get(qid)
        if language in LANG_COLORS:
            tick_label.set_color(LANG_COLORS[language])
            
    plt.xticks(rotation=90, ha='center', fontsize=9)
    plt.yticks(fontsize=9)
    
    lang_patches = [mpatches.Patch(color=color, label=lang.upper()) for lang, color in LANG_COLORS.items()]
    ax.legend(handles=lang_patches, title='Question Language', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(SCALE_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Scale heatmap saved to '{SCALE_PLOT_FILE}'")
    plt.show()

def main():
    df = load_and_prepare_data(RESULTS_FILE)
    if df is None:
        return
    df_yesno = df[df['system_prompt_id'].str.startswith('yesno')].copy()
    df_scale = df[df['system_prompt_id'].str.startswith('scale')].copy()
    create_yesno_heatmap(df_yesno)
    create_scale_heatmap(df_scale)

if __name__ == "__main__":
    main()