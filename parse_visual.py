import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
import numpy as np

# --- Configuration ---
RESULTS_FILE = 'bulk_test_results.json'
YESNO_PLOT_FILE = 'results_yes_no_heatmap.png'
SCALE_PLOT_FILE = 'results_scale_heatmap.png'

# Define consistent colors for languages
LANG_COLORS = {
    'ro': '#0077b6',  # Blue
    'en': '#009e73',  # Green
    'hu': '#d55e00',  # Orange
    'ru': '#cc79a7'   # Mauve
}

def load_and_prepare_data(file_path):
    print(f"Loading results from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at '{file_path}'.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The results are in the 'results' key
        df = pd.DataFrame(data.get('results', []))
        if df.empty:
            print("Warning: The results file is empty or contains no results.")
            return None
        return df
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing the results file: {e}")
        return None

def create_yesno_heatmap(df):
    """Creates a heatmap for Yes/No/Error responses."""
    if df.empty:
        print("No 'Yes/No' data found to generate a plot.")
        return

    print("Generating 'Yes/No' response heatmap...")

    # 1. Map text responses to numerical values for coloring
    response_map = {'Yes': 1, 'No': 0}
    df['response_code'] = df['response'].map(response_map).fillna(-1) # -1 for errors/other text

    # 2. Create the pivot tables
    # One for the color codes (numeric)
    pivot_color = df.pivot_table(index='model', columns='question_id', values='response_code')
    # One for the text annotations (strings)
    pivot_annot = df.pivot_table(index='model', columns='question_id', values='response', aggfunc=lambda x: ' '.join(x))

    # 3. Set up the plot
    fig, ax = plt.subplots(figsize=(24, 12))
    sns.heatmap(
        pivot_color,
        ax=ax,
        annot=pivot_annot, # Use the text pivot for annotations
        fmt='s',           # 's' format for strings
        cmap=['#f94144', '#f9c74f', '#90be6d'], # Red (Error), Yellow (No), Green (Yes)
        linewidths=1.5,
        linecolor='white',
        cbar=False,        # We'll use a custom legend
        annot_kws={"size": 10, "weight": "bold"}
    )

    # 4. Customize the plot
    ax.set_title('Model Responses: Yes/No Questions', fontsize=20, pad=20)
    ax.set_xlabel('Question ID', fontsize=14, labelpad=15)
    ax.set_ylabel('Model', fontsize=14, labelpad=15)

    # Color the x-axis labels by language
    qid_to_lang = df.set_index('question_id')['question_language'].to_dict()
    for tick_label in ax.get_xticklabels():
        qid = tick_label.get_text()
        language = qid_to_lang.get(qid)
        if language in LANG_COLORS:
            tick_label.set_color(LANG_COLORS[language])

    plt.xticks(rotation=45, ha='right')

    # 5. Create custom legends
    lang_patches = [mpatches.Patch(color=color, label=lang.upper()) for lang, color in LANG_COLORS.items()]
    status_patches = [
        mpatches.Patch(color='#90be6d', label='Yes'),
        mpatches.Patch(color='#f9c74f', label='No'),
        mpatches.Patch(color='#f94144', label='Error / Invalid')
    ]
    
    # Place legends outside the plot
    legend1 = ax.legend(handles=lang_patches, title='Question Language', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.add_artist(legend1)
    ax.legend(handles=status_patches, title='Response', bbox_to_anchor=(1.01, 0.7), loc='upper left')
    
    # 6. Save and show
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(YESNO_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Yes/No heatmap saved to '{YESNO_PLOT_FILE}'")
    plt.show()

def create_scale_heatmap(df):
    """Creates a heatmap for numerical scale responses."""
    if df.empty:
        print("No 'Scale' data found to generate a plot.")
        return

    print("Generating 'Scale' response heatmap...")

    # 1. Convert response to a number, coercing errors to NaN
    df['response_num'] = pd.to_numeric(df['response'], errors='coerce')

    # 2. Create the pivot table
    pivot_data = df.pivot_table(index='model', columns='question_id', values='response_num')

    # 3. Set up the plot
    fig, ax = plt.subplots(figsize=(24, 12))
    sns.heatmap(
        pivot_data,
        ax=ax,
        annot=True,        # Show the numbers in the cells
        fmt=".1f",         # Format numbers as floats with one decimal
        cmap="viridis",    # A nice perceptually uniform colormap
        linewidths=1.5,
        linecolor='white',
        cbar_kws={'label': 'Agreement Score (1=Disagree, 5=Agree)'}
    )

    # 4. Customize the plot
    ax.set_title('Model Responses: Scale Questions (1-5)', fontsize=20, pad=20)
    ax.set_xlabel('Question ID', fontsize=14, labelpad=15)
    ax.set_ylabel('Model', fontsize=14, labelpad=15)

    # Color the x-axis labels by language
    qid_to_lang = df.set_index('question_id')['question_language'].to_dict()
    for tick_label in ax.get_xticklabels():
        qid = tick_label.get_text()
        language = qid_to_lang.get(qid)
        if language in LANG_COLORS:
            tick_label.set_color(LANG_COLORS[language])

    plt.xticks(rotation=45, ha='right')

    # 5. Create language legend
    lang_patches = [mpatches.Patch(color=color, label=lang.upper()) for lang, color in LANG_COLORS.items()]
    ax.legend(handles=lang_patches, title='Question Language', bbox_to_anchor=(1.01, 1), loc='upper left')

    # 6. Save and show
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(SCALE_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Scale heatmap saved to '{SCALE_PLOT_FILE}'")
    plt.show()


def main():
    """Main function to run the analysis and generate plots."""
    # Load the data
    df = load_and_prepare_data(RESULTS_FILE)
    
    if df is None:
        return

    # Filter data into two categories based on 'system_prompt_id'
    df_yesno = df[df['system_prompt_id'].str.startswith('yesno')].copy()
    df_scale = df[df['system_prompt_id'].str.startswith('scale')].copy()

    # Generate the two separate graphics
    create_yesno_heatmap(df_yesno)
    create_scale_heatmap(df_scale)


if __name__ == "__main__":
    main()