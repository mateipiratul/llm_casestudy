import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os

# --- Configuration ---
RESULTS_FILE = 'bulk_test_results.json'
OUTPUT_IMAGE_FILE = 'llm_test_results_visualization.png'

# Define colors for languages and statuses for consistency
LANG_COLORS = {
    'ro': '#0077b6',  # Blue for Romanian
    'en': '#009e73',  # Green for English
    'hu': '#d55e00',  # Orange for Hungarian
    'ru': '#cc79a7'   # Mauve for Russian
}
STATUS_COLORS = {
    'success': '#90be6d', # Light Green
    'error': '#f94144'    # Light Red
}

def create_results_visualization(file_path):
    """
    Parses the LLM test results JSON file and creates a 2D visualization.
    """
    # 1. Parse the results file
    print(f"Loading results from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at '{file_path}'.")
        print("Please run the main testing script first to generate the results.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get('results', [])
        if not results:
            print("Error: The results file does not contain any test data.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{file_path}'. The file may be corrupted.")
        return

    # Convert the results into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # We only need a few columns for this visualization
    df = df[['model', 'question_id', 'question_language', 'status']]

    # Map the status ('success'/'error') to a numerical value for the heatmap
    # 1 for success, 0 for error
    df['status_code'] = df['status'].apply(lambda x: 1 if x == 'success' else 0)

    # Pivot the DataFrame to create the 2D grid structure:
    # - Rows (index) = models
    # - Columns = question_ids
    # - Values = status_code (1 or 0)
    pivot_df = df.pivot_table(
        index='model', 
        columns='question_id', 
        values='status_code'
    )

    # 2. Create the visualization
    print("Generating plot...")
    
    # Set up the plot size
    # We make it wider to accommodate question IDs
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create the heatmap using seaborn
    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap=[STATUS_COLORS['error'], STATUS_COLORS['success']], # Custom colormap for error/success
        linewidths=1.5,
        linecolor='white',
        cbar=False, # We will create a custom legend instead of a color bar
        annot=False # Annotations (1/0) can make it look cluttered
    )

    # --- Customize the plot for clarity ---

    # Set titles and labels
    ax.set_title('LLM Performance on Historical Controversies', fontsize=20, pad=20)
    ax.set_xlabel('Question ID', fontsize=14, labelpad=15)
    ax.set_ylabel('Model', fontsize=14, labelpad=15)

    # Color the x-axis labels (question_ids) based on language
    # First, create a mapping from question_id to its language
    qid_to_lang = df.set_index('question_id')['question_language'].to_dict()
    
    for tick_label in ax.get_xticklabels():
        qid = tick_label.get_text()
        language = qid_to_lang.get(qid)
        if language in LANG_COLORS:
            tick_label.set_color(LANG_COLORS[language])
            tick_label.set_weight('bold')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # --- Create custom legends ---

    # Legend for languages (using the colored labels)
    lang_patches = [mpatches.Patch(color=color, label=lang.upper()) for lang, color in LANG_COLORS.items()]
    lang_legend = plt.legend(
        handles=lang_patches, 
        title='Question Language', 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        fontsize='large',
        title_fontsize='large'
    )
    ax.add_artist(lang_legend) # Add the first legend manually

    # Legend for status (the cells)
    status_patches = [
        mpatches.Patch(color=STATUS_COLORS['success'], label='Success'),
        mpatches.Patch(color=STATUS_COLORS['error'], label='Error')
    ]
    plt.legend(
        handles=status_patches, 
        title='Test Status', 
        bbox_to_anchor=(1.02, 0.5), 
        loc='center left',
        fontsize='large',
        title_fontsize='large'
    )

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make space on the right for legends

    # Save the plot to a file
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight')
    print(f"Visualization saved successfully to '{OUTPUT_IMAGE_FILE}'")
    
    # Display the plot
    plt.show()


if __name__ == "__main__":
    create_results_visualization(RESULTS_FILE)