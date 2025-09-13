import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from datetime import datetime
import numpy as np 

FIG_DPI = 200
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
    os.makedirs(p, exist_ok=True)

model_data = [
    {
        "name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "short_name": "Qwen2.5-7B",
        "developer": "Alibaba",
        "release_date": "2025-07-21",
        "total_params_b": 7,
        "active_params_b": 7,
        "architecture": "Dense",
    },
    {
        "name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "short_name": "Mistral-Small-24B",
        "developer": "Mistral AI",
        "release_date": "2025-01-30",
        "total_params_b": 24,
        "active_params_b": 6, # estimated active params
        "architecture": "MoE",
    },
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "short_name": "Llama-3.3-70B",
        "developer": "Meta",
        "release_date": "2024-12-06",
        "total_params_b": 70,
        "active_params_b": 70,
        "architecture": "Dense",
    },
    {
        "name": "openai/gpt-oss-20b",
        "short_name": "GPT-OSS-20B",
        "developer": "OpenAI",
        "release_date": "2025-08-07",
        "total_params_b": 20,
        "active_params_b": 20,
        "architecture": "Dense",
    },
    {
        "name": "google/gemma-3n-E4B-it",
        "short_name": "Gemma-3n-4B",
        "developer": "Google",
        "release_date": "2025-07-10",
        "total_params_b": 4,
        "active_params_b": 4,
        "architecture": "Dense",
    },
    {
        "name": "deepseek-ai/DeepSeek-V3.1",
        "short_name": "DeepSeek-V3.1",
        "developer": "DeepSeek AI",
        "release_date": "2025-08-21",
        "total_params_b": 685,
        "active_params_b": 37,
        "architecture": "MoE",
    },
    {
        "name": "deepseek-ai/DeepSeek-R1",
        "short_name": "DeepSeek-R1",
        "developer": "DeepSeek AI",
        "release_date": "2025-05-29",
        "total_params_b": 671,
        "active_params_b": 37,
        "architecture": "MoE",
    },
    {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "short_name": "Qwen3-235B",
        "developer": "Alibaba",
        "release_date": "2025-07-21",
        "total_params_b": 235,
        "active_params_b": 22,
        "architecture": "MoE",
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "short_name": "Llama-4-Scout-17B",
        "developer": "Meta",
        "release_date": "2025-04-05",
        "total_params_b": 109, # estimated total params
        "active_params_b": 17,
        "architecture": "MoE",
    },
    {
        "name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "short_name": "Llama-4-Maverick-17B",
        "developer": "Meta",
        "release_date": "2025-04-05",
        "total_params_b": 400, # estimated total params
        "active_params_b": 17,
        "architecture": "MoE",
    },
    {
        "name": "lgai/exaone-3-5-32b-instruct",
        "short_name": "Exaone-3.5-32B",
        "developer": "LG AI",
        "release_date": "2024-12-11", # approx public date
        "total_params_b": 32,
        "active_params_b": 32,
        "architecture": "Dense",
    },
    {
        "name": "moonshotai/Kimi-K2-Instruct",
        "short_name": "Kimi-K2",
        "developer": "Moonshot AI",
        "release_date": "2025-09-04",
        "total_params_b": 1000, # estimate
        "active_params_b": 32,
        "architecture": "Dense",
    },
    {
        "name": "arcee-ai/virtuoso-large",
        "short_name": "Virtuoso-Large",
        "developer": "Arcee AI",
        "release_date": "2025-05-05", # approx public date
        "total_params_b": 70, # based on Llama-3 70B
        "active_params_b": 70,
        "architecture": "Dense",
    },
]

def create_model_timeline_plot(data, output_path="../analysis_reports/model_timeline_plot.png"):
    df = pd.DataFrame(data)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df.sort_values('release_date')

    developers = df['developer'].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(developers)))
    color_map = dict(zip(developers, colors))
    df['color'] = df['developer'].map(color_map)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.scatter(
        df['release_date'],
        df['total_params_b'],
        s=df['active_params_b'] * 15,
        c=df['color'],
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )

    for i, row in df.iterrows():
        ax.text(
            row['release_date'],
            row['total_params_b'],
            f"  {row['short_name']}",
            fontsize=8,
            verticalalignment='center'
        )

    ax.set_yscale('log')
    top_limit = df['total_params_b'].max() * 1.2
    ax.set_ylim(bottom=3, top=top_limit)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{int(y)}B'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_title("Evolution of Selected LLMs by Size and Release Date", fontsize=16, pad=20)
    ax.set_xlabel("Release Date", fontsize=12)
    ax.set_ylabel("Total Parameters (Billions, Log Scale)", fontsize=12)

    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    developer_patches = [plt.Line2D([0], [0], marker='o', color='w', label=dev,
                                    markerfacecolor=color, markersize=10)
                         for dev, color in color_map.items()]
    legend1 = ax.legend(handles=developer_patches, title="Developer", 
                        bbox_to_anchor=(0.02, 0.98), loc='upper left', 
                        borderaxespad=0.)
    ax.add_artist(legend1)

    size_legend_points = [10, 50, 100]
    size_handles = [plt.scatter([], [], s=p * 15, c='gray', alpha=0.7, edgecolors='w',
                                label=f'{p}B') for p in size_legend_points]
    legend2 = ax.legend(handles=size_handles, title="Active Parameters",
                        bbox_to_anchor=(0.02, 0.3), loc='upper left', 
                        borderaxespad=0., labelspacing=2)

    fig.tight_layout()
    
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Plot saved successfully to '{output_path}'")
    plt.close()  # Add plt.close() for memory management


if __name__ == "__main__":
    print("--- LLM Timeline Plot Generator ---")
    print("NOTE: Release dates and parameter counts are based on publicly available,")
    print("often estimated data and are subject to change. This plot is for illustrative purposes.\n")
    
    create_model_timeline_plot(model_data)