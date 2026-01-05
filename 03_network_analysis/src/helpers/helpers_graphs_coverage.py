import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .graph_style_guide import (
    apply_thesis_style,
    COLOR_IXELLE,
    COLOR_LLN,
    COLOR_WAHA,
    setup_grid,
    save_figure,
    FONT_SIZE_LABEL,
    FONT_SIZE_ANNOTATION,
)

PREFIX_TO_ENVIRONMENT = {
    "ixelle": "Ixelles (Urban)",
    "lln": "LLN (Suburban)",
    "waha": "Waha (Rural)",
}


def extract_summary_rows(df_coverage: pd.DataFrame) -> pd.DataFrame:
    
    df_summary = df_coverage[
        (df_coverage['row_type'] == 'SUMMARY') &
        (df_coverage['session'] != 'GLOBAL_SUMMARY')
    ].copy()
    
    return df_summary


def generate_coverage_comparison_graph(
    df_coverage: pd.DataFrame,
    output_dir: Path,
    output_filename: str = "coverage_comparison.pdf",
) -> Path:

    
    # Extract summary rows
    df_summary = extract_summary_rows(df_coverage)
    
    if df_summary.empty:
        raise ValueError("No SUMMARY rows found in coverage data")
    
    if len(df_summary) < 3:
        raise ValueError(f"Expected at least 3 environment summaries, found {len(df_summary)}")
    
    # Sort by prefix for consistent ordering
    prefix_order = ['ixelle', 'lln', 'waha']
    df_summary['prefix'] = pd.Categorical(
        df_summary['prefix'],
        categories=prefix_order,
        ordered=True
    )
    df_summary = df_summary.sort_values('prefix')
    
    # Extract data
    environments = []
    serving_coverage = []
    neighbor_coverage = []
    overall_coverage = []
    
    for _, row in df_summary.iterrows():
        prefix = row['prefix']
        environments.append(PREFIX_TO_ENVIRONMENT.get(prefix, prefix.capitalize()))
        
        serving_coverage.append(float(row['avg_serving_cov_pct']))
        neighbor_coverage.append(float(row['avg_neighbor_cov_pct']))
        overall_coverage.append(float(row['avg_overall_cov_pct']))
    
    # Create figure with unified style
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar chart setup
    x = np.arange(len(environments))
    width = 0.25
    
    color_serving = '#0173B2'    # Blue
    color_neighbor = '#DE8F05'   # Orange
    color_overall = '#029E73'    # Green
    
    # Create bars
    bars1 = ax.bar(
        x - width,
        serving_coverage,
        width,
        label='Serving Coverage',
        color=color_serving,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    bars2 = ax.bar(
        x,
        neighbor_coverage,
        width,
        label='Neighbor Coverage',
        color=color_neighbor,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    bars3 = ax.bar(
        x + width,
        overall_coverage,
        width,
        label='Overall Coverage',
        color=color_overall,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    # Labels and title
    ax.set_ylabel('Coverage Percentage (%)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title(
        'Tower Database Coverage by Environment',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    
    # Set ticks
    ax.set_xticks(x)
    ax.set_xticklabels(environments, fontsize=10)
    ax.set_ylim(0, 115)
    
    # Grid styling
    setup_grid(ax, axis='y', alpha=0.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=FONT_SIZE_ANNOTATION,
                fontweight='normal',
            )
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Legend - matches the bar colors
    ax.legend(
        loc='upper right',
        fontsize=10,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
    )
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        save_figure(fig, str(output_path), format='pdf', dpi=300)
        return output_path
    except IOError as e:
        raise IOError(f"Failed to save PDF to {output_path}: {e}")