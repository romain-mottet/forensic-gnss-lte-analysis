import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple
from matplotlib.patches import Patch
from .graph_style_guide import (
    apply_thesis_style,
    COLOR_IXELLE,
    COLOR_LLN,
    COLOR_WAHA,
    add_point_labels,
    setup_grid,
    format_legend,
    save_figure,
    FONT_SIZE_LABEL,
    FONT_SIZE_ANNOTATION,
)


PREFIX_TO_COLOR = {
    "ixelle": COLOR_IXELLE,
    "lln": COLOR_LLN,
    "waha": COLOR_WAHA,
}

PREFIX_TO_ENVIRONMENT = {
    "ixelle": "Ixelles (Urban)",
    "lln": "LLN (Suburban)",
    "waha": "Waha (Rural)",
}


def extract_cells_vs_accuracy_data(
    df_sessions: pd.DataFrame,
    df_trilat: pd.DataFrame
) -> Tuple[list, list, list, list, list]:

    
    # Get unique sessions with cell counts
    df_sessions_unique = df_sessions[['prefix', 'session', 'found_cells']].drop_duplicates()
    
    # Get unique trilateration records per session with CEP
    df_trilat_unique = df_trilat.dropna(subset=['cep_m'])
    df_trilat_unique = df_trilat_unique.groupby(['prefix', 'session']).first().reset_index()
    
    # Merge on prefix + session
    merged = df_sessions_unique.merge(
        df_trilat_unique[['prefix', 'session', 'cep_m']],
        on=['prefix', 'session'],
        how='inner'
    )
    
    if merged.empty:
        raise ValueError(
            "No valid data found after merging sessions with trilateration records"
        )
    
    # Extract data
    cell_counts = merged['found_cells'].tolist()
    cep_values = merged['cep_m'].tolist()
    prefixes = merged['prefix'].tolist()
    sessions = merged['session'].tolist()
    
    # Map prefixes to colors
    colors = [PREFIX_TO_COLOR.get(p, '#999999') for p in prefixes]
    
    return cell_counts, cep_values, prefixes, sessions, colors


def generate_cells_vs_accuracy_scatter(
    df_sessions: pd.DataFrame,
    df_trilat: pd.DataFrame,
    output_dir: Path,
    output_filename: str = "cells_vs_accuracy.pdf"
) -> Path:
    
    # Extract data
    cell_counts, cep_values, prefixes, sessions, colors = extract_cells_vs_accuracy_data(
        df_sessions, df_trilat
    )
    
    if len(cell_counts) < 3:
        raise ValueError(f"Expected at least 3 data points, found {len(cell_counts)}")
    
    # Create figure with unified style
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with uniform point size
    scatter = ax.scatter(
        cell_counts,
        cep_values,
        s=150,
        c=colors,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5,
        zorder=3,
    )
    
    # Add vertical threshold line at x=5
    ax.axvline(
        x=5,
        color='#666666',
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        zorder=2,
        label='Threshold (5 cells)',
    )
    
    # Add session labels on points
    for i, session in enumerate(sessions):
        session_num = session.split('_')[-1]
        ax.annotate(
            session_num,
            xy=(cell_counts[i], cep_values[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=FONT_SIZE_ANNOTATION,
            fontweight='normal',
            alpha=0.8,
            zorder=4,
        )
    
    # Set axis labels and title
    ax.set_xlabel('Number of Cells Found', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel('CEP - Circular Error Probable (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title(
        'Trilateration Accuracy vs Cell Count',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    
    # Set axis limits (0-800m focused scale)
    ax.set_ylim(0, 800)
    ax.set_xlim(-0.5, max(cell_counts) + 1)
    
    # Grid styling
    setup_grid(ax, axis='both', alpha=0.4)
    
    # Create custom legend for locations
    legend_elements = [
        Patch(
            facecolor=COLOR_IXELLE,
            edgecolor='black',
            label='Ixelles (Urban)',
            alpha=0.85,
        ),
        Patch(
            facecolor=COLOR_LLN,
            edgecolor='black',
            label='LLN (Suburban)',
            alpha=0.85,
        ),
        Patch(
            facecolor=COLOR_WAHA,
            edgecolor='black',
            label='Waha (Rural)',
            alpha=0.85,
        ),
        plt.Line2D(
            [0],
            [0],
            color='#666666',
            linestyle='--',
            linewidth=1.5,
            label='Threshold (5 cells)',
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=FONT_SIZE_ANNOTATION,
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