import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .graph_style_guide import (
    apply_thesis_style,
    COLOR_PATH_LOSS,
    COLOR_FORMULA,
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


def extract_summary_by_context(df_distance: pd.DataFrame) -> pd.DataFrame:
    
    df_filtered = df_distance[
        (df_distance['context'] == 'default') &
        (df_distance['prefix'].isin(['ixelle', 'lln', 'waha']))
    ].copy()
    
    return df_filtered


def extract_cell_level_errors(df_merged: pd.DataFrame) -> tuple:
    
    df_path_loss = df_merged[df_merged['distance_method'] == 'path_loss'].copy()
    df_formula = df_merged[df_merged['distance_method'] == 'formula'].copy()
    
    path_loss_errors = df_path_loss['mae_m'].dropna().tolist()
    formula_errors = df_formula['mae_m'].dropna().tolist()
    
    return path_loss_errors, formula_errors


def generate_distance_mae_comparison_graph(
    df_distance: pd.DataFrame,
    output_dir: Path,
    output_filename: str = "distance_mae_comparison.pdf",
) -> Path:
    
    # Extract summary rows
    df_summary = extract_summary_by_context(df_distance)
    
    if df_summary.empty:
        raise ValueError("No default context rows found in distance_summary_by_context.csv")
    
    # Extract data and map to environments
    environments = []
    mae_path_loss = []
    mae_formula = []
    
    for prefix in ['ixelle', 'lln', 'waha']:
        prefix_default = df_summary[df_summary['prefix'] == prefix]
        
        if prefix_default.empty:
            continue
        
        # Get path_loss MAE (default context)
        row_default = prefix_default.iloc[0]
        mae_path_loss.append(float(row_default['avg_mae_m']))
        
        # Get formula MAE
        prefix_formula = df_distance[
            (df_distance['prefix'] == prefix) &
            (df_distance['context'].str.contains('formula', na=False))
        ]
        
        if not prefix_formula.empty:
            mae_formula.append(float(prefix_formula.iloc[0]['avg_mae_m']))
        else:
            mae_formula.append(float(row_default['avg_mae_m']))
        
        env_label = PREFIX_TO_ENVIRONMENT.get(prefix, prefix.capitalize())
        environments.append(env_label)
    
    if len(environments) < 3:
        raise ValueError(f"Expected at least 3 environments, found {len(environments)}")
    
    # Create figure with unified style
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Bar chart setup
    x = np.arange(len(environments))
    bar_width = 0.35
    
    # Create bars
    bars1 = ax.bar(
        x - bar_width/2,
        mae_path_loss,
        bar_width,
        label='Path Loss',
        color=COLOR_PATH_LOSS,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    bars2 = ax.bar(
        x + bar_width/2,
        mae_formula,
        bar_width,
        label='Formula',
        color=COLOR_FORMULA,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    # Labels and title
    ax.set_ylabel('Mean Absolute Error (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xlabel('Location', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title(
        'Distance Estimation Accuracy: Path Loss vs Formula',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    
    # Set ticks
    ax.set_xticks(x)
    ax.set_xticklabels(environments, fontsize=10)
    
    # Grid styling
    setup_grid(ax, axis='y', alpha=0.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}',
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
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='#CCCCCC', fancybox=False)
    
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


def generate_distance_error_distribution_graph(
    df_merged: pd.DataFrame,
    output_dir: Path,
    output_filename: str = "distance_error_distribution.pdf",
) -> Path:
    
    # Extract cell-level errors
    path_loss_errors, formula_errors = extract_cell_level_errors(df_merged)
    
    if not path_loss_errors or not formula_errors:
        raise ValueError(
            "Unable to extract error data. Ensure distance_method column exists."
        )
    
    # Create figure with unified style
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create box plot
    bp = ax.boxplot(
        [path_loss_errors, formula_errors],
        patch_artist=True,
        labels=['Path Loss', 'Formula'],
        widths=0.5,
        showmeans=True,
        meanprops=dict(
            marker='D',
            markerfacecolor='red',
            markeredgecolor='darkred',
            markersize=6,
        ),
    )
    
    # Color the boxes
    colors = [COLOR_PATH_LOSS, COLOR_FORMULA]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)
    
    # Style whiskers and other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element in bp:
            for item in bp[element]:
                if hasattr(item, 'set_color'):
                    item.set_color('black')
                if hasattr(item, 'set_linewidth'):
                    item.set_linewidth(1.0)
    
    # Labels and title
    ax.set_ylabel('Mean Absolute Error (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title(
        'Distance Error Distribution by Method',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    
    # Grid styling
    setup_grid(ax, axis='y', alpha=0.5)
    
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