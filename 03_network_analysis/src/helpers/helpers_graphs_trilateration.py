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


def extract_trilateration_accuracy_by_location(df_context: pd.DataFrame) -> tuple:
    
    locations = ['ixelle', 'lln', 'waha']
    path_loss_rmse_values = []
    formula_rmse_values = []
    
    for location in locations:
        location_data = df_context[df_context['location'] == location]
        
        if location_data.empty:
            raise ValueError(f"No data found for location: {location}")
        
        # Separate path_loss (non-formula) and formula contexts
        path_loss_rows = location_data[
            ~location_data['context'].str.contains('formula', case=False, na=False) &
            (location_data['avg_rmse_m'].notna())
        ]
        
        formula_rows = location_data[
            location_data['context'].str.contains('formula', case=False, na=False) &
            (location_data['avg_rmse_m'].notna())
        ]
        
        if not path_loss_rows.empty:
            path_loss_rmse = path_loss_rows['avg_rmse_m'].mean()
        else:
            path_loss_rmse = 0.0
        
        if not formula_rows.empty:
            formula_rmse = formula_rows['avg_rmse_m'].mean()
        else:
            formula_rmse = 0.0
        
        path_loss_rmse_values.append(path_loss_rmse)
        formula_rmse_values.append(formula_rmse)
    
    environment_labels = [PREFIX_TO_ENVIRONMENT[loc] for loc in locations]
    
    return environment_labels, path_loss_rmse_values, formula_rmse_values


def generate_trilateration_accuracy_graph(
    df_context: pd.DataFrame,
    output_dir: Path,
    output_filename: str = "trilateration_accuracy.pdf"
) -> Path:
    
    # Extract data
    locations, path_loss_rmse, formula_rmse = extract_trilateration_accuracy_by_location(df_context)
    
    if len(locations) < 3:
        raise ValueError(f"Expected at least 3 locations, found {len(locations)}")
    
    # Create figure with unified style
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Bar chart setup
    x = np.arange(len(locations))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(
        x - width/2,
        path_loss_rmse,
        width,
        label='Path Loss',
        color=COLOR_PATH_LOSS,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    bars2 = ax.bar(
        x + width/2,
        formula_rmse,
        width,
        label='Formula',
        color=COLOR_FORMULA,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
    )
    
    # Labels and title
    ax.set_ylabel('RMSE (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xlabel('Location', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title(
        'Trilateration Accuracy: Path Loss vs Formula',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    
    # Set ticks
    ax.set_xticks(x)
    ax.set_xticklabels(locations, fontsize=10)
    
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