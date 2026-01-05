import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

# Location-based colors (for scatter plots, cells vs accuracy)
COLOR_IXELLE = '#0173B2'   # Blue - Ixelles (Urban)
COLOR_LLN = '#DE8F05'      # Orange - LLN (Suburban)
COLOR_WAHA = '#029E73'     # Green - Waha (Rural)

# Method-based colors (for bar charts comparing methods)
# Using consistent colors from Okabe-Ito palette
COLOR_PATH_LOSS = '#0173B2'   # Blue - Path Loss (matches Ixelle/location color)
COLOR_FORMULA = '#DE8F05'     # Orange - Formula (matches LLN/location color)

# Neutral colors
COLOR_DARK = '#333333'     # Dark gray for text
COLOR_GRID = '#CCCCCC'     # Light gray for grid
COLOR_BORDER = '#999999'   # Medium gray for borders

# ============================================================================
# TYPOGRAPHY CONSTANTS
# ============================================================================

FONT_FAMILY = 'sans-serif'
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_LEGEND = 10
FONT_SIZE_TICK = 10
FONT_SIZE_ANNOTATION = 9

FONT_WEIGHT_BOLD = 'bold'
FONT_WEIGHT_NORMAL = 'normal'

# ============================================================================
# FIGURE SETTINGS
# ============================================================================

DPI_PRINT = 300
DPI_SCREEN = 100

FIGURE_WIDTH = 11
FIGURE_HEIGHT = 7
FIGURE_RATIO = (FIGURE_WIDTH, FIGURE_HEIGHT)

# ============================================================================
# MATPLOTLIB GLOBAL SETTINGS
# ============================================================================

def apply_thesis_style():
    """
    Apply global matplotlib settings for thesis-consistent styling.
    
    Sets font, colors, sizes, and visual parameters that apply to all figures
    created after this function is called.
    """
    
    # Font settings
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_LABEL
    
    # Figure settings
    plt.rcParams['figure.figsize'] = FIGURE_RATIO
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    
    # Axes settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLOR_DARK
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['axes.labelcolor'] = COLOR_DARK
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['axes.titleweight'] = FONT_WEIGHT_BOLD
    
    # Grid settings
    plt.rcParams['axes.grid'] = False  # We'll enable per-axis
    plt.rcParams['grid.color'] = COLOR_GRID
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    
    # Tick settings
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['xtick.color'] = COLOR_DARK
    plt.rcParams['ytick.color'] = COLOR_DARK
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    
    # Legend settings
    plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['legend.edgecolor'] = COLOR_GRID
    plt.rcParams['legend.fancybox'] = False
    
    # Line and patch settings
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['patch.edgecolor'] = COLOR_DARK


def create_figure(figsize=None, title=None, xlabel=None, ylabel=None):
    
    if figsize is None:
        figsize = FIGURE_RATIO
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight=FONT_WEIGHT_BOLD, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, fontweight=FONT_WEIGHT_BOLD)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight=FONT_WEIGHT_BOLD)
    
    return fig, ax


def setup_grid(ax, axis='y', linestyle='-', alpha=0.5):
    
    ax.grid(True, axis=axis, linestyle=linestyle, alpha=alpha, color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def add_value_labels(bars, decimal_places=1, fontsize=None):

    if fontsize is None:
        fontsize = FONT_SIZE_ANNOTATION
    
    ax = bars[0].axes
    
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.{decimal_places}f}'
        
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight=FONT_WEIGHT_NORMAL,
        )


def add_point_labels(ax, x, y, labels, fontsize=None, offset=(5, 5)):
    
    if fontsize is None:
        fontsize = FONT_SIZE_ANNOTATION
    
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=offset,
            textcoords='offset points',
            fontsize=fontsize,
            fontweight=FONT_WEIGHT_NORMAL,
            alpha=0.8,
        )


def format_legend(ax, location='upper right', ncols=1):
    
    ax.legend(
        loc=location,
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.95,
        edgecolor=COLOR_GRID,
        fancybox=False,
        ncol=ncols,
    )


def save_figure(fig, filepath, format='pdf', dpi=DPI_PRINT, tight_layout=True):
    
    if tight_layout:
        fig.tight_layout()
    
    fig.savefig(
        str(filepath),
        format=format,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='white',
    )
    
    plt.close(fig)


def get_location_color(location):
    
    colors = {
        'ixelle': COLOR_IXELLE,
        'lln': COLOR_LLN,
        'waha': COLOR_WAHA,
    }
    
    return colors.get(location, COLOR_DARK)


def get_method_color(method):

    colors = {
        'path_loss': COLOR_PATH_LOSS,
        'formula': COLOR_FORMULA,
    }
    
    return colors.get(method, COLOR_DARK)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # Colors
    'COLOR_IXELLE',
    'COLOR_LLN',
    'COLOR_WAHA',
    'COLOR_PATH_LOSS',
    'COLOR_FORMULA',
    'COLOR_DARK',
    'COLOR_GRID',
    'COLOR_BORDER',
    # Typography
    'FONT_FAMILY',
    'FONT_SIZE_TITLE',
    'FONT_SIZE_LABEL',
    'FONT_SIZE_LEGEND',
    'FONT_SIZE_TICK',
    'FONT_SIZE_ANNOTATION',
    # Functions
    'apply_thesis_style',
    'create_figure',
    'setup_grid',
    'add_value_labels',
    'add_point_labels',
    'format_legend',
    'save_figure',
    'get_location_color',
    'get_method_color',
]