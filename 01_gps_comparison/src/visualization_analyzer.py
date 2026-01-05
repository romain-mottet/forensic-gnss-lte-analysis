import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from typing import Dict, List
from src.config.parameters import RESULTS_DIR, DATA_DIR
from src.helpers.print_helper import PipelineMessages
from src.helpers.waypoint_helper import haversine_distance

# ============================================================================
# UNIFIED STYLE CONSTANTS (Okabe-Ito Colorblind-Accessible Palette)
# ============================================================================

# Okabe-Ito Color Palette
COLOR_BLUE = '#0173B2'      # Blue - Primary/First device/Method
COLOR_ORANGE = '#DE8F05'    # Orange - Secondary/Second device/Method
COLOR_GREEN = '#029E73'     # Green - Tertiary/Accent
COLOR_DARK = '#333333'      # Dark gray - Text
COLOR_GRID = '#CCCCCC'      # Light gray - Grid

# Typography Settings
FONT_FAMILY = 'sans-serif'
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_LEGEND = 10
FONT_SIZE_TICK = 10
FONT_SIZE_ANNOTATION = 9

# Figure Settings
FIGURE_WIDTH = 11
FIGURE_HEIGHT = 7
DPI_PRINT = 300


def apply_thesis_style():
    
    plt.style.use('default')
    
    # Font settings
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZE_LABEL,
        
        # Figure settings
        'figure.figsize': (FIGURE_WIDTH, FIGURE_HEIGHT),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.titlesize': FONT_SIZE_TITLE,
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': COLOR_DARK,
        'axes.linewidth': 1.0,
        'axes.labelsize': FONT_SIZE_LABEL,
        'axes.labelcolor': COLOR_DARK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.titleweight': 'bold',
        'axes.grid': False,
        
        # Grid settings
        'grid.color': COLOR_GRID,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Tick settings
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'xtick.color': COLOR_DARK,
        'ytick.color': COLOR_DARK,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        
        # Legend settings
        'legend.fontsize': FONT_SIZE_LEGEND,
        'legend.framealpha': 0.95,
        'legend.edgecolor': COLOR_GRID,
        'legend.fancybox': False,
    })


def load_csv_data(csv_file: Path) -> pd.DataFrame:    
    if not csv_file.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_file)


def parse_timestamp(ts_str: str) -> float:
    from datetime import datetime
    try:
        dt = datetime.strptime(ts_str, "%Y.%m.%d_%H.%M.%S")
        return dt.timestamp()
    except:
        return np.nan


def calculate_gps_errors(unified_file: Path, waypoints_file: Path) -> pd.DataFrame:
    """Calculate GPS errors for phone/watch from unified dataset."""
    
    if not unified_file.exists():
        return pd.DataFrame()
    
    # Load unified data
    unified_df = pd.read_csv(unified_file)
    
    # Load waypoints
    waypoints = pd.read_csv(waypoints_file)
    
    errors = []
    
    for _, row in unified_df.iterrows():
        try:
            ts = parse_timestamp(row['timestamp'])
            phone_lat, phone_lon = row['smartphone_latitude'], row['smartphone_longitude']
            watch_lat, watch_lon = row['watch_latitude'], row['watch_longitude']
            location = row['location']
            
            # Find nearest waypoint
            nearest_wp = waypoints.iloc[(waypoints['timestamp'].apply(parse_timestamp) - ts).abs().argsort()[:1]]
            
            if not nearest_wp.empty:
                gt_lat = float(nearest_wp.iloc[0]['theoretical_latitude'])
                gt_lon = float(nearest_wp.iloc[0]['theoretical_longitude'])
                
                phone_error = haversine_distance(phone_lat, phone_lon, gt_lat, gt_lon)
                watch_error = haversine_distance(watch_lat, watch_lon, gt_lat, gt_lon)
                
                errors.append({
                    'timestamp': row['timestamp'],
                    'location': location,
                    'phone_error_m': phone_error,
                    'watch_error_m': watch_error,
                    'phone_watch_dist_m': haversine_distance(phone_lat, phone_lon, watch_lat, watch_lon)
                })
        except:
            continue
    
    return pd.DataFrame(errors)


def create_visualizations() -> bool:
    """Create all 6 thesis-ready visualizations with unified styling."""
    
    PipelineMessages.step12_start()
    
    # Apply unified style
    apply_thesis_style()
    
    viz_dir = RESULTS_DIR / "visualization"
    viz_dir.mkdir(exist_ok=True)
    
    # Load analysis results
    device_df = load_csv_data(RESULTS_DIR / "device_comparison.csv")
    location_df = load_csv_data(RESULTS_DIR / "location_comparison.csv")
    cloud_df = load_csv_data(RESULTS_DIR / "cloud_coverage_comparison.csv")
    agreement_df = load_csv_data(RESULTS_DIR / "phone_watch_agreement.csv")
    gps_errors = calculate_gps_errors(RESULTS_DIR / "unified_gps_dataset.csv", DATA_DIR / "ground_truth_waypoints.csv")
    
    # =========================================================================
    # FIGURE 1: LLN Dominance (Location Comparison)
    # =========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    
    loc_metrics = location_df.pivot(index='Location', columns='Device', values='CEP (50th %ile)')
    
    x = np.arange(len(loc_metrics.index))
    width = 0.35
    
    # Plot bars with unified colors
    bars1 = ax.bar(
        x - width/2, 
        loc_metrics.iloc[:, 0], 
        width, 
        label=loc_metrics.columns[0],
        color=COLOR_BLUE,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85
    )
    
    bars2 = ax.bar(
        x + width/2, 
        loc_metrics.iloc[:, 1], 
        width, 
        label=loc_metrics.columns[1],
        color=COLOR_ORANGE,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85
    )
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
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
            )
    
    ax.set_ylabel('CEP (50th percentile, meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title('Figure 1: GPS Accuracy by Location (CEP)', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(loc_metrics.index)
    ax.legend(title='Device', fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor=COLOR_GRID, fancybox=False)
    ax.grid(axis='y', alpha=0.5, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'figure_01_lln_dominance.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Device Comparison (Global Metrics)
    # =========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    
    metrics = ['CEP (50th %ile)', 'R95 (95th %ile)', 'RMSE']
    x = np.arange(len(metrics))
    width = 0.35
    
    phone_vals = device_df[device_df['Device'] == 'Smartphone'][metrics].values.flatten()
    watch_vals = device_df[device_df['Device'] == 'Watch'][metrics].values.flatten()
    
    bars1 = ax.bar(
        x - width/2, 
        phone_vals, 
        width, 
        label='Smartphone',
        color=COLOR_BLUE,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85
    )
    
    bars2 = ax.bar(
        x + width/2, 
        watch_vals, 
        width, 
        label='Watch',
        color=COLOR_ORANGE,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85
    )
    
    # Add value labels
    for bars in [bars1, bars2]:
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
            )
    
    ax.set_ylabel('Error (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title('Figure 2: Global Accuracy Metrics Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor=COLOR_GRID, fancybox=False)
    ax.grid(axis='y', alpha=0.5, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'figure_02_device_comparison.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Phone-Watch Agreement Scatter
    # =========================================================================
    
    if not gps_errors.empty and len(gps_errors) > 10:
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        
        ax.scatter(
            gps_errors['phone_error_m'], 
            gps_errors['watch_error_m'],
            alpha=0.7, 
            s=80, 
            color=COLOR_BLUE,
            edgecolors='black',
            linewidth=1.0
        )
        
        max_err = max(gps_errors['phone_error_m'].max(), gps_errors['watch_error_m'].max())
        
        # Perfect agreement line
        ax.plot([0, max_err], [0, max_err], '--', lw=2, alpha=0.6, label='Perfect Agreement', color=COLOR_DARK)
        
        # 10m threshold line
        ax.plot([0, 10], [10, 0], '--', lw=1.5, alpha=0.5, label='10m Threshold', color=COLOR_GREEN)
        
        ax.set_xlabel('Smartphone Error (m)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('Watch Error (m)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title('Figure 3: Phone-Watch Error Agreement', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor=COLOR_GRID, fancybox=False)
        ax.grid(alpha=0.5, color=COLOR_GRID)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'figure_03_phone_watch_scatter.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # =========================================================================
    # FIGURE 4: Cloud Coverage Trends
    # =========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    
    cloud_pivot = cloud_df.pivot(index='Cloud_Coverage', columns='Device', values='CEP (50th %ile)')
    
    colors_line = [COLOR_BLUE, COLOR_ORANGE]
    for idx, (col, color) in enumerate(zip(cloud_pivot.columns, colors_line)):
        ax.plot(
            cloud_pivot.index, 
            cloud_pivot[col], 
            marker='o', 
            linewidth=2, 
            markersize=8,
            label=col,
            color=color,
            alpha=0.85
        )
    
    ax.set_ylabel('CEP (50th percentile, meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xlabel('Cloud Coverage (0=clear, 5=overcast)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title('Figure 4: GPS Accuracy vs Cloud Coverage', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(title='Device', fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor=COLOR_GRID, fancybox=False)
    ax.grid(alpha=0.5, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'figure_04_cloud_coverage_trend.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Error Distributions (Boxplot)
    # =========================================================================
    
    if not gps_errors.empty:
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        
        box_data = [gps_errors['phone_error_m'].dropna(), gps_errors['watch_error_m'].dropna()]
        
        bp = ax.boxplot(
            box_data, 
            labels=['Smartphone', 'Watch'], 
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(
                marker='D',
                markerfacecolor='red',
                markeredgecolor='darkred',
                markersize=6,
            )
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], [COLOR_BLUE, COLOR_ORANGE]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        
        # Style whiskers and other elements
        for whisker in bp['whiskers']:
            whisker.set_color(COLOR_DARK)
            whisker.set_linewidth(1.0)
        
        for cap in bp['caps']:
            cap.set_color(COLOR_DARK)
            cap.set_linewidth(1.0)
        
        ax.set_ylabel('GPS Error (meters)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title('Figure 5: GPS Error Distributions', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.5, color=COLOR_GRID)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'figure_05_error_distributions.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # =========================================================================
    # FIGURE 6: Location × Device Heatmap
    # =========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    
    loc_pivot = location_df.pivot(index='Location', columns='Device', values='Mean')
    
    sns.heatmap(
        loc_pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlGn_r', 
        center=10, 
        ax=ax,
        cbar_kws={'label': 'Mean Error (m)'},
        linewidths=1.0,
        linecolor='white',
    )
    
    ax.set_title('Figure 6: Mean GPS Error Heatmap (Location × Device)', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'figure_06_location_heatmap.pdf', dpi=DPI_PRINT, bbox_inches='tight', facecolor='white')
    plt.close()
    
    PipelineMessages.step12_complete(len(list(viz_dir.glob('figure_*.pdf'))), str(viz_dir.resolve()))
    
    return True