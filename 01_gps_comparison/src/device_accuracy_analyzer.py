import csv
import statistics
import math
from pathlib import Path
from typing import Dict

from src.helpers.waypoint_helper import get_all_waypoint_distances
from src.config.parameters import RESULTS_DIR
from src.helpers.print_helper import PipelineMessages


def calculate_accuracy_metrics(distances: list) -> Dict:
    """    
    Args:
        distances: List of distance measurements in meters
        
    Returns:
        Dict with CEP (50th), R95 (95th), RMSE, Mean, Std Dev, Samples
    """
    if not distances:
        return {
            'cep_50': 0.0, 'r95': 0.0, 'rmse': 0.0, 
            'mean': 0.0, 'std_dev': 0.0, 'samples': 0
        }
    
    n = len(distances)
    sorted_distances = sorted(distances)
    
    cep_50_idx = int(n * 0.5) - 1 if n > 0 else 0
    cep_50 = sorted_distances[cep_50_idx] if n > 0 else 0.0
    
    r95_idx = int(n * 0.95) - 1 if n > 0 else 0
    r95 = sorted_distances[r95_idx] if n > 0 else 0.0
    
    mean_dist = statistics.mean(distances)
    
    std_dev = statistics.stdev(distances) if n > 1 else 0.0
    
    rmse = math.sqrt(statistics.mean([d**2 for d in distances]))
    
    return {
        'cep_50': round(cep_50, 1),
        'r95': round(r95, 1),
        'rmse': round(rmse, 1),
        'mean': round(mean_dist, 1),
        'std_dev': round(std_dev, 1),
        'samples': n
    }


def run_device_accuracy_analysis() -> bool:
    PipelineMessages.step8_start()
    
    try:
        # Get all waypoint distances (already filtered by WAYPOINT_TIME_WINDOW_SECONDS_PARAM)
        distances_data = get_all_waypoint_distances()
        smartphone_dists = distances_data['smartphone_distances']
        watch_dists = distances_data['watch_distances']
        total_samples = distances_data['total_samples']
        
        if total_samples == 0:
            PipelineMessages.step8_no_data()
            return False
        
        # Calculate metrics for both devices
        smartphone_metrics = calculate_accuracy_metrics(smartphone_dists)
        watch_metrics = calculate_accuracy_metrics(watch_dists)
        
        # Save to CSV
        output_file = RESULTS_DIR / "device_comparison.csv"
        RESULTS_DIR.mkdir(exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Device', 'CEP (50th %ile)', 'R95 (95th %ile)', 
                'RMSE', 'Mean', 'Std Dev', 'Samples'
            ])
            writer.writerow([
                'Smartphone', 
                smartphone_metrics['cep_50'],
                smartphone_metrics['r95'],
                smartphone_metrics['rmse'],
                smartphone_metrics['mean'],
                smartphone_metrics['std_dev'],
                smartphone_metrics['samples']
            ])
            writer.writerow([
                'Watch',
                watch_metrics['cep_50'],
                watch_metrics['r95'],
                watch_metrics['rmse'],
                watch_metrics['mean'],
                watch_metrics['std_dev'],
                watch_metrics['samples']
            ])
        
        PipelineMessages.step8_summary_success(
            smartphone_metrics['samples'], 
            watch_metrics['samples'],
            str(output_file.resolve())
        )
        
        return True
        
    except Exception as e:
        PipelineMessages.step8_error(str(e))
        return False
