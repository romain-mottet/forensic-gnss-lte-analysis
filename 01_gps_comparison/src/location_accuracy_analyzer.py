import csv
import statistics
import math
from pathlib import Path
from typing import Dict, List

from src.config.parameters import SMARTPHONE_DIR, RESULTS_DIR
from src.helpers.waypoint_helper import get_location_waypoint_distances
from src.helpers.print_helper import PipelineMessages


def calculate_accuracy_metrics(distances: list) -> Dict:
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


def get_locations() -> List[str]:
    """Dynamically discover locations from data/SMARTPHONE/ directory structure."""
    if not SMARTPHONE_DIR.exists():
        return []
    
    locations = []
    for location_dir in SMARTPHONE_DIR.iterdir():
        if location_dir.is_dir():
            locations.append(location_dir.name)
    
    return sorted(locations)


def run_location_accuracy_analysis() -> bool:
    PipelineMessages.step9_start()
    
    try:
        # Discover locations dynamically
        locations = get_locations()
        if not locations:
            PipelineMessages.step9_no_locations()
            return False
        
        PipelineMessages.step9_discovered_locations(len(locations), locations)
        
        # Calculate metrics for each location + device
        all_results = []
        output_file = RESULTS_DIR / "location_comparison.csv"
        RESULTS_DIR.mkdir(exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Location', 'Device', 'CEP (50th %ile)', 'R95 (95th %ile)', 
                'RMSE', 'Mean', 'Std Dev', 'Samples'
            ])
            
            for location in locations:
                # Get waypoint distances for this location (all sessions)
                distances_data = get_location_waypoint_distances(location)
                smartphone_dists = distances_data['smartphone_distances']
                watch_dists = distances_data['watch_distances']
                samples = distances_data['total_samples']
                
                if samples == 0:
                    PipelineMessages.step9_location_no_data(location)
                    continue
                
                # Calculate metrics
                smartphone_metrics = calculate_accuracy_metrics(smartphone_dists)
                watch_metrics = calculate_accuracy_metrics(watch_dists)
                
                # Write Smartphone row
                writer.writerow([
                    location, 'Smartphone',
                    smartphone_metrics['cep_50'], smartphone_metrics['r95'],
                    smartphone_metrics['rmse'], smartphone_metrics['mean'],
                    smartphone_metrics['std_dev'], smartphone_metrics['samples']
                ])
                
                # Write Watch row
                writer.writerow([
                    location, 'Watch',
                    watch_metrics['cep_50'], watch_metrics['r95'],
                    watch_metrics['rmse'], watch_metrics['mean'],
                    watch_metrics['std_dev'], watch_metrics['samples']
                ])
                
                all_results.append({
                    'location': location,
                    'smartphone_samples': smartphone_metrics['samples'],
                    'watch_samples': watch_metrics['samples']
                })
        
        # Summary
        total_locations = len(all_results)
        PipelineMessages.step9_summary_success(
            total_locations, str(output_file.resolve()), all_results
        )
        
        return True
        
    except Exception as e:
        PipelineMessages.step9_error(str(e))
        return False
