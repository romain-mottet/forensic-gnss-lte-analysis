import csv
import statistics
import math
from pathlib import Path
from typing import Dict, List
from collections import Counter

from src.config.parameters import DATA_DIR, RESULTS_DIR
from src.helpers.waypoint_helper import get_cloud_coverage_distances
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


def get_cloud_categories() -> List[str]:
    """Dynamically discover unique cloud_coverage values from waypoints."""
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    if not waypoints_file.exists():
        return []
    
    cloud_values = set()
    with open(waypoints_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cloud_val = row.get('cloud_coverage', '').strip()
            if cloud_val:
                cloud_values.add(cloud_val)
    
    # Convert to sorted list of strings for consistent output
    categories = sorted([str(val) for val in cloud_values], key=lambda x: int(x) if x.isdigit() else x)
    return categories


def run_cloud_accuracy_analysis() -> bool:
    PipelineMessages.step10_start()
    
    try:
        # Discover cloud categories dynamically
        categories = get_cloud_categories()
        if not categories:
            PipelineMessages.step10_no_categories()
            return False
        
        PipelineMessages.step10_discovered_categories(len(categories), categories)
        
        # Calculate metrics for each cloud category + device
        all_results = []
        output_file = RESULTS_DIR / "cloud_coverage_comparison.csv"
        RESULTS_DIR.mkdir(exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Cloud_Coverage', 'Device', 'CEP (50th %ile)', 'R95 (95th %ile)', 
                'RMSE', 'Mean', 'Std Dev', 'Samples'
            ])
            
            for category in categories:
                # Get waypoint distances for this cloud category
                distances_data = get_cloud_coverage_distances(category)
                smartphone_dists = distances_data['smartphone_distances']
                watch_dists = distances_data['watch_distances']
                samples = distances_data['total_samples']
                
                if samples == 0:
                    PipelineMessages.step10_category_no_data(category)
                    continue
                
                # Calculate metrics
                smartphone_metrics = calculate_accuracy_metrics(smartphone_dists)
                watch_metrics = calculate_accuracy_metrics(watch_dists)
                
                # Write Smartphone row
                writer.writerow([
                    category, 'Smartphone',
                    smartphone_metrics['cep_50'], smartphone_metrics['r95'],
                    smartphone_metrics['rmse'], smartphone_metrics['mean'],
                    smartphone_metrics['std_dev'], smartphone_metrics['samples']
                ])
                
                # Write Watch row
                writer.writerow([
                    category, 'Watch',
                    watch_metrics['cep_50'], watch_metrics['r95'],
                    watch_metrics['rmse'], watch_metrics['mean'],
                    watch_metrics['std_dev'], watch_metrics['samples']
                ])
                
                all_results.append({
                    'category': category,
                    'smartphone_samples': smartphone_metrics['samples'],
                    'watch_samples': watch_metrics['samples']
                })
        
        # Summary
        total_categories = len(all_results)
        PipelineMessages.step10_summary_success(
            total_categories, str(output_file.resolve()), all_results
        )
        
        return True
        
    except Exception as e:
        PipelineMessages.step10_error(str(e))
        return False
