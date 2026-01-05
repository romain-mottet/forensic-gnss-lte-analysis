import csv
import statistics
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.config.parameters import (
    DATA_DIR, RESULTS_DIR, WAYPOINT_TIME_WINDOW_SECONDS_PARAM, 
    PHONE_WATCH_DISTANCE_THRESHOLD  # NEW parameter
)
from src.helpers.waypoint_helper import haversine_distance
from src.helpers.print_helper import PipelineMessages


def parse_timestamp(ts_str: str) -> float:
    from datetime import datetime
    dt = datetime.strptime(ts_str, "%Y.%m.%d_%H.%M.%S")
    return dt.timestamp()


def find_nearest_waypoint(unified_row: dict, waypoints: List[dict]) -> Optional[dict]:
    """Find nearest waypoint within time window for given unified dataset row."""
    row_ts = parse_timestamp(unified_row['timestamp'])
    best_waypoint = None
    min_time_diff = float('inf')
    
    for waypoint in waypoints:
        waypoint_ts = parse_timestamp(waypoint['timestamp']) if waypoint['timestamp'] else None
        if waypoint_ts is None:
            continue
            
        time_diff = abs(row_ts - waypoint_ts)
        if time_diff <= WAYPOINT_TIME_WINDOW_SECONDS_PARAM and time_diff < min_time_diff:
            min_time_diff = time_diff
            best_waypoint = waypoint
    
    return best_waypoint


def calculate_phone_watch_distance(row: dict) -> float:
    phone_lat = float(row['smartphone_latitude'])
    phone_lon = float(row['smartphone_longitude'])
    watch_lat = float(row['watch_latitude'])
    watch_lon = float(row['watch_longitude'])
    return haversine_distance(phone_lat, phone_lon, watch_lat, watch_lon)


def determine_closer_device(phone_dist: float, watch_dist: float) -> str:
    if phone_dist < watch_dist:
        return 'Smartphone'
    elif watch_dist < phone_dist:
        return 'Watch'
    else:
        return 'Tie'


def run_phone_watch_agreement_analysis() -> bool:
    PipelineMessages.step11_start()
    
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv" 
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    output_file = RESULTS_DIR / "phone_watch_agreement.csv"
    
    if not unified_file.exists():
        PipelineMessages.step11_error_no_unified()
        print(f"  Expected: {unified_file.resolve()}")
        return False
    
    if not waypoints_file.exists():
        PipelineMessages.step11_error_no_waypoints()
        return False
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load waypoints
    waypoints = []
    with open(waypoints_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        waypoints = list(reader)
    
    PipelineMessages.step11_loading_data(len(waypoints))
    
    # Process unified dataset
    distances = []
    disagreement_analysis = defaultdict(int)  # >10m: smartphone_closer, watch_closer, tie
    
    with open(unified_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = 0
        valid_rows = 0
        
        for row in reader:
            total_rows += 1
            
            # Skip if missing coordinates
            if not all(k in row for k in ['smartphone_latitude', 'smartphone_longitude', 
                                        'watch_latitude', 'watch_longitude']):
                continue
            
            try:
                phone_lat, phone_lon = float(row['smartphone_latitude']), float(row['smartphone_longitude'])
                watch_lat, watch_lon = float(row['watch_latitude']), float(row['watch_longitude'])
            except ValueError:
                continue
            
            pw_distance = calculate_phone_watch_distance(row)
            distances.append(pw_distance)
            valid_rows += 1
            
            # Agreement analysis
            if pw_distance > PHONE_WATCH_DISTANCE_THRESHOLD:
                waypoint = find_nearest_waypoint(row, waypoints)
                if waypoint:
                    # Calculate distances to ground truth (theoretical coordinates)
                    gt_lat = float(waypoint['theoretical_latitude'])
                    gt_lon = float(waypoint['theoretical_longitude'])
                    
                    phone_gt_dist = haversine_distance(phone_lat, phone_lon, gt_lat, gt_lon)
                    watch_gt_dist = haversine_distance(watch_lat, watch_lon, gt_lat, gt_lon)
                    
                    winner = determine_closer_device(phone_gt_dist, watch_gt_dist)
                    disagreement_analysis[winner] += 1
    
    if not distances:
        PipelineMessages.step11_no_valid_data()
        return False
    
    # Calculate statistics
    distances_sorted = sorted(distances)
    n = len(distances)
    
    stats = {
        'total_timestamps': n,
        'mean_distance_m': round(statistics.mean(distances), 1),
        'std_dev_m': round(statistics.stdev(distances), 1) if n > 1 else 0.0,
        'min_distance_m': round(min(distances), 1),
        'max_distance_m': round(max(distances), 1),
        'median_distance_m': round(statistics.median(distances), 1),
        'p95_distance_m': round(distances_sorted[int(n * 0.95) - 1], 1),
        'agreement_rate_10m_percent': round(
            (sum(1 for d in distances if d <= PHONE_WATCH_DISTANCE_THRESHOLD) / n) * 100, 1
        ),
        'disagreements_gt_10m': len(distances) - sum(1 for d in distances if d <= PHONE_WATCH_DISTANCE_THRESHOLD),
        'smartphone_closer': disagreement_analysis['Smartphone'],
        'watch_closer': disagreement_analysis['Watch'],
        'ties': disagreement_analysis['Tie']
    }
    
    # Write summary CSV (single row)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Metric', 'Value'
        ])
        writer.writerow(['Total Timestamps', stats['total_timestamps']])
        writer.writerow(['Mean Phone-Watch Distance (m)', stats['mean_distance_m']])
        writer.writerow(['Std Dev Distance (m)', stats['std_dev_m']])
        writer.writerow(['Min Distance (m)', stats['min_distance_m']])
        writer.writerow(['Max Distance (m)', stats['max_distance_m']])
        writer.writerow(['Median Distance (m)', stats['median_distance_m']])
        writer.writerow(['95th Percentile Distance (m)', stats['p95_distance_m']])
        writer.writerow(['Agreement Rate ≤10m (%)', stats['agreement_rate_10m_percent']])
        writer.writerow(['Disagreements >10m', stats['disagreements_gt_10m']])
        writer.writerow(['When >10m: Smartphone closer', stats['smartphone_closer']])
        writer.writerow(['When >10m: Watch closer', stats['watch_closer']])
        writer.writerow(['When >10m: Tie', stats['ties']])
    
    PipelineMessages.step11_summary_success(
        stats['total_timestamps'], 
        stats['agreement_rate_10m_percent'],
        stats['mean_distance_m'],
        str(output_file.resolve())
    )
    
    return True
