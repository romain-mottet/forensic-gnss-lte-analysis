import csv
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from src.config.parameters import (
    DATA_DIR, RESULTS_DIR, TIMESTAMP_FORMAT_PARAM, WAYPOINT_TIME_WINDOW_SECONDS_PARAM
)


def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        return datetime.strptime(timestamp_str, TIMESTAMP_FORMAT_PARAM)
    except ValueError:
        return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
    except (ValueError, TypeError):
        return None

    R = 6371000  # Earth radius in meters
    lat1_rad = lat1 * 3.14159265359 / 180
    lat2_rad = lat2 * 3.14159265359 / 180
    dlat_rad = (lat2 - lat1) * 3.14159265359 / 180
    dlon_rad = (lon2 - lon1) * 3.14159265359 / 180
    
    a = (dlat_rad / 2) ** 2 + (dlon_rad / 2) ** 2
    a = a * (1 + 0.001 * (lat1 + lat2) / 2 / 90)
    c = 2 * (a ** 0.5) if a < 1 else 2
    
    return R * c


def load_waypoints(waypoints_file: Path) -> List[Dict]:
    waypoints = []
    with open(waypoints_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append(row)
    return waypoints


def load_unified_dataset(unified_file: Path) -> List[Dict]:
    measurements = []
    with open(unified_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurements.append(row)
    return measurements


def find_closest_measurement(waypoint_time: datetime, measurements: List[Dict],
                           location: str, session: int) -> Tuple[Dict, float]:

    closest = None
    min_time_diff = float('inf')
    
    for meas in measurements:
        if (meas.get('location') == location and
            str(meas.get('session_number', '')) == str(session)):
            meas_time = parse_timestamp(meas['timestamp'])
            if meas_time:
                time_diff = abs((waypoint_time - meas_time).total_seconds())
                if time_diff > WAYPOINT_TIME_WINDOW_SECONDS_PARAM:
                    continue
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest = meas
    
    return closest, min_time_diff


def get_session_waypoint_details(location: str, session: int) -> Dict:
    """
    Get DETAILED waypoint data for specific location/session (for reports).
    Returns: {'sessions': [{'location', 'session', 'smartphone_distances', 'watch_distances', 
                           'waypoints': [...], 'filtered_count'}]}
    """
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists() or not unified_file.exists():
        return {'sessions': []}
    
    waypoints = load_waypoints(waypoints_file)
    measurements = load_unified_dataset(unified_file)
    
    # Filter waypoints for this session
    session_waypoints = []
    for wp in waypoints:
        if (wp['location'] == location and 
            int(float(wp['session_number'])) == session):
            session_waypoints.append(wp)
    
    session_results = {
        'location': location,
        'session': session,
        'smartphone_distances': [],
        'watch_distances': [],
        'waypoints': [],
        'filtered_count': 0
    }
    
    for wp in session_waypoints:
        wp_time = parse_timestamp(wp['timestamp'])
        if not wp_time:
            session_results['filtered_count'] += 1
            continue
        
        closest_meas, time_diff = find_closest_measurement(
            wp_time, measurements, location, session)
        
        if closest_meas and time_diff < float('inf'):
            try:
                wp_lat = float(wp['theoretical_latitude'])
                wp_lon = float(wp['theoretical_longitude'])
                
                # Smartphone distance
                sp_lat = float(closest_meas['smartphone_latitude'])
                sp_lon = float(closest_meas['smartphone_longitude'])
                sp_dist = haversine_distance(wp_lat, wp_lon, sp_lat, sp_lon)
                
                # Watch distance
                watch_lat = float(closest_meas['watch_latitude'])
                watch_lon = float(closest_meas['watch_longitude'])
                watch_dist = haversine_distance(wp_lat, wp_lon, watch_lat, watch_lon)
                
                if sp_dist is not None and watch_dist is not None:
                    session_results['smartphone_distances'].append(sp_dist)
                    session_results['watch_distances'].append(watch_dist)
                    session_results['waypoints'].append({
                        'waypoint_name': wp['waypoint_name'],
                        'waypoint_id': wp['waypoint_id'],
                        'time_diff_s': time_diff,
                        'smartphone_dist_m': sp_dist,
                        'watch_dist_m': watch_dist
                    })
                else:
                    session_results['filtered_count'] += 1
                    
            except (ValueError, KeyError):
                session_results['filtered_count'] += 1
        else:
            session_results['filtered_count'] += 1
    
    return {'sessions': [session_results]}


def _calculate_distances_for_waypoints(waypoints: List[Dict], measurements: List[Dict],
                                    session_filter: Dict = None) -> Dict:
    """
    Internal: Calculate distances for filtered waypoints (distances only).
    Returns: {'smartphone_distances': [floats], 'watch_distances': [floats], 'total_samples': int}
    """
    smartphone_distances = []
    watch_distances = []
    total_samples = 0
    
    for wp in waypoints:
        # Apply session filter if provided
        if session_filter:
            if session_filter.get('location') and wp['location'] != session_filter['location']:
                continue
            if session_filter.get('session') and int(float(wp['session_number'])) != session_filter['session']:
                continue
            if session_filter.get('cloud_coverage') and wp.get('cloud_coverage') != session_filter['cloud_coverage']:
                continue
        
        wp_time = parse_timestamp(wp['timestamp'])
        if not wp_time:
            continue
        
        location = wp['location']
        session = int(float(wp['session_number']))
        
        closest_meas, time_diff = find_closest_measurement(wp_time, measurements, location, session)
        
        if closest_meas and time_diff < float('inf'):
            try:
                wp_lat = float(wp['theoretical_latitude'])
                wp_lon = float(wp['theoretical_longitude'])
                
                sp_lat = float(closest_meas['smartphone_latitude'])
                sp_lon = float(closest_meas['smartphone_longitude'])
                sp_dist = haversine_distance(wp_lat, wp_lon, sp_lat, sp_lon)
                
                watch_lat = float(closest_meas['watch_latitude'])
                watch_lon = float(closest_meas['watch_longitude'])
                watch_dist = haversine_distance(wp_lat, wp_lon, watch_lat, watch_lon)
                
                if sp_dist is not None and watch_dist is not None:
                    smartphone_distances.append(sp_dist)
                    watch_distances.append(watch_dist)
                    total_samples += 1
                    
            except (ValueError, KeyError):
                continue
    
    return {
        'smartphone_distances': smartphone_distances,
        'watch_distances': watch_distances,
        'total_samples': total_samples
    }


def get_all_waypoint_distances() -> Dict:
    """Get ALL waypoint distances across ALL sessions/locations."""
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists() or not unified_file.exists():
        return {'smartphone_distances': [], 'watch_distances': [], 'total_samples': 0}
    
    waypoints = load_waypoints(waypoints_file)
    measurements = load_unified_dataset(unified_file)
    
    return _calculate_distances_for_waypoints(waypoints, measurements)


def get_session_waypoint_distances(location: str, session: int) -> Dict:
    """Get waypoint distances for specific location/session (distances only)."""
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists() or not unified_file.exists():
        return {'smartphone_distances': [], 'watch_distances': [], 'total_samples': 0}
    
    waypoints = load_waypoints(waypoints_file)
    measurements = load_unified_dataset(unified_file)
    
    return _calculate_distances_for_waypoints(
        waypoints, measurements,
        session_filter={'location': location, 'session': session}
    )


def get_location_waypoint_distances(location: str) -> Dict:
    """Get waypoint distances for specific location (all sessions)."""
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists() or not unified_file.exists():
        return {'smartphone_distances': [], 'watch_distances': [], 'total_samples': 0}
    
    waypoints = load_waypoints(waypoints_file)
    measurements = load_unified_dataset(unified_file)
    
    return _calculate_distances_for_waypoints(
        waypoints, measurements,
        session_filter={'location': location}
    )


def get_cloud_coverage_distances(cloud_category: str) -> Dict:
    """Get waypoint distances filtered by cloud coverage category."""
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists() or not unified_file.exists():
        return {'smartphone_distances': [], 'watch_distances': [], 'total_samples': 0}
    
    waypoints = load_waypoints(waypoints_file)
    measurements = load_unified_dataset(unified_file)
    
    return _calculate_distances_for_waypoints(
        waypoints, measurements,
        session_filter={'cloud_coverage': cloud_category}
    )
