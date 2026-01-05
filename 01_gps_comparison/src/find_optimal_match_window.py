import csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import statistics

from src.config.parameters import (
    TIMESTAMP_FORMAT_PARAM, MINIMUM_MATCH_RATE_PARAM, MAXIMUM_MEAN_ERROR_PARAM
)


def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        return datetime.strptime(timestamp_str, TIMESTAMP_FORMAT_PARAM)
    except ValueError:
        return None


def load_csv_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


def detect_time_offset(smartphone_data: List[Dict], watch_data: List[Dict]) -> int:
    if not smartphone_data or not watch_data:
        return 0
    
    sp_time = parse_timestamp(smartphone_data[0].get('timestamp'))
    watch_time = parse_timestamp(watch_data[0].get('timestamp'))
    
    if sp_time and watch_time:
        offset = (sp_time - watch_time).total_seconds()
        return int(offset)
    
    return 0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
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


def test_time_window(smartphone_data: List[Dict], watch_data: List[Dict],
                     time_window_seconds: float, offset_seconds: int) -> Dict:
    """
    Test a specific time window for matching.
    
    Args:
        smartphone_data: List of smartphone GPS records
        watch_data: List of watch GPS records
        time_window_seconds: Time tolerance in seconds (e.g., 0.5, 1.0, 1.5)
        offset_seconds: Known time offset between devices
    
    Returns:
        Dictionary with metrics: match_rate, matched_count, mean_error_m, max_error_m, qualified
    """
    matched = 0
    errors = []
    unmatched = []
    
    for sp_record in smartphone_data:
        sp_timestamp_str = sp_record.get('timestamp')
        if not sp_timestamp_str:
            continue
        
        sp_time = parse_timestamp(sp_timestamp_str)
        if not sp_time:
            continue
        
        # Find closest watch record within time window
        best_match = None
        best_diff = time_window_seconds + 0.1  # Small buffer
        
        for watch_record in watch_data:
            watch_timestamp_str = watch_record.get('timestamp')
            if not watch_timestamp_str:
                continue
            
            watch_time = parse_timestamp(watch_timestamp_str)
            if not watch_time:
                continue
            
            # Apply offset and calculate time difference
            adjusted_watch_time = watch_time + timedelta(seconds=offset_seconds)
            time_diff = abs((sp_time - adjusted_watch_time).total_seconds())
            
            if time_diff <= time_window_seconds and time_diff < best_diff:
                best_diff = time_diff
                best_match = (watch_record, time_diff)
        
        if best_match:
            watch_record, time_diff = best_match
            matched += 1
            
            # Calculate distance error
            distance = haversine_distance(
                sp_record.get('gps_longitude'),
                sp_record.get('gps_latitude'),
                watch_record.get('gps_longitude'),
                watch_record.get('gps_latitude')
            )
            
            if distance is not None:
                errors.append(distance)
            else:
                unmatched.append(sp_timestamp_str)
    
    total = len(smartphone_data)
    match_rate = (matched / total * 100) if total > 0 else 0
    mean_error = statistics.mean(errors) if errors else None
    max_error = max(errors) if errors else None
    
    # Check if this window qualifies (BOTH thresholds must be met)
    qualifies = (
        match_rate >= MINIMUM_MATCH_RATE_PARAM and
        mean_error is not None and
        mean_error <= MAXIMUM_MEAN_ERROR_PARAM
    )
    
    return {
        'time_window_seconds': time_window_seconds,
        'matched_count': matched,
        'total_count': total,
        'match_rate': match_rate,
        'mean_error_m': mean_error,
        'max_error_m': max_error,
        'unmatched_count': len(unmatched),
        'qualifies': qualifies  # NEW: qualification status
    }


def find_optimal_window(smartphone_file: Path, watch_file: Path,
                        test_windows: List[float] = None) -> Dict:
    """
    Test multiple time windows and find the optimal one.
    
    Selection criteria:
    1. Filter: Match rate >= 85% AND Mean error <= 10m
    2. Among qualified windows: Select the one with LOWEST mean error
    3. If no windows qualify: Return the one closest to thresholds
    
    Args:
        smartphone_file: Path to smartphone ground truth CSV
        watch_file: Path to watch ground truth CSV
        test_windows: List of time windows to test (default: 0.5s to 8.0s in 0.5s increments)
    
    Returns:
        Dictionary with results for all tested windows and recommendation
    """
    if test_windows is None:
        from src.config.parameters import TEST_WINDOWS_PARAM
        test_windows = TEST_WINDOWS_PARAM
    
    try:
        smartphone_data = load_csv_data(smartphone_file)
        watch_data = load_csv_data(watch_file)
    except Exception as e:
        return {'error': str(e)}
    
    if not smartphone_data or not watch_data:
        return {'error': 'Empty data files'}
    
    # Detect offset
    offset = detect_time_offset(smartphone_data, watch_data)
    
    # Test each window
    results = {
        'file_pair': {
            'smartphone': smartphone_file.name,
            'watch': watch_file.name
        },
        'offset_seconds': offset,
        'test_results': []
    }
    
    qualified_results = []
    best_result = None
    best_score = float('inf')  # Lower is better for error
    
    for window in sorted(test_windows):
        test_result = test_time_window(smartphone_data, watch_data, window, offset)
        test_result['time_window_seconds'] = window
        results['test_results'].append(test_result)
        
        # Track qualified candidates
        if test_result['qualifies']:
            qualified_results.append(test_result)
            # Among qualified, select lowest error
            if test_result['mean_error_m'] is not None and test_result['mean_error_m'] < best_score:
                best_score = test_result['mean_error_m']
                best_result = test_result
    
    # If no windows qualify, find the one with best combined score
    # (closest to meeting both thresholds)
    if best_result is None:
        best_score = float('inf')
        for test_result in results['test_results']:
            if test_result['match_rate'] is not None and test_result['mean_error_m'] is not None:
                # Score: penalize not meeting match rate AND error threshold
                match_penalty = max(0, MINIMUM_MATCH_RATE_PARAM - test_result['match_rate']) * 2
                error_penalty = max(0, test_result['mean_error_m'] - MAXIMUM_MEAN_ERROR_PARAM) * 1
                score = match_penalty + error_penalty
                
                if score < best_score:
                    best_score = score
                    best_result = test_result
    
    results['optimal_window'] = best_result['time_window_seconds'] if best_result else None
    results['best_match_rate'] = best_result['match_rate'] if best_result else None
    results['best_mean_error'] = best_result['mean_error_m'] if best_result else None
    results['best_matched_count'] = best_result['matched_count'] if best_result else None
    results['qualified_windows_count'] = len(qualified_results)
    results['qualified_windows'] = [r['time_window_seconds'] for r in qualified_results]
    
    return results


def format_results_table(results: Dict) -> str:
    if 'error' in results:
        return f"Error: {results['error']}"
    
    lines = []
    lines.append(f"\n{'='*110}")
    lines.append(f" {results['file_pair']['smartphone']} {results['file_pair']['watch']}")
    lines.append(f" Time offset: {results['offset_seconds']:+d} seconds")
    lines.append(f"{'='*110}")
    
    # Quality thresholds info
    lines.append(f" Quality Thresholds: Match Rate {MINIMUM_MATCH_RATE_PARAM:.1f}% AND Mean Error {MAXIMUM_MEAN_ERROR_PARAM:.1f}m")
    lines.append(f"{'='*110}")
    
    lines.append(f"{'Window':<10} {'Matched':<12} {'Rate':<10} {'Mean Err':<12} {'Max Err':<12} {'Qualifies':<12} {'Status':<15}")
    lines.append(f"{'-'*110}")
    
    for result in results['test_results']:
        window = result['time_window_seconds']
        matched = f"{result['matched_count']}/{result['total_count']}"
        rate = f"{result['match_rate']:.1f}%"
        mean_err = f"{result['mean_error_m']:.1f}m" if result['mean_error_m'] else "N/A"
        max_err = f"{result['max_error_m']:.1f}m" if result['max_error_m'] else "N/A"
        qualifies = "YES" if result['qualifies'] else "NO"
        
        # Highlight optimal and current recommendation
        if window == results['optimal_window'] and result['qualifies']:
            status = "OPTIMAL"
        elif window == results['optimal_window']:
            status = "FALLBACK"
        else:
            status = ""
        
        lines.append(f"{window:<10.1f} {matched:<12} {rate:<10} {mean_err:<12} {max_err:<12} {qualifies:<12} {status:<15}")
    
    lines.append(f"{'='*110}")
    
    # Summary section
    lines.append(f"\n ANALYSIS SUMMARY:")
    lines.append(f"  Total windows tested: {len(results['test_results'])}")
    lines.append(f"  Qualified windows (meet both thresholds): {results['qualified_windows_count']}")
    if results['qualified_windows']:
        lines.append(f"  Qualified windows: {', '.join([f'{w:.1f}s' for w in results['qualified_windows']])}")
    
    lines.append(f"\OPTIMAL WINDOW SELECTION:")
    if results['best_match_rate'] is not None:
        lines.append(f"  Recommended window: {results['optimal_window']:.1f} seconds")
        
        # Show if this is a qualified window
        for result in results['test_results']:
            if result['time_window_seconds'] == results['optimal_window']:
                qualifies_text = "QUALIFIES" if result['qualifies'] else "— Does not fully qualify"
                lines.append(f"  Status: {qualifies_text}")
                break
        
        lines.append(f"  Match rate: {results['best_match_rate']:.1f}% (threshold: {MINIMUM_MATCH_RATE_PARAM:.1f}%)")
        lines.append(f"  Mean error: {results['best_mean_error']:.1f}m (threshold: {MAXIMUM_MEAN_ERROR_PARAM:.1f}m)")
        lines.append(f"  Records merged: {results['best_matched_count']}")
    else:
        lines.append(f"  Could not determine optimal window")
    
    lines.append(f"{'='*110}\n")
    
    return '\n'.join(lines)