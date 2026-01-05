import csv
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from src.config.parameters import TIMESTAMP_FORMAT_PARAM, TIME_WINDOW_SECONDS_PARAM, ALLOWED_OFFSET_SECONDS_PARAM


def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        return datetime.strptime(timestamp_str, TIMESTAMP_FORMAT_PARAM)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def load_csv_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV file: {file_path}")
            for row in reader:
                data.append(row)
    except Exception as e:
        raise ValueError(f"Error reading CSV {file_path}: {str(e)}") from e
    
    return data


def find_matching_files(smartphone_dir: Path, watch_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching smartphone and watch CSV files by location and number."""
    matches = []
    smartphone_files = {}
    watch_files = {}
    
    # Index smartphone files
    for f in smartphone_dir.glob('*.csv'):
        if f.name.startswith('ground_truth_'):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                location = parts[2]
                try:
                    number = int(parts[3]) if len(parts) > 3 else None
                    key = (location, number)
                    smartphone_files[key] = f
                except (ValueError, IndexError):
                    pass
    
    # Index watch files
    for f in watch_dir.glob('*.csv'):
        if f.name.startswith('gps_ground_truth_'):
            parts = f.stem.split('_')
            if len(parts) >= 4:
                location = parts[3]
                try:
                    number = int(parts[4]) if len(parts) > 4 else None
                    key = (location, number)
                    watch_files[key] = f
                except (ValueError, IndexError):
                    pass
    
    # Find matching pairs
    for key in smartphone_files:
        if key in watch_files:
            matches.append((smartphone_files[key], watch_files[key]))
    
    return sorted(matches, key=lambda x: str(x[0]))


def detect_time_offset(smartphone_data: List[Dict], watch_data: List[Dict]) -> timedelta:
    """
    Detect time offset between smartphone and watch files.
    
    Returns the timedelta to ADD to watch timestamps to match smartphone.
    """
    if not smartphone_data or not watch_data:
        return timedelta(0)
    
    try:
        sp_first = parse_timestamp(smartphone_data[0].get('timestamp'))
        watch_first = parse_timestamp(watch_data[0].get('timestamp'))
        offset = sp_first - watch_first
        return offset
    except:
        return timedelta(0)


def format_offset(offset: timedelta) -> str:
    """Format timedelta as ±HH:MM string."""
    if offset == timedelta(0):
        return "synchronized"
    
    total_seconds = offset.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((abs(total_seconds) % 3600) // 60)
    sign = '+' if total_seconds >= 0 else '-'
    
    return f"{sign}{abs(hours):02d}:{minutes:02d}"


def merge_by_timestamp(smartphone_data: List[Dict], watch_data: List[Dict],
                      time_window_seconds: int = TIME_WINDOW_SECONDS_PARAM,
                      allowed_offset_seconds: int = ALLOWED_OFFSET_SECONDS_PARAM) -> Tuple[List[Dict], Dict]:
    """
    Merge smartphone and watch data by matching timestamps.
    
    STRICT MATCHING: 4-second window for 10-minute interval data.
    OFFSET CONTROL: Only allows exactly ±1 hour offset (3600 seconds).
    
    Args:
        smartphone_data: List of smartphone GPS records
        watch_data: List of watch GPS records
        time_window_seconds: Allow matching within ±N seconds (default: 4s for 10-min intervals)
        allowed_offset_seconds: Maximum allowed device offset (default: 3600s = ±1 hour)
    
    Returns:
        Tuple of (merged_data, statistics)
    """
    merged = []
    watch_by_time = {}
    
    stats = {
        'smartphone_total': len(smartphone_data),
        'watch_total': len(watch_data),
        'exact_matches': 0,
        'window_matches': 0,
        'total_matched': 0,
        'time_offset_detected': timedelta(0),
        'offset_valid': True,
        'offset_formatted': 'unknown'
    }
    
    # Detect time offset
    offset = detect_time_offset(smartphone_data, watch_data)
    stats['time_offset_detected'] = offset
    stats['offset_formatted'] = format_offset(offset)
    
    # Check if offset is within allowed range (strict control)
    offset_seconds = abs(offset.total_seconds())
    
    if offset_seconds > allowed_offset_seconds:
        stats['offset_valid'] = False
        print(f" ⚠️ REJECTED: Offset {stats['offset_formatted']} exceeds ±1 hour limit")
        return [], stats
    
    # Apply offset to watch timestamps and index by time
    for record in watch_data:
        timestamp_str = record.get('timestamp')
        if timestamp_str:
            try:
                original_time = parse_timestamp(timestamp_str)
                adjusted_time = original_time + offset
                watch_by_time[adjusted_time] = record
            except:
                pass
    
    # Try to match smartphone records with watch records
    for sp_record in smartphone_data:
        timestamp_str = sp_record.get('timestamp')
        if not timestamp_str:
            continue
        
        try:
            sp_time = parse_timestamp(timestamp_str)
        except:
            continue
        
        # First try exact match
        if sp_time in watch_by_time:
            watch_record = watch_by_time[sp_time]
            merged_record = create_merged_record(sp_time.strftime(TIMESTAMP_FORMAT_PARAM),
                                               sp_record, watch_record)
            merged.append(merged_record)
            stats['exact_matches'] += 1
            stats['total_matched'] += 1
        else:
            # Try time window match (strict 4-second window)
            best_match = None
            best_diff = timedelta(seconds=time_window_seconds + 1)
            
            for watch_time, watch_record in watch_by_time.items():
                diff = abs((sp_time - watch_time).total_seconds())
                if diff <= time_window_seconds and diff < best_diff.total_seconds():
                    best_diff = timedelta(seconds=diff)
                    best_match = (watch_time, watch_record)
            
            if best_match:
                watch_time, watch_record = best_match
                merged_record = create_merged_record(sp_time.strftime(TIMESTAMP_FORMAT_PARAM),
                                                   sp_record, watch_record)
                merged.append(merged_record)
                stats['window_matches'] += 1
                stats['total_matched'] += 1
    
    return merged, stats


def create_merged_record(timestamp: str, sp_record: Dict, watch_record: Dict) -> Dict:
    return {
        'timestamp': timestamp,
        'smartphone_longitude': sp_record.get('gps_longitude'),
        'smartphone_latitude': sp_record.get('gps_latitude'),
        'smartphone_accuracy_m': sp_record.get('gps_accuracy_m'),
        'watch_longitude': watch_record.get('gps_longitude'),
        'watch_latitude': watch_record.get('gps_latitude'),
        'watch_accuracy_m': watch_record.get('gps_accuracy_m'),
    }


def write_merged_csv(output_path: Path, merged_data: List[Dict]) -> None:
    if not merged_data:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'timestamp',
        'smartphone_longitude',
        'smartphone_latitude',
        'smartphone_accuracy_m',
        'watch_longitude',
        'watch_latitude',
        'watch_accuracy_m',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)


def extract_location_number(file_path: Path) -> Tuple[str, int]:
    """Extract location name and number from filename."""
    stem = file_path.stem
    if stem.startswith('ground_truth_'):
        parts = stem.replace('ground_truth_', '').split('_')
    else:
        parts = stem.replace('gps_ground_truth_', '').split('_')
    
    location = parts[0]
    number = int(parts[1]) if len(parts) > 1 else 0
    
    return location, number


def process_location_pair(smartphone_file: Path, watch_file: Path, output_base: Path,
                         time_window_seconds: int = TIME_WINDOW_SECONDS_PARAM,
                         allowed_offset_seconds: int = ALLOWED_OFFSET_SECONDS_PARAM) -> Dict:
    """Process a matched pair of smartphone and watch files."""
    result = {
        'smartphone_file': smartphone_file.name,
        'watch_file': watch_file.name,
        'success': False,
        'merged_count': 0,
        'exact_matches': 0,
        'window_matches': 0,
        'smartphone_count': 0,
        'watch_count': 0,
        'time_offset': '',
        'output_file': None,
        'error': None
    }
    
    try:
        # Load data
        smartphone_data = load_csv_data(smartphone_file)
        watch_data = load_csv_data(watch_file)
        
        result['smartphone_count'] = len(smartphone_data)
        result['watch_count'] = len(watch_data)
        
        # Merge by timestamp (with strict time window and offset control)
        merged_data, stats = merge_by_timestamp(
            smartphone_data, watch_data,
            time_window_seconds=time_window_seconds,
            allowed_offset_seconds=allowed_offset_seconds
        )
        
        # Check if offset was valid
        if not stats['offset_valid']:
            result['time_offset'] = stats['offset_formatted']
            result['error'] = f"Offset {stats['offset_formatted']} exceeds ±1 hour limit"
            return result
        
        result['time_offset'] = stats['offset_formatted']
        
        # Determine output location
        location, number = extract_location_number(smartphone_file)
        location_dir = output_base / location
        location_dir.mkdir(parents=True, exist_ok=True)
        
        # Write output
        output_file = location_dir / f"comparison_{location}_{number}.csv"
        write_merged_csv(output_file, merged_data)
        
        result['merged_count'] = len(merged_data)
        result['exact_matches'] = stats['exact_matches']
        result['window_matches'] = stats['window_matches']
        result['output_file'] = str(output_file)
        result['success'] = True
    
    except Exception as e:
        result['error'] = str(e)
    
    return result