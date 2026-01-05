from datetime import datetime
from typing import List, Dict, Tuple, Optional


def parse_timestamp(timestamp_str: str, format_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(timestamp_str, format_str)
    except (ValueError, TypeError):
        return None


def find_duplicate_timestamps(rows: List[Dict]) -> Dict[str, List[int]]:
    """    
    Args:
        rows: List of CSV rows as dictionaries
    
    Returns:
        Dict mapping timestamp -> list of row indices where it appears
        Only includes timestamps that appear > 1 time
    """
    timestamp_indices: Dict[str, List[int]] = {}
    
    for i, row in enumerate(rows):
        ts = row.get('timestamp', '')
        if ts:
            timestamp_indices.setdefault(ts, []).append(i)
    
    # Return only duplicates (timestamps appearing > 1 time)
    return {ts: idxs for ts, idxs in timestamp_indices.items() if len(idxs) > 1}


def is_cache_duplicate(row_idx: int, rows: List[Dict], duplicates: Dict[str, List[int]]) -> bool:
    """
    Check if a row is a duplicate timestamp AND has identical coordinates.
    
    Args:
        row_idx: Index of the row to check
        rows: List of all CSV rows
        duplicates: Dict of duplicate timestamps (from find_duplicate_timestamps)
    
    Returns:
        True if this row is a cache duplicate, False otherwise
    """
    if row_idx >= len(rows):
        return False
    
    current_row = rows[row_idx]
    current_ts = current_row.get('timestamp', '')
    
    if current_ts not in duplicates:
        return False
    
    try:
        current_lon = float(current_row.get('smartphone_longitude', ''))
        current_lat = float(current_row.get('smartphone_latitude', ''))
    except (ValueError, TypeError):
        return False
    
    # Check if another row with same timestamp has identical coordinates
    for idx in duplicates[current_ts]:
        if idx == row_idx:
            continue
        
        other_row = rows[idx]
        try:
            other_lon = float(other_row.get('smartphone_longitude', ''))
            other_lat = float(other_row.get('smartphone_latitude', ''))
            
            # If coordinates match, this is a cache duplicate
            if current_lon == other_lon and current_lat == other_lat:
                return True
        except (ValueError, TypeError):
            continue
    
    return False


def find_major_gaps(timestamps: List[str], format_str: str, gap_threshold: int) -> List[Dict]:
    """    
    Args:
        timestamps: List of timestamp strings (in order)
        format_str: Timestamp format string
        gap_threshold: Minimum gap size in seconds to flag as major
    
    Returns:
        List of gap dictionaries with keys:
        - prev_index: Index of record before gap
        - curr_index: Index of record after gap
        - prev_timestamp: Timestamp before gap
        - curr_timestamp: Timestamp after gap
        - gap_seconds: Size of gap in seconds
    """
    gaps = []
    
    if len(timestamps) < 2:
        return gaps
    
    for i in range(1, len(timestamps)):
        t_prev_str = timestamps[i - 1]
        t_curr_str = timestamps[i]
        
        t_prev = parse_timestamp(t_prev_str, format_str)
        t_curr = parse_timestamp(t_curr_str, format_str)
        
        if t_prev is None or t_curr is None:
            continue
        
        gap_seconds = (t_curr - t_prev).total_seconds()
        
        if gap_seconds >= gap_threshold:
            gaps.append({
                'prev_index': i - 1,
                'curr_index': i,
                'prev_timestamp': t_prev_str,
                'curr_timestamp': t_curr_str,
                'gap_seconds': gap_seconds,
            })
    
    return gaps


def find_near_major_gap_indices(
    timestamps: List[str],
    format_str: str,
    major_gaps: List[Dict],
    window_seconds: int
) -> set:

    near_gap_indices = set()
    
    if not major_gaps:
        return near_gap_indices
    
    for gap in major_gaps:
        # Get the timestamp at the gap boundary
        gap_idx = gap['curr_index']  # First record after gap
        gap_timestamp_str = gap['curr_timestamp']
        gap_timestamp = parse_timestamp(gap_timestamp_str, format_str)
        
        if gap_timestamp is None:
            continue
        
        # Mark rows within window_seconds before and after the gap
        for i, ts_str in enumerate(timestamps):
            ts = parse_timestamp(ts_str, format_str)
            if ts is None:
                continue
            
            time_to_gap = abs((ts - gap_timestamp).total_seconds())
            
            # Include this row if it's within window_seconds of the gap
            if time_to_gap <= window_seconds:
                near_gap_indices.add(i)
    
    return near_gap_indices
