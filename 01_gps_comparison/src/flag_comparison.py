import csv
from pathlib import Path
from typing import Dict, List, Tuple

from src.config.parameters import (
    TIMESTAMP_FORMAT_PARAM,
    MAJOR_GAP_THRESHOLD_SECONDS_PARAM,
    NEAR_GAP_WINDOW_SECONDS_PARAM,
    COMPARISON_FLAG_PREFIX,
    RESULTS_DIR,
)
from src.helpers.flag_helper import (
    find_duplicate_timestamps,
    is_cache_duplicate,
    find_major_gaps,
    find_near_major_gap_indices,
)


def load_csv(file_path: Path) -> Tuple[List[Dict], List[str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    
    return rows, fieldnames


def add_flags_to_rows(
    rows: List[Dict],
    major_gap_threshold: int,
    near_gap_window: int,
    timestamp_format: str,
) -> List[Dict]:
    """
    Add quality flags to each row.
    
    Flags added:
    - is_cache_duplicate: True if duplicate timestamp with identical coordinates
    - near_major_gap_Xs: True if within X seconds of a gap >= major_gap_threshold
    
    Args:
        rows: List of comparison CSV rows
        major_gap_threshold: Gap size threshold in seconds (e.g., 60)
        near_gap_window: Window around gaps to flag in seconds (e.g., 30)
        timestamp_format: Timestamp format string
    
    Returns:
        Rows with added flag columns
    """
    
    # Step 1: Detect duplicate timestamps
    duplicates = find_duplicate_timestamps(rows)
    
    # Step 2: Detect major gaps
    timestamps = [row.get('timestamp', '') for row in rows]
    major_gaps = find_major_gaps(timestamps, timestamp_format, major_gap_threshold)
    
    # Step 3: Find indices near major gaps
    near_major_gap_indices = find_near_major_gap_indices(
        timestamps, timestamp_format, major_gaps, near_gap_window
    )
    
    # Step 4: Add flags to each row
    flagged_rows = []
    
    for i, row in enumerate(rows):
        # Create new row with original data
        flagged_row = dict(row)
        
        # Add is_cache_duplicate flag
        flagged_row['is_cache_duplicate'] = is_cache_duplicate(i, rows, duplicates)
        
        # Add near_major_gap_Xs flag (X = near_gap_window)
        gap_flag_name = f'near_major_gap_{near_gap_window}s'
        flagged_row[gap_flag_name] = i in near_major_gap_indices
        
        flagged_rows.append(flagged_row)
    
    return flagged_rows


def write_flagged_csv(
    output_path: Path,
    flagged_rows: List[Dict],
    original_fieldnames: List[str],
) -> None:
    
    if not flagged_rows:
        return
    
    # Build fieldnames: original + new flags
    new_flags = ['is_cache_duplicate', 'near_major_gap_30s']  # 30s is default
    fieldnames = original_fieldnames + new_flags
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)


def process_comparison_file(
    comparison_file: Path,
    output_base: Path,
    major_gap_threshold: int = MAJOR_GAP_THRESHOLD_SECONDS_PARAM,
    near_gap_window: int = NEAR_GAP_WINDOW_SECONDS_PARAM,
    timestamp_format: str = TIMESTAMP_FORMAT_PARAM,
) -> Dict:
    
    result = {
        'input_file': comparison_file.name,
        'output_file': None,
        'success': False,
        'total_rows': 0,
        'cache_duplicates': 0,
        'near_major_gap': 0,
        'error': None,
    }
    
    try:
        # Load CSV
        rows, fieldnames = load_csv(comparison_file)
        result['total_rows'] = len(rows)
        
        if not rows:
            result['error'] = "Empty CSV file"
            return result
        
        # Add flags
        flagged_rows = add_flags_to_rows(
            rows,
            major_gap_threshold,
            near_gap_window,
            timestamp_format,
        )
        
        # Count flags for reporting
        cache_dup_count = sum(1 for row in flagged_rows if row.get('is_cache_duplicate', False))
        near_gap_count = sum(1 for row in flagged_rows if row.get('near_major_gap_30s', False))
        
        result['cache_duplicates'] = cache_dup_count
        result['near_major_gap'] = near_gap_count
        
        # Determine output filename and path
        location = comparison_file.parent.name
        filename = comparison_file.name  # e.g., comparison_lln_5.csv
        
        # Insert "flag_" prefix: comparison_lln_5.csv → comparison_flag_lln_5.csv
        filename_parts = filename.split('_', 1)  # Split on first underscore
        if len(filename_parts) == 2:
            flagged_filename = f"{filename_parts[0]}_flag_{filename_parts[1]}"
        else:
            flagged_filename = f"{COMPARISON_FLAG_PREFIX}{filename}"
        
        output_file = output_base / location / flagged_filename
        
        # Write flagged CSV
        write_flagged_csv(output_file, flagged_rows, fieldnames)
        
        result['output_file'] = str(output_file)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_for_all_files(
    major_gap_threshold: int = MAJOR_GAP_THRESHOLD_SECONDS_PARAM,
    near_gap_window: int = NEAR_GAP_WINDOW_SECONDS_PARAM,
) -> Dict:
    
    summary = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'results': [],
    }
    
    if not RESULTS_DIR.exists():
        summary['error'] = f"Results directory not found: {RESULTS_DIR}"
        return summary
    
    # Process each location directory
    for location_dir in sorted(RESULTS_DIR.iterdir()):
        if not location_dir.is_dir():
            continue
        
        # Process each comparison_*.csv file
        for comparison_file in sorted(location_dir.glob("comparison_*.csv")):
            # Skip if this is already a flagged file
            if "_flag_" in comparison_file.name:
                continue
            
            result = process_comparison_file(
                comparison_file,
                RESULTS_DIR,
                major_gap_threshold,
                near_gap_window,
            )
            
            summary['total'] += 1
            if result['success']:
                summary['successful'] += 1
            else:
                summary['failed'] += 1
            
            summary['results'].append(result)
    
    return summary
