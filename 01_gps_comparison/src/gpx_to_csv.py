import gpxpy
import csv
from pathlib import Path

from src.config.parameters import TIMESTAMP_FORMAT_PARAM


def convert_gpx_to_csv(gpx_file_path, output_csv_path=None):
    """
    Convert a single GPX file to ground truth CSV format.
    
    IMPORTANT: Accuracy is ONLY written if found in GPX file.
    No fallbacks or multipliers - if accuracy missing, cell is empty.
    
    Args:
        gpx_file_path (str or Path): Path to the GPX file
        output_csv_path (str or Path, optional): Path for output CSV.
                                                  If None, uses gps_ground_truth_{stem}.csv
    
    Returns:
        dict: Results dictionary with keys:
            - 'success' (bool): Whether conversion succeeded
            - 'input_file' (str): Input file path
            - 'output_file' (str): Output file path
            - 'trackpoints' (int): Number of points converted
            - 'points_with_accuracy' (int): Points with actual accuracy data
            - 'error' (str): Error message if failed
    """
    
    gpx_path = Path(gpx_file_path)
    
    # Validate input file exists
    if not gpx_path.exists():
        return {
            'success': False,
            'input_file': str(gpx_path),
            'output_file': None,
            'trackpoints': 0,
            'points_with_accuracy': 0,
            'error': f'GPX file not found: {gpx_path}'
        }
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = gpx_path.parent / f"gps_ground_truth_{gpx_path.stem}.csv"
    else:
        output_csv_path = Path(output_csv_path)
    
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse GPX file
        with open(gpx_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        
        # Collect all trackpoints
        trackpoints = []
        
        # Extract from all tracks and segments
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    trackpoints.append(point)
        
        # Extract from routes if available
        for route in gpx.routes:
            for point in route.points:
                trackpoints.append(point)
        
        # Extract from waypoints if available
        for point in gpx.waypoints:
            trackpoints.append(point)
        
        if not trackpoints:
            return {
                'success': False,
                'input_file': str(gpx_path),
                'output_file': str(output_csv_path),
                'trackpoints': 0,
                'points_with_accuracy': 0,
                'error': 'No trackpoints found in GPX file'
            }
        
        # Write to CSV
        points_with_accuracy = 0
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['timestamp', 'gps_longitude', 'gps_latitude', 'gps_accuracy_m'])
            
            # Write data rows
            for point in trackpoints:
                # Convert timestamp from ISO 8601 to YYYY.MM.DD_HH.MM.SS
                timestamp = point.time
                if timestamp:
                    formatted_timestamp = timestamp.strftime(TIMESTAMP_FORMAT_PARAM)
                else:
                    formatted_timestamp = "UNKNOWN"
                
                longitude = point.longitude
                latitude = point.latitude
                
                # Extract accuracy from GPX file
                # ONLY if it exists - NO FALLBACKS, NO MULTIPLIERS
                accuracy = None
                
                # Try to get HDOP (most common in GPX files)
                if hasattr(point, 'horizontal_dilution') and point.horizontal_dilution:
                    accuracy = point.horizontal_dilution
                
                # Try to get VDOP if no HDOP
                elif hasattr(point, 'vertical_dilution') and point.vertical_dilution:
                    accuracy = point.vertical_dilution
                
                # Try extensions (some Garmin files include accuracy directly)
                if accuracy is None and hasattr(point, 'extensions') and point.extensions:
                    try:
                        if 'accuracy' in point.extensions:
                            accuracy = float(point.extensions['accuracy'])
                    except (ValueError, TypeError):
                        pass
                
                # Write accuracy as-is, or empty string if not found
                accuracy_value = str(accuracy) if accuracy is not None else ""
                
                if accuracy is not None:
                    points_with_accuracy += 1
                
                writer.writerow([formatted_timestamp, longitude, latitude, accuracy_value])
        
        return {
            'success': True,
            'input_file': str(gpx_path),
            'output_file': str(output_csv_path),
            'trackpoints': len(trackpoints),
            'points_with_accuracy': points_with_accuracy,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'input_file': str(gpx_path),
            'output_file': str(output_csv_path),
            'trackpoints': 0,
            'points_with_accuracy': 0,
            'error': f'{type(e).__name__}: {str(e)}'
        }


def find_all_gpx_files(root_directory):
    root_path = Path(root_directory)
    if not root_path.exists():
        return []
    
    return sorted(list(root_path.glob('**/*.gpx')))


def process_all_gpx_files(root_directory):
    """    
    Returns:
        dict: Results summary with keys:
            - 'total' (int): Total files found
            - 'successful' (int): Successfully converted
            - 'failed' (int): Failed conversions
            - 'results' (list): List of result dicts from convert_gpx_to_csv
    """
    gpx_files = find_all_gpx_files(root_directory)
    
    if not gpx_files:
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'results': []
        }
    
    results = []
    for gpx_file in gpx_files:
        result = convert_gpx_to_csv(gpx_file)
        results.append(result)
    
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    return {
        'total': len(gpx_files),
        'successful': successful,
        'failed': failed,
        'results': results
    }