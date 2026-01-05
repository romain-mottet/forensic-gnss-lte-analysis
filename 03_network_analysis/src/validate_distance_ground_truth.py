import csv
import json
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def load_pci_db(pci_path: Path) -> Dict:
    if not pci_path.is_file():
        return {}
    try:
        with pci_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_towers_db(towers_path: Path) -> Dict:
    if not towers_path.is_file():
        return {}
    try:
        with towers_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_tower_location(cell_id: str, pci_db: Dict, towers_db: Dict) -> Tuple[float, float]:
    # Try pci.json first
    if str(cell_id) in pci_db:
        pci_record = pci_db[str(cell_id)]
        try:
            lat = float(pci_record.get("latitude", float("nan")))
            lon = float(pci_record.get("longitude", float("nan")))
            if not (math.isnan(lat) or math.isnan(lon)):
                return lat, lon
        except (ValueError, TypeError):
            pass
    
    # Fallback to towers.json
    if str(cell_id) in towers_db:
        tower_record = towers_db[str(cell_id)]
        try:
            lat = float(tower_record.get("latitude", float("nan")))
            lon = float(tower_record.get("longitude", float("nan")))
            if not (math.isnan(lat) or math.isnan(lon)):
                return lat, lon
        except (ValueError, TypeError):
            pass
    
    return float("nan"), float("nan")


def load_ground_truth(ground_truth_path: Path) -> Dict[str, Tuple[float, float]]:
    gt_data = {}
    try:
        with ground_truth_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row.get("timestamp", "").strip()
                if timestamp.startswith("20"):
                    try:
                        lon = float(row.get("gps_longitude", 0))
                        lat = float(row.get("gps_latitude", 0))
                        gt_data[timestamp] = (lat, lon)
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        print(f"Warning: Error loading ground truth: {e}")
    
    return gt_data


def compute_per_cell_metrics(
    distance_estimates_path: Path,
    ground_truth_path: Path,
    pci_path: Path,
    towers_path: Path
) -> Tuple[List[float], Dict]:
    # Load databases
    pci_db = load_pci_db(pci_path)
    towers_db = load_towers_db(towers_path)
    gt_positions = load_ground_truth(ground_truth_path)
    
    all_errors = []
    cell_errors = defaultdict(list)
    cell_info = defaultdict(lambda: {'frequency_mhz': 0, 'method': 'unknown', 'confidence': []})
    
    # Load distance estimates and compute errors
    try:
        with distance_estimates_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row.get("timestamp", "").strip()
                if not timestamp.startswith("20"):
                    continue
                
                try:
                    distance_m = float(row.get("distance_m", 0))
                    cell_id = row.get("cell_id", "")
                    tower_lat = float(row.get("tower_lat", float("nan")))
                    tower_lon = float(row.get("tower_lon", float("nan")))
                    frequency_mhz = float(row.get("frequency_mhz", 0))
                    distance_method = row.get("distance_method", "unknown")
                    confidence_level = row.get("confidence_level", "unknown")
                    
                    # Skip if no valid tower location
                    if math.isnan(tower_lat) or math.isnan(tower_lon):
                        continue
                    
                    # Find matching ground truth for this timestamp
                    gt_lat, gt_lon = None, None
                    for gt_ts, (gt_lat_, gt_lon_) in gt_positions.items():
                        if gt_ts.startswith(timestamp.split("_")[0]):  # Match date prefix
                            gt_lat, gt_lon = gt_lat_, gt_lon_
                            break
                    
                    if gt_lat is None or gt_lon is None:
                        continue
                    
                    # Calculate actual distance
                    actual_distance = haversine(gt_lat, gt_lon, tower_lat, tower_lon)
                    
                    # Calculate error
                    error_m = abs(distance_m - actual_distance)
                    
                    all_errors.append(error_m)
                    cell_errors[cell_id].append(error_m)
                    
                    # Store cell metadata
                    cell_info[cell_id]['frequency_mhz'] = frequency_mhz
                    cell_info[cell_id]['method'] = distance_method
                    cell_info[cell_id]['confidence'].append(confidence_level)
                    
                except (ValueError, TypeError):
                    continue
    
    except Exception as e:
        print(f"Error processing distance estimates: {e}")
        return [], {}
    
    # Compute per-cell metrics
    cell_metrics = {}
    for cell_id, errors in cell_errors.items():
        if not errors:
            continue
        
        errors_array = pd.Series(errors)
        info = cell_info[cell_id]
        
        # Count confidence levels
        confidence_counts = pd.Series(info['confidence']).value_counts()
        primary_confidence = confidence_counts.index[0] if len(confidence_counts) > 0 else "unknown"
        
        cell_metrics[cell_id] = {
            'sample_count': len(errors),
            'mae': float(errors_array.mean()),
            'rmse': float(math.sqrt((errors_array ** 2).mean())),
            'cep': float(errors_array.quantile(0.50)),
            'r95': float(errors_array.quantile(0.95)),
            'mean_error': float(errors_array.mean()),
            'std_error': float(errors_array.std()),
            'min_error': float(errors_array.min()),
            'max_error': float(errors_array.max()),
            'frequency_mhz': int(info['frequency_mhz']),
            'distance_method': info['method'],
            'primary_confidence': primary_confidence,
        }
    
    return all_errors, cell_metrics


def save_csv_report(cell_metrics: Dict, output_csv_path: Path, context: str) -> None:
    if not cell_metrics:
        print(f"Warning: No cell metrics to save to CSV")
        return
    
    # Prepare CSV data
    csv_rows = []
    for cell_id, metrics in sorted(cell_metrics.items()):
        row = {
            'cell_id': cell_id,
            'sample_count': metrics['sample_count'],
            'mae_m': round(metrics['mae'], 1),
            'rmse_m': round(metrics['rmse'], 1),
            'cep_m': round(metrics['cep'], 1),
            'r95_m': round(metrics['r95'], 1),
            'mean_error_m': round(metrics['mean_error'], 1),
            'std_error_m': round(metrics['std_error'], 1) if not math.isnan(metrics['std_error']) else '',
            'min_error_m': round(metrics['min_error'], 1),
            'max_error_m': round(metrics['max_error'], 1),
            'frequency_mhz': metrics['frequency_mhz'],
            'distance_method': metrics['distance_method'],
            'primary_confidence': metrics['primary_confidence'],
            'context': context  # NEW COLUMN!
        }
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"✓ Saved per-cell metrics CSV with context: {output_csv_path}")


def save_txt_report(
    all_errors: List[float],
    cell_metrics: Dict,
    output_txt_path: Path,
    distance_estimates_path: Path,
    ground_truth_path: Path,
    context: str
) -> None:
    if not all_errors:
        print(f"Warning: No errors computed for text report")
        return
    
    errors = sorted(all_errors)
    n = len(errors)
    mae = sum(errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    cep = errors[int(n * 0.50)] if n > 0 else 0
    r95 = errors[int(n * 0.95)] if n > 0 else 0
    mean_error = sum(errors) / n
    std_error = math.sqrt(sum((e - mae) ** 2 for e in errors) / n)
    
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 2.2.5: DISTANCE ESTIMATION GROUND TRUTH VALIDATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Context: {context}\n\n")
        
        f.write("OVERALL STATISTICS (Aggregated across all cells)\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"SAMPLES: {len(errors)} distance measurements compared to ground truth\n\n")
        
        f.write("ERROR METRICS:\n")
        f.write(f"  MAE (mean absolute error):     {mae:7.1f} m\n")
        f.write(f"  RMSE (root mean square error):  {rmse:7.1f} m\n")
        f.write(f"  CEP (50th percentile):          {cep:7.1f} m\n")
        f.write(f"  R95 (95th percentile):          {r95:7.1f} m\n")
        f.write(f"  Mean error (bias):              {mean_error:+7.1f} m\n\n")
        
        f.write("ERROR DISTRIBUTION:\n")
        f.write(f"  Min error:                      {min(errors):7.1f} m\n")
        f.write(f"  Max error:                      {max(errors):7.1f} m\n")
        f.write(f"  Std dev:                        {std_error:7.1f} m\n\n")
        
        # Per-cell breakdown
        if cell_metrics:
            f.write("-" * 80 + "\n")
            f.write("PER-CELL ACCURACY BREAKDOWN\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort by RMSE (worst first)
            sorted_cells = sorted(cell_metrics.items(), key=lambda x: x[1]['rmse'], reverse=True)
            
            f.write(f"{'Cell ID':<15} {'Samples':<10} {'MAE (m)':<10} {'RMSE (m)':<10} {'Method':<12} {'Confidence':<12}\n")
            f.write(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}\n")
            
            for cell_id, metrics in sorted_cells:
                f.write(
                    f"{cell_id:<15} {metrics['sample_count']:<10} {metrics['mae']:<10.1f} "
                    f"{metrics['rmse']:<10.1f} {metrics['distance_method']:<12} {metrics['primary_confidence']:<12}\n"
                )
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("INPUT FILES:\n")
        f.write(f"  Distance estimates: {distance_estimates_path.name}\n")
        f.write(f"  Ground truth: {ground_truth_path.name}\n")
        f.write(f"  Tower databases: pci.json, towers.json\n")
        f.write(f"  Context: {context}\n\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved text report: {output_txt_path}")


def compute_distance_accuracy(
    distance_estimates_path: Path,
    ground_truth_path: Path,
    pci_path: Path,
    towers_path: Path,
    output_path: Path,
    context: str = "default"
) -> Dict:
    
    print("\n" + "=" * 70)
    print("PHASE 2.2.5: DISTANCE ESTIMATION GROUND TRUTH VALIDATION")
    print(f"Context: {context}")
    print("=" * 70)
    
    # Compute per-cell metrics
    print("\n[1/3] Computing per-cell accuracy metrics...")
    all_errors, cell_metrics = compute_per_cell_metrics(
        distance_estimates_path,
        ground_truth_path,
        pci_path,
        towers_path
    )
    
    if not all_errors:
        print("  No matching distance/ground truth pairs found!")
        return {"samples": 0, "mae": 0, "rmse": 0, "cep": 0, "r95": 0, "mean_error": 0}
    
    print(f"✓ Computed metrics for {len(cell_metrics)} cells")
    
    # Save CSV report (machine-readable with context)
    print("[2/3] Saving machine-readable CSV report with context...")
    csv_output_path = Path(str(output_path).replace('.txt', '.csv'))
    save_csv_report(cell_metrics, csv_output_path, context)
    
    # Save TXT report (human-readable with context)
    print("[3/3] Saving human-readable text report with context...")
    save_txt_report(all_errors, cell_metrics, output_path, distance_estimates_path, ground_truth_path, context)
    
    # Compute overall metrics
    n = len(all_errors)
    all_errors_sorted = sorted(all_errors)
    mae = sum(all_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in all_errors) / n)
    cep = all_errors_sorted[int(n * 0.50)] if n > 0 else 0
    r95 = all_errors_sorted[int(n * 0.95)] if n > 0 else 0
    mean_error = sum(all_errors) / n
    
    return {
        "samples": n,
        "mae": mae,
        "rmse": rmse,
        "cep": cep,
        "r95": r95,
        "mean_error": mean_error
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 6:
        print("Usage: python validate_distance_ground_truth.py <distance_estimates.csv> <ground_truth.csv> <pci.json> <towers.json> <output.txt> [context]")
        sys.exit(1)
    
    distance_estimates_path = Path(sys.argv[1])
    ground_truth_path = Path(sys.argv[2])
    pci_path = Path(sys.argv[3])
    towers_path = Path(sys.argv[4])
    output_path = Path(sys.argv[5])
    context = sys.argv[6] if len(sys.argv) > 6 else "default"
    
    metrics = compute_distance_accuracy(
        distance_estimates_path,
        ground_truth_path,
        pci_path,
        towers_path,
        output_path,
        context=context
    )
    
    print("\n" + "=" * 70)
    print(" VALIDATION COMPLETE")
    print(f"   Context: {context}")
    print(f"   Samples: {metrics['samples']}")
    print(f"   MAE: {metrics['mae']:.1f}m")
    print(f"   RMSE: {metrics['rmse']:.1f}m")
    print(f"   CEP: {metrics['cep']:.1f}m")
    print(f"   R95: {metrics['r95']:.1f}m")
    print("=" * 70)