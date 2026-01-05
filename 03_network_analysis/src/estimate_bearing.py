import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import csv
import pandas as pd


#Can be really upgraded, laking informations about azimuth so this part was a bit left like it is

def normalize_bearing(bearing: float) -> float:
    return bearing % 360.0


def calculate_bearing_confidence(
    rsrq_db: Optional[float],
    rssi_dbm: Optional[float] = None,
    accuracy_m: Optional[float] = None,
    ta_slots: Optional[float] = None,
    speed_kmh: Optional[float] = None
) -> Tuple[str, int]:

    score = 0

    if rsrq_db is None:
        rsrq_db = 7.0  # Default to MEDIUM if missing
    
    try:
        rsrq_db = float(rsrq_db)
    except (ValueError, TypeError):
        rsrq_db = 7.0
    
    if rsrq_db >= 10:
        score += 50  # Excellent signal quality
    elif rsrq_db >= 5:
        score += 30  # Good signal quality
    elif rsrq_db >= -5:
        score += 20  # Acceptable signal quality
    else:
        score += 10  # Poor signal quality (base points)
    
    # ============================================================================
    # FACTOR 2: RSSI (Signal Strength Boost) - +15 points maximum
    # ============================================================================
    if rssi_dbm is not None:
        try:
            rssi_dbm = float(rssi_dbm)
            if rssi_dbm > -70:
                score += 15  # Strong signal - significant boost
            elif rssi_dbm > -80:
                score += 7   # Moderate signal - partial boost
        except (ValueError, TypeError):
            pass
    
    # ============================================================================
    # FACTOR 3: GPS Accuracy (Position Precision Boost) - +15 points maximum
    # ============================================================================
    if accuracy_m is not None:
        try:
            accuracy_m = float(accuracy_m)
            if accuracy_m <= 5:
                score += 15  # Excellent GPS precision
            elif accuracy_m <= 10:
                score += 10  # Good GPS precision
            elif accuracy_m <= 20:
                score += 5   # Fair GPS precision
        except (ValueError, TypeError):
            pass
    
    # ============================================================================
    # FACTOR 4: TA - Timing Advance (Tower Proximity Boost) - +10 points maximum
    # ============================================================================
    if ta_slots is not None:
        try:
            ta_slots = float(ta_slots)
            if ta_slots <= 2:
                score += 10  # Very close to tower - high clarity
            elif ta_slots <= 5:
                score += 5   # Close to tower - moderate clarity
        except (ValueError, TypeError):
            pass
    
    # ============================================================================
    # FACTOR 5: Speed (Movement Stability Boost) - +10 points maximum
    # ============================================================================
    if speed_kmh is not None:
        try:
            speed_kmh = float(speed_kmh)
            if speed_kmh <= 5:
                score += 10  # Steady/stationary - high stability
            elif speed_kmh <= 15:
                score += 5   # Moderate speed - moderate stability
        except (ValueError, TypeError):
            pass
    
    # ============================================================================
    # SCORE TO CONFIDENCE MAPPING
    # ============================================================================
    if score >= 60:
        return "HIGH", 30    # 30° uncertainty - very reliable
    elif score >= 40:
        return "MEDIUM", 60  # 60° uncertainty - reasonably reliable
    else:
        return "LOW", 90     # 90° uncertainty - poor reliability


def calculate_bearing_range(azimuth: float, uncertainty: int) -> Tuple[float, float]:

    bearing_min = normalize_bearing(azimuth - uncertainty)
    bearing_max = normalize_bearing(azimuth + uncertainty)
    return bearing_min, bearing_max


def load_towers_db(towers_path: Path) -> Dict[str, Dict]:
    if not towers_path.is_file():
        return {}
    
    try:
        with towers_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_signal_lookup(signal_data_path: Path) -> Dict[Tuple[str, str], Dict]:
    lookup = {}
    
    try:
        with signal_data_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("timestamp", "").strip()
                cell_id = row.get("cell_id", "").strip()
                signal_type = row.get("signal_type", "").upper()
                
                if not ts or not ts.startswith("20"):
                    continue
                
                if (ts, cell_id) not in lookup:
                    lookup[(ts, cell_id)] = {}
                
                try:
                    value = float(row.get("value", 0))
                    
                    if signal_type == "RSRQ":
                        lookup[(ts, cell_id)]['rsrq'] = value
                    elif signal_type == "RSSI":
                        lookup[(ts, cell_id)]['rssi'] = value
                    elif signal_type == "TA":
                        lookup[(ts, cell_id)]['ta'] = value
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    
    return lookup


def build_metadata_lookup(signal_metadata_path: Path) -> Dict[Tuple[str, str], Dict]:
    lookup = {}
    
    try:
        with signal_metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("timestamp", "").strip()
                cell_id = row.get("cell_id", "").strip()
                
                if not ts or not ts.startswith("20"):
                    continue
                
                if (ts, cell_id) not in lookup:
                    lookup[(ts, cell_id)] = {}
                
                try:
                    if "accuracy_m" in row and row.get("accuracy_m"):
                        lookup[(ts, cell_id)]['accuracy_m'] = float(row["accuracy_m"])
                    if "speed_kmh" in row and row.get("speed_kmh"):
                        lookup[(ts, cell_id)]['speed_kmh'] = float(row["speed_kmh"])
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    
    return lookup


def estimate_bearings(
    distance_estimates_path: Path,
    signal_data_path: Path,
    signal_metadata_path: Path,
    towers_path: Path,
    pci_path: Path,
    output_path: Path,
    excel_safe: bool = True
) -> None:
    
    # Load reference databases
    towers_db = load_towers_db(towers_path)
    pci_db = load_towers_db(pci_path)
    signal_lookup = build_signal_lookup(signal_data_path)
    metadata_lookup = build_metadata_lookup(signal_metadata_path)
    
    # Collect records (list of dicts for pandas)
    records: List[Dict] = []
    
    try:
        with distance_estimates_path.open("r", encoding="utf-8") as in_f:
            reader = csv.DictReader(in_f)
            for row in reader:
                ts = row.get("timestamp", "").strip()
                cell_id = row.get("cell_id", "").strip()
                cell_type = row.get("cell_type", "serving").strip()
                
                # Validate timestamp
                if not ts or not ts.startswith("20"):
                    continue
                
                # Try to get tower from towers.json first, then pci.json
                tower = towers_db.get(str(cell_id))
                if tower is None:
                    tower = pci_db.get(str(cell_id))
                
                if tower is None:
                    # Tower not found - mark invalid
                    records.append({
                        "timestamp": ts,
                        "cell_id": cell_id,
                        "cell_type": cell_type,
                        "azimuth_degrees": None,
                        "rsrq_db": None,
                        "rssi_dbm": None,
                        "accuracy_m": None,
                        "ta_slots": None,
                        "speed_kmh": None,
                        "bearing_degrees": None,
                        "bearing_min": None,
                        "bearing_max": None,
                        "uncertainty_degrees": None,
                        "confidence_level": None,
                        "valid": False
                    })
                    continue
                
                # Get azimuth from tower
                try:
                    azimuth = float(tower.get("azimuth", 0))
                except (ValueError, TypeError):
                    azimuth = 0.0
                azimuth = normalize_bearing(azimuth)
                
                # Get all signal quality factors
                signal_data = signal_lookup.get((ts, cell_id), {})
                metadata = metadata_lookup.get((ts, cell_id), {})
                
                rsrq = signal_data.get('rsrq', None)
                rssi = signal_data.get('rssi', None)
                ta = signal_data.get('ta', None)
                accuracy = metadata.get('accuracy_m', None)
                speed = metadata.get('speed_kmh', None)
                
                # Calculate multi-factor confidence
                confidence_level, uncertainty = calculate_bearing_confidence(
                    rsrq_db=rsrq,
                    rssi_dbm=rssi,
                    accuracy_m=accuracy,
                    ta_slots=ta,
                    speed_kmh=speed
                )
                
                # Calculate bearing range with 360° wrapping
                bearing_min, bearing_max = calculate_bearing_range(azimuth, uncertainty)
                
                # Bearing is the azimuth (central estimate)
                bearing = azimuth
                
                # Append record (store as-is, pandas handles rounding)
                records.append({
                    "timestamp": ts,
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                    "azimuth_degrees": azimuth,
                    "rsrq_db": rsrq,
                    "rssi_dbm": rssi,
                    "accuracy_m": accuracy,
                    "ta_slots": ta,
                    "speed_kmh": speed,
                    "bearing_degrees": bearing,
                    "bearing_min": bearing_min,
                    "bearing_max": bearing_max,
                    "uncertainty_degrees": uncertainty,
                    "confidence_level": confidence_level,
                    "valid": True
                })
    
    except Exception as e:
        print(f"Warning: Error reading distance estimates: {e}")
        pass
    
    # Create DataFrame from records
    df = pd.DataFrame.from_records(records)
    
    # Define column order
    col_order = [
        "timestamp", "cell_id", "cell_type",
        "azimuth_degrees", "rsrq_db", "rssi_dbm",
        "accuracy_m", "ta_slots", "speed_kmh",
        "bearing_degrees", "bearing_min", "bearing_max",
        "uncertainty_degrees", "confidence_level", "valid",
    ]
    df = df.reindex(columns=col_order)
    
    # Float columns that need decimal formatting
    float_cols = [
        "azimuth_degrees", "rsrq_db", "rssi_dbm", "accuracy_m",
        "ta_slots", "speed_kmh", "bearing_degrees",
        "bearing_min", "bearing_max",
    ]
    
    # Excel-safe formatting: render floats as text with .1f (keeps 4.0, 5.0, 6.0 visible)
    if excel_safe:
        def format_float(v):
            """Format value as text with 1 decimal place, or empty string if NaN."""
            return "" if pd.isna(v) else f"{float(v):.1f}"
        
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].apply(format_float)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    if excel_safe:
        # Already formatted as text; write as-is
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        # Keep numeric types; let pandas format with %.1f at write time
        df.to_csv(output_path, index=False, encoding="utf-8", float_format="%.1f")
    
    print(f"✓ Bearing estimates saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    base_path = Path("data/lln_1_data")
    
    estimate_bearings(
        distance_estimates_path=base_path / "distance_estimates.csv",
        signal_data_path=base_path / "signal_data.csv",
        signal_metadata_path=base_path / "signal_metadata.csv",
        towers_path=Path("data/tower_data/towers.json"),
        pci_path=Path("data/tower_data/pci.json"),
        output_path=base_path / "bearing_estimates.csv",
        excel_safe=True 
    )
    
    print("✓ Bearing estimates with multi-factor confidence saved to bearing_estimates.csv")