"""
Centralized functions for finding smartphone/watch file pairs and other patterns.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from src.config.parameters import SMARTPHONE_DIR, WATCH_DIR, LOCATION_NAMES

def discover_location_file_pairs() -> Dict[str, List[Tuple[Path, Path]]]:

    file_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    
    if not SMARTPHONE_DIR.exists() or not WATCH_DIR.exists():
        return file_pairs
    
    for location_name in LOCATION_NAMES:
        smartphone_location_dir = SMARTPHONE_DIR / location_name
        watch_location_dir = WATCH_DIR / location_name
        
        if not smartphone_location_dir.exists() or not watch_location_dir.exists():
            continue
            
        location_pairs = []
        smartphone_files = sorted(smartphone_location_dir.glob("ground_truth_*.csv"))
        
        for spfile in smartphone_files:
            # Extract location and session number from filename
            # Example: ground_truth_ixelle_2.csv -> location='ixelle', number='2'
            parts = spfile.stem.split("_")
            if len(parts) >= 3:
                location = parts[2]
                number = parts[3] if len(parts) >= 4 else "5"
            else:
                continue
                
            watchfile = watch_location_dir / f"gps_ground_truth_{location}_{number}.csv"
            
            if watchfile.exists():
                location_pairs.append((spfile, watchfile))
        
        if location_pairs:
            file_pairs[location_name] = location_pairs
    
    return file_pairs
