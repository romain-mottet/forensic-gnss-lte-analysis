import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict

from src.config.parameters import RESULTS_DIR, DATA_DIR


def extract_session_number_from_source(source_file: str) -> int:
    """    
    Args:
        source_file: Source file name (e.g., 'comparison_flag_ixelle_2.csv')
    
    Returns:
        int: Session number, or NaN if not found
    """
    match = re.search(r'_(\d+)\.csv$', source_file)
    if match:
        return int(match.group(1))
    return np.nan


def get_cloud_coverage_for_session(
    location: str,
    session_num: int,
    waypoints_df: pd.DataFrame
) -> float:
    """
    
    Args:
        location: Location name (e.g., 'ixelle', 'lln', 'waha')
        session_num: Session number extracted from source file
        waypoints_df: DataFrame containing waypoint records
    
    Returns:
        float: Cloud coverage value, or NaN if not found
    """
    if pd.isna(session_num):
        return np.nan
    
    # Filter waypoints for this location and session
    session_wp = waypoints_df[
        (waypoints_df['location'] == location) &
        (waypoints_df['session_number'] == session_num)
    ]
    
    if len(session_wp) > 0:
        # Get cloud coverage from first waypoint (consistent for whole session)
        cc = session_wp['cloud_coverage'].iloc[0]
        return cc if pd.notna(cc) else np.nan
    
    return np.nan


def run_step_6_create_unified_dataset(output_file: Path = None) -> bool:
    """    
    Args:
        output_file: Output path for unified dataset (default: results/unified_gps_dataset.csv)
    
    Returns:
        bool: True if successful, False otherwise
    """
    results_dir = RESULTS_DIR
    data_dir = DATA_DIR
    waypoints_file = data_dir / 'ground_truth_waypoints.csv'
    
    if output_file is None:
        output_file = results_dir / 'unified_gps_dataset.csv'
    
    try:
        # Step 1: Read waypoints to get cloud coverage by session
        print("📂 Reading waypoints...")
        print(f"  Looking for: {waypoints_file}")
        
        if not waypoints_file.exists():
            print(f"❌ Waypoints file not found: {waypoints_file}")
            print(f"\n📋 Debug info:")
            print(f"  RESULTS_DIR: {results_dir}")
            print(f"  data_dir: {data_dir}")
            print(f"  Checking if data_dir exists: {data_dir.exists()}")
            if data_dir.exists():
                print(f"  Contents of {data_dir}:")
                for item in data_dir.iterdir():
                    print(f"    - {item.name}")
            return False
        
        waypoints_df = pd.read_csv(waypoints_file)
        print(f"  ✓ Loaded {len(waypoints_df)} waypoint records")
        
        # Step 2: Read all comparison_flag_*.csv files from location subdirectories
        all_data = []
        total_files = 0
        
        print("\n📁 Reading comparison files...")
        
        # Iterate through location subdirectories (ixelle, lln, waha, etc.)
        for location_dir in sorted(results_dir.iterdir()):
            if not location_dir.is_dir():
                continue
            
            location_name = location_dir.name
            
            # Find all flagged comparison files in this location
            comparison_files = sorted(location_dir.glob('comparison_flag_*.csv'))
            
            if comparison_files:
                print(f"  📁 {location_name}: Found {len(comparison_files)} file(s)")
            
            # Step 3: Read and process each comparison file
            for comp_file in comparison_files:
                try:
                    df = pd.read_csv(comp_file)
                    
                    # Add metadata columns
                    df['source_file'] = comp_file.name
                    df['location'] = location_name
                    
                    # Extract session number from filename
                    session_num = extract_session_number_from_source(comp_file.name)
                    df['session_number'] = session_num
                    
                    all_data.append(df)
                    total_files += 1
                    print(f"    ✓ Loaded {comp_file.name} ({len(df)} records)")
                except Exception as e:
                    print(f"    ⚠️  Could not read {comp_file.name}: {e}")
                    continue
        
        if not all_data:
            print(f"❌ No comparison files found in {results_dir}")
            return False
        
        print(f"\n✅ Found {total_files} total comparison file(s)")
        
        # Combine all data
        print("\n📦 Combining all files into unified dataset...")
        unified_df = pd.concat(all_data, ignore_index=True)
        print(f"  • Total combined records: {len(unified_df):,}")
        
        # Step 4: AUTO-POPULATE cloud_coverage for each row from waypoints
        print("\n📝 AUTO-POPULATING cloud_coverage...")
        
        rows_needing_coverage = unified_df['cloud_coverage'].isna().sum() if 'cloud_coverage' in unified_df.columns else len(unified_df)
        print(f"  • Records needing cloud_coverage: {rows_needing_coverage:,}")
        
        # Initialize cloud_coverage column if it doesn't exist
        if 'cloud_coverage' not in unified_df.columns:
            unified_df['cloud_coverage'] = np.nan
        
        # Populate cloud coverage for each row
        for idx, row in unified_df.iterrows():
            if pd.isna(row.get('cloud_coverage')):
                location = row.get('location')
                session_num = row.get('session_number')
                
                if pd.notna(location) and pd.notna(session_num):
                    cloud_cov = get_cloud_coverage_for_session(location, session_num, waypoints_df)
                    unified_df.at[idx, 'cloud_coverage'] = cloud_cov
        
        coverage_populated = unified_df['cloud_coverage'].notna().sum()
        print(f"  ✓ Cloud coverage populated for {coverage_populated:,} / {len(unified_df):,} records")
        
        # Step 5: Save unified dataset
        output_file.parent.mkdir(parents=True, exist_ok=True)
        unified_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✅ UNIFIED DATASET CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"📁 Output: {output_file}")
        print(f"📊 SUMMARY:")
        print(f"  • Total records: {len(unified_df):,}")
        print(f"  • Records with cloud_coverage: {coverage_populated:,} / {len(unified_df):,}")
        print(f"  • Locations: {unified_df['location'].nunique()}")
        print(f"  • Sessions: {unified_df['session_number'].nunique()}")
        print(f"{'='*70}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error creating unified dataset: {e}")
        import traceback
        traceback.print_exc()
        return False