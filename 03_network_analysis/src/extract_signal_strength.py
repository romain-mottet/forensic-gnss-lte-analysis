import pandas as pd
import numpy as np
from pathlib import Path
import sys


def extract_signal_strength(
    parsed_log_file: str,
    output_dir: str,
    base_name: str,
    context: str,
    verbose: bool = False
) -> dict:
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # File suffix
        file_suffix = f"_{base_name}_{context}"
        
        # Load data
        df = pd.read_csv(parsed_log_file, encoding='utf-8')
        measurements = len(df)
        
        # =====================================================================
        # PART 1: EXTRACT CORE SIGNAL DATA (RSRP, RSSI, RSRQ, TA)
        # =====================================================================
        
        signal_records = []
        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            cell_id = str(int(row['RAWCELLID']))
            
            if pd.notna(row['Level']):
                signal_records.append({
                    'timestamp': timestamp,
                    'cell_id': cell_id,
                    'signal_type': 'RSRP',
                    'value': int(row['Level']),
                    'unit': 'dBm'
                })
            
            if 'LTERSSI' in df.columns and pd.notna(row['LTERSSI']):
                signal_records.append({
                    'timestamp': timestamp,
                    'cell_id': cell_id,
                    'signal_type': 'RSSI',
                    'value': int(row['LTERSSI']),
                    'unit': 'dBm'
                })
            
            if 'Qual' in df.columns and pd.notna(row['Qual']):
                signal_records.append({
                    'timestamp': timestamp,
                    'cell_id': cell_id,
                    'signal_type': 'RSRQ',
                    'value': int(row['Qual']),
                    'unit': 'dB'
                })
            
            if 'TA' in df.columns and pd.notna(row['TA']):
                signal_records.append({
                    'timestamp': timestamp,
                    'cell_id': cell_id,
                    'signal_type': 'TA',
                    'value': int(row['TA']),
                    'unit': 'time_slots'
                })
        
        signal_df = pd.DataFrame(signal_records)
        signal_file = f"{output_dir}/signal_data{file_suffix}.csv"
        signal_df.to_csv(signal_file, index=False)
        
        # =====================================================================
        # PART 2: EXTRACT SIGNAL METADATA (GPS, frequency, tower info)
        # =====================================================================
        
        metadata_records = []
        for idx, row in df.iterrows():
            try:
                metadata_records.append({
                    'timestamp': row['Timestamp'],
                    'cell_id': str(int(row['RAWCELLID'])),
                    'lac': int(row.get('LAC')) if pd.notna(row.get('LAC')) else np.nan,
                    'node': int(row.get('Node')) if pd.notna(row.get('Node')) else np.nan,
                    'arfcn': int(row.get('ARFCN')) if pd.notna(row.get('ARFCN')) else np.nan,
                    'psc': int(row.get('PSC')) if pd.notna(row.get('PSC')) else np.nan,
                    'longitude': row.get('Longitude') if pd.notna(row.get('Longitude')) else np.nan,
                    'latitude': row.get('Latitude') if pd.notna(row.get('Latitude')) else np.nan,
                    'accuracy_m': row.get('Accuracy') if pd.notna(row.get('Accuracy')) else np.nan,
                    'speed_kmh': row.get('Speed') if pd.notna(row.get('Speed')) else np.nan,
                    'bearing_deg': row.get('Bearing') if pd.notna(row.get('Bearing')) else np.nan
                })
            except Exception as e:
                if verbose:
                    print(f"  Warning: Row {idx} skipped ({e})")
                continue
        
        metadata_df = pd.DataFrame(metadata_records)
        metadata_file = f"{output_dir}/signal_metadata{file_suffix}.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        # =====================================================================
        # PART 3: EXTRACT NEIGHBOR CELL DATA (for triangulation)
        # =====================================================================
        
        neighbor_records = []
        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            cell_id = str(int(row['RAWCELLID']))
            
            # Use NCell1-18 (correct column names)
            for i in range(1, 19):
                ncell_col = f'NCell{i}'
                nsignal_col = f'NRxLev{i}'
                ntech_col = f'NTech{i}'
                
                if ncell_col in df.columns and pd.notna(row[ncell_col]):
                    try:
                        neighbor_records.append({
                            'timestamp': timestamp,
                            'cell_id': cell_id,
                            'neighbor_index': i,
                            'neighbor_cell_id': str(int(row[ncell_col])),
                            'neighbor_tech': row.get(ntech_col, '4G') if ntech_col in df.columns and pd.notna(row.get(ntech_col)) else '4G',
                            'neighbor_rsrp': int(row[nsignal_col]) if nsignal_col in df.columns and pd.notna(row[nsignal_col]) else -120
                        })
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Neighbor row {idx}, index {i} skipped ({e})")
                        continue
        
        if len(neighbor_records) > 0:
            neighbor_df = pd.DataFrame(neighbor_records)
        else:
            neighbor_df = pd.DataFrame(columns=['timestamp', 'cell_id', 'neighbor_index',
                                              'neighbor_cell_id', 'neighbor_tech', 'neighbor_rsrp'])
        
        neighbor_file = f"{output_dir}/neighbor_signals{file_suffix}.csv"
        neighbor_df.to_csv(neighbor_file, index=False)
        
        # =====================================================================
        # PART 4: EXTRACT GPS GROUND TRUTH (for validation)
        # =====================================================================
        
        gps_records = []
        gps_valid_count = 0
        for idx, row in df.iterrows():
            has_gps = pd.notna(row.get('Longitude')) and pd.notna(row.get('Latitude'))
            
            gps_records.append({
                'timestamp': row['Timestamp'],
                'gps_longitude': row.get('Longitude') if pd.notna(row.get('Longitude')) else np.nan,
                'gps_latitude': row.get('Latitude') if pd.notna(row.get('Latitude')) else np.nan,
                'gps_accuracy_m': row.get('Accuracy') if pd.notna(row.get('Accuracy')) else np.nan,
                'base_name': base_name,
                'context': context
            })
            
            if has_gps:
                gps_valid_count += 1
        
        gps_df = pd.DataFrame(gps_records)
        gps_file = f"{output_dir}/ground_truth{file_suffix}.csv"
        gps_df.to_csv(gps_file, index=False)
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        
        signal_count = len(signal_df)
        metadata_count = len(metadata_df)
        neighbor_count = len(neighbor_df)
        gps_count = len(gps_df)
        unique_neighbors = neighbor_df['neighbor_cell_id'].nunique() if len(neighbor_df) > 0 else 0
        gps_coverage = 100 * gps_valid_count / measurements if measurements > 0 else 0
        
        return {
            'success': True,
            'signal_records': signal_count,
            'metadata_records': metadata_count,
            'neighbor_records': neighbor_count,
            'unique_neighbors': unique_neighbors,
            'gps_valid': gps_valid_count,
            'gps_coverage': gps_coverage,
            'output_dir': output_dir,
            'ground_truth_file': gps_file,
            'file_suffix': file_suffix
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    print("="*70)
    print("STEP 2.1: EXTRACT SIGNAL STRENGTH DATA")
    print("="*70)
    print()
    
    if len(sys.argv) < 4:
        print("Usage: python extract_signal_strength.py <parsed_file> <output_dir> <base_name> [context]")
        print("Example: python extract_signal_strength.py parsed_ixelle_4.csv ixelle_4_data/ ixelle_4 town")
        sys.exit(1)
    
    parsed_file = sys.argv[1]
    output_dir = sys.argv[2]
    base_name = sys.argv[3]
    context = sys.argv[4] if len(sys.argv) > 4 else 'default'
    
    result = extract_signal_strength(parsed_file, output_dir, base_name, context, verbose=True)
    
    print("\n" + "="*70)
    if result.get('success'):
        print("✓ STEP 2.1 COMPLETE")
        print(f"  Signal records: {result['signal_records']}")
        print(f"  Metadata: {result['metadata_records']}")
        print(f"  Neighbors: {result['neighbor_records']} ({result['unique_neighbors']} unique)")
        print(f"  GPS coverage: {result['gps_coverage']:.0f}%")
        print(f"  File suffix: {result.get('file_suffix', 'N/A')}")
        print(f"  Ground truth file: {result.get('ground_truth_file', 'N/A')}")
    else:
        print("✗ STEP 2.1 FAILED")
        print(f"  Error: {result.get('error', 'Unknown error')}")
    print("="*70)
    
    return 0 if result.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())
