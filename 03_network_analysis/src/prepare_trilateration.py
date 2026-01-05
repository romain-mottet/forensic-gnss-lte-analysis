import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

try:
    from config.algorithm_params import AlgorithmParams, get_algorithm_params
except ImportError:
    AlgorithmParams = None
    get_algorithm_params = None


class TrilaterationInputPreparer:
    
    def __init__(
        self,
        input_file: str = "distance_estimates.csv",
        output_file: str = "trilateration_input.csv",
        params: Optional[AlgorithmParams] = None,
    ):

        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df_input = None
        self.df_output = None
        
        if params is not None:
            self.params = params
        else:

            if get_algorithm_params is not None:
                self.params = get_algorithm_params("default")
            else:
                # Fallback: use inline defaults
                self.params = self._create_default_params()
    
    @staticmethod
    def _create_default_params() -> AlgorithmParams:
        """Create default parameters for backward compatibility.
        
        Used when algorithm_params module not available.
        """
        class DefaultParams:
            rsrp_quality_threshold_dbm = -120.0
            min_cells_required = 3
            max_cells_to_keep = 5
            context = "default"
        
        return DefaultParams()
    
    def load_data(self) -> pd.DataFrame:
        print(f"Loading {self.input_file}...")
        self.df_input = pd.read_csv(self.input_file)
        
        print(f" Loaded {len(self.df_input)} total observations")
        print(f" Columns: {self.df_input.columns.tolist()}")
        print(f" Context: {self.params.context}")
        print(f" Parameters: RSRP_threshold={self.params.rsrp_quality_threshold_dbm} dBm, "
              f"min_cells={self.params.min_cells_required}, "
              f"max_cells={self.params.max_cells_to_keep}")
        
        return self.df_input
    
    def prepare_trilateration_input(self) -> pd.DataFrame:
        if self.df_input is None:
            raise ValueError("No input data loaded. Call load_data() first.")
        
        results = []
        grouped = self.df_input.groupby('timestamp')
        
        print(f"\nProcessing {grouped.ngroups} unique timestamps...")
        
        for timestamp, group in grouped:
            result = self._process_timestamp(timestamp, group)
            if result is not None:
                results.extend(result)
        
        if len(results) == 0:
            self.df_output = pd.DataFrame(columns=[
                'timestamp', 'cell_id', 'tower_lat', 'tower_lon', 
                'distance_m', 'uncertainty_m', 'rsrp_dbm', 'num_cells_available'
            ])
            print(f"\nProcessing complete!")
            print(f" Output rows: 0")
            print(f"  WARNING: No timestamps had sufficient cells for trilateration")
            print(f"   - Required: {self.params.min_cells_required} cells minimum")
            print(f"   - Consider using a different context or check your data")
        else:
            self.df_output = pd.DataFrame(results)
            
            self.df_output = self.df_output.sort_values('timestamp').reset_index(drop=True)
            
            print(f"\nProcessing complete!")
            print(f" Output rows: {len(self.df_output)}")
            print(f" Timestamps with sufficient cells: {self.df_output['timestamp'].nunique()}")
        
        return self.df_output
    
    def _process_timestamp(self, timestamp: str, group: pd.DataFrame) -> Optional[List[Dict]]:

        # Step 1: Filter by RSRP quality threshold (from params!)
        cells = group[group['rsrp_dbm'] >= self.params.rsrp_quality_threshold_dbm].copy()
        
        initial_count = len(group)
        filtered_count = len(cells)
        removed_count = initial_count - filtered_count
        
        # Step 2: Check minimum cells for trilateration (from params!)
        if len(cells) < self.params.min_cells_required:
            print(f" {timestamp}: INSUFFICIENT_CELLS - {filtered_count} cells "
                  f"(need {self.params.min_cells_required}, removed {removed_count} by RSRP)")
            return None
        
        # Step 3: Sort by signal strength (descending) - best signal first
        cells = cells.sort_values('rsrp_dbm', ascending=False)
        
        # Step 4: Keep only top N cells (from params!)
        cells = cells.head(self.params.max_cells_to_keep)
        kept_count = len(cells)
        
        print(f" {timestamp}: OK - {kept_count} cells "
              f"(filtered from {filtered_count}, removed {removed_count} by RSRP)")
        
        # Step 5: Extract trilateration input for each cell
        rows = []
        for _, row in cells.iterrows():
            tower_lat = row['tower_lat'] if pd.notna(row['tower_lat']) else None
            tower_lon = row['tower_lon'] if pd.notna(row['tower_lon']) else None
            
            # NEW: Lookup tower azimuth from towers.json
            tower_azimuth = self._get_tower_azimuth(row['cell_id'])
            
            trilateration_row = {
                'timestamp': timestamp,
                'cell_id': int(row['cell_id']),
                'tower_lat': tower_lat,
                'tower_lon': tower_lon,
                'distance_m': round(row['distance_m'], 1),
                'uncertainty_m': round(row['uncertainty_m'], 1),
                'rsrp_dbm': row['rsrp_dbm'],
                'num_cells_available': len(cells),
                'tower_azimuth': tower_azimuth  # NEW COLUMN
            }
            rows.append(trilateration_row)
        
        return rows
    
    def save_output(self) -> None:
        """Save trilateration input to CSV."""
        if self.df_output is None:
            raise ValueError("No output data to save. Call prepare_trilateration_input() first.")
        
        if len(self.df_output) == 0:
            print(f"\n WARNING: Output is empty. Saving empty file to {self.output_file}")
        
        print(f"\nSaving to {self.output_file}...")
        self.df_output.to_csv(self.output_file, index=False)
        print(f" Saved {len(self.df_output)} rows to {self.output_file}")
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if self.df_output is None or len(self.df_output) == 0:
            return {
                'rows': 0,
                'timestamps': 0
            }
        
        cells_per_ts = self.df_output.groupby('timestamp').size()
        
        # Handle NaN tower coordinates (missing cells)
        valid_coords = self.df_output.dropna(subset=['tower_lat', 'tower_lon'])
        
        return {
            'rows': len(self.df_output),
            'timestamps': self.df_output['timestamp'].nunique(),
            'cells_min': cells_per_ts.min(),
            'cells_max': cells_per_ts.max(),
            'cells_mean': cells_per_ts.mean(),
            'valid_coords': len(valid_coords),
            'total_rows': len(self.df_output),
            'coord_coverage': 100.0 * len(valid_coords) / len(self.df_output) if len(self.df_output) > 0 else 0
        }
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        if self.df_output is None:
            print("No output data to summarize.")
            return
        
        print("="*70)
        print("TRILATERATION INPUT SUMMARY")
        print("="*70)
        print(f"Context: {self.params.context}")
        print(f"Total output rows: {len(self.df_output)}")
        print(f"Unique timestamps: {self.df_output['timestamp'].nunique()}")
        
        if len(self.df_output) > 0:
            cells_per_ts = self.df_output.groupby('timestamp').size()
            print(f"\nCells per timestamp:")
            print(f"  Min: {cells_per_ts.min()}")
            print(f"  Max: {cells_per_ts.max()}")
            print(f"  Mean: {cells_per_ts.mean():.1f}")
            
            print(f"\nRSRP range: {self.df_output['rsrp_dbm'].min()} to {self.df_output['rsrp_dbm'].max()} dBm")
            print(f"Distance range: {self.df_output['distance_m'].min():.1f} to {self.df_output['distance_m'].max():.1f} m")
            print(f"Uncertainty range: {self.df_output['uncertainty_m'].min():.1f} to {self.df_output['uncertainty_m'].max():.1f} m")
            
            # Handle NaN tower coordinates (missing cells)
            valid_coords = self.df_output.dropna(subset=['tower_lat', 'tower_lon'])
            print(f"\nRows with valid tower coordinates:")
            print(f"  {len(valid_coords)} / {len(self.df_output)} ({100*len(valid_coords)/len(self.df_output):.1f}%)")
            
            # Show sample
            print(f"\nSample output (first 8 rows):")
            print(self.df_output.head(8).to_string(index=False))
        else:
            print("\n Output is empty - no timestamps had sufficient cells")
        
        print("="*70)
    
    def _get_tower_azimuth(self, cell_id: int) -> float:
        try:
            # Load towers.json once (or cache it)
            towers_path = Path("data/tower_data/towers.json")
            if not towers_path.exists():
                return 0.0
            
            with open(towers_path, 'r') as f:
                towers = json.load(f)
            
            cell_key = str(cell_id)
            if cell_key in towers:
                azimuth = towers[cell_key].get('azimuth', 0.0)
                if isinstance(azimuth, (int, float)) and azimuth > 0:
                    return azimuth
            return 0.0  # Invalid/missing
        except:
            return 0.0
    
    def run(self) -> pd.DataFrame:

        self.load_data()
        self.prepare_trilateration_input()
        self.save_output()
        self.print_summary()
        return self.df_output


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else "distance_estimates.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "trilateration_input.csv"
    context = sys.argv[3] if len(sys.argv) > 3 else "default"
    
    # Get algorithm parameters for the context
    params = None
    if get_algorithm_params is not None:
        params = get_algorithm_params(context)
        print(f"[INFO] Using {context.upper()} context parameters")
    
    # Create and run the preparer (with parameters)
    preparer = TrilaterationInputPreparer(
        input_file=input_file,
        output_file=output_file,
        params=params
    )
    
    df_output = preparer.run()
    
    print("\n✓ Step 3.1 complete: trilateration_input.csv ready for trilateration solver")
