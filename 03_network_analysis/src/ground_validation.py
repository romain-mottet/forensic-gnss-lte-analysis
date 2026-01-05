from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class GroundTruthValidator:
    
    def __init__(self,
                 results_file: str = 'trilateration_results.csv',
                 ground_truth_file: str = None, 
                 base_name: str = None,  
                 validation_output: str = 'validation_results.csv',
                 summary_output: str = 'accuracy_summary.txt'):
        
        self.results_file = Path(results_file)
        self.base_name = base_name 
        

        if ground_truth_file is None:
            if base_name and base_name.strip(): 
                # Use base_name to construct filename
                ground_truth_file = f'ground_truth_{base_name}.csv'
                print(f"[DEBUG] Using base_name '{base_name}' → {ground_truth_file}")
            else:
                # Fallback only if base_name is empty/None
                ground_truth_file = 'ground_truth.csv'
                print(f"[DEBUG] No base_name provided, using fallback: {ground_truth_file}")
        else:
            print(f"[DEBUG] Using provided ground_truth_file: {ground_truth_file}")
        
        self.ground_truth_file = Path(ground_truth_file)
        self.validation_output = Path(validation_output)
        self.summary_output = Path(summary_output)
        
        self.metrics_csv_output = Path(str(summary_output).replace('.txt', '_metrics.csv'))
        
        self.df_results = None
        self.df_truth = None
        self.df_validation = None
        self.ground_truth_available = False
    
    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        self.df_results = pd.read_csv(self.results_file)
        
        # Check if ground truth file exists
        if not self.ground_truth_file.exists():
            print(f"[INFO] Ground truth file not found: {self.ground_truth_file}")
            print(f"[INFO] Validation will be skipped (metrics will be NaN)")
            self.ground_truth_available = False
            self.df_truth = None
            return self.df_results, None
        
        self.df_truth = pd.read_csv(self.ground_truth_file)
        self.ground_truth_available = True
        print(f"[INFO] Ground truth file loaded: {self.ground_truth_file}")
        return self.df_results, self.df_truth
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:

        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        R = 6_371_000  # Earth radius in meters
        return R * c
    
    def validate(self) -> Optional[pd.DataFrame]:
        if self.df_results is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Skip validation if ground truth not available
        if not self.ground_truth_available or self.df_truth is None:
            print("[INFO] Skipping validation - ground truth not available")
            return None
        
        validation = []
        
        for _, result in self.df_results.iterrows():
            timestamp = result['timestamp']
            est_lat = result['est_lat']
            est_lon = result['est_lon']
            
            # Find ground truth for this timestamp
            truth_rows = self.df_truth[self.df_truth['timestamp'] == timestamp]
            if len(truth_rows) == 0:
                continue  # No ground truth for this time
            
            truth_row = truth_rows.iloc[0]  # Take first entry if multiple
            truth_lat = truth_row['gps_latitude']
            truth_lon = truth_row['gps_longitude']
            
            # Calculate error distance
            error_m = self.haversine_distance(est_lat, est_lon, truth_lat, truth_lon)
            
            # Get trilateration metrics
            residual_error = result['residual_error_m'] if pd.notna(result['residual_error_m']) else np.nan
            gdop = result['gdop'] if pd.notna(result['gdop']) else np.nan
            
            # Check if within uncertainty bounds
            uncertainty_m = 200
            within_bounds = error_m < uncertainty_m
            
            validation.append({
                'timestamp': timestamp,
                'est_lat': est_lat,
                'est_lon': est_lon,
                'truth_lat': truth_lat,
                'truth_lon': truth_lon,
                'error_m': error_m,
                'within_uncertainty': within_bounds,
                'num_cells': result['num_cells_used'],
                'gdop': gdop,
                'residual_error_m': residual_error,
                'quality_flag': result['quality_flag']
            })
        
        self.df_validation = pd.DataFrame(validation)
        
        if len(self.df_validation) > 0:
            self.df_validation.to_csv(self.validation_output, index=False)
        
        return self.df_validation
    
    def compute_accuracy_metrics(self) -> Dict:

        if self.df_validation is None or len(self.df_validation) == 0:
            return {
                'samples': 0,
                'cep': np.nan,
                'r95': np.nan,
                'rmse': np.nan,
                'mean_error': np.nan,
                'std_error': np.nan,
                'min_error': np.nan,
                'max_error': np.nan,
                'accuracy_by_cells': {},
                'quality_counts': {},
                'within_bounds_pct': np.nan,
                'good_quality_count': 0,
            }
        
        errors = self.df_validation['error_m'].values

        cep = np.percentile(errors, 50)
        r95 = np.percentile(errors, 95)
        
        rmse = np.sqrt(np.mean(errors ** 2))
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        accuracy_by_cells = {}
        for num_cells in [1, 3, 4, 5]:
            subset = self.df_validation[self.df_validation['num_cells'] == num_cells]
            if len(subset) > 0:
                accuracy_by_cells[num_cells] = np.sqrt(np.mean(subset['error_m'].values ** 2))
        
        quality_counts = self.df_validation['quality_flag'].value_counts().to_dict()
        
        within_bounds_pct = 100 * self.df_validation['within_uncertainty'].sum() / len(self.df_validation)
        
        good_quality_count = len(self.df_validation[self.df_validation['gdop'] < 8])
        
        return {
            'samples': len(self.df_validation),
            'cep': cep,
            'r95': r95,
            'rmse': rmse,
            'mean_error': mean_error,
            'std_error': std_error,
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'accuracy_by_cells': accuracy_by_cells,
            'quality_counts': quality_counts,
            'within_bounds_pct': within_bounds_pct,
            'good_quality_count': good_quality_count,
        }
    
    def generate_summary(self) -> str:
        metrics = self.compute_accuracy_metrics()

        if metrics['samples'] == 0:
            return f"""
{'='*80}
GEOLOCATION ACCURACY SUMMARY (Phase 3.3 Validation)
{'='*80}

STATUS: No ground truth data available

To validate your positioning estimates, you need a ground truth GPS data file:

File name: {self.ground_truth_file}

Format: CSV with columns
  - timestamp (same format as trilateration_results.csv)
  - gps_latitude (decimal degrees)
  - gps_longitude (decimal degrees)

Example rows:
  timestamp,gps_latitude,gps_longitude
  2025.12.15_18.07.54,50.8245,-4.3612
  2025.12.15_18.07.55,50.8246,-4.3611

Once you create the ground truth file, run the pipeline again and validation
will compare your estimates against the ground truth.

{'='*80}
"""
        
        summary = f"""
{'='*80}
GEOLOCATION ACCURACY SUMMARY (Phase 3.3 Validation)
{'='*80}

SAMPLES
  Total positions compared to ground truth: {metrics['samples']}

ERROR METRICS
  CEP (50th percentile):       {metrics['cep']:>10.1f} m
  R95 (95th percentile):       {metrics['r95']:>10.1f} m
  RMSE (root mean square):     {metrics['rmse']:>10.1f} m
  Mean error (bias):           {metrics['mean_error']:>10.1f} m
  Std deviation:               {metrics['std_error']:>10.1f} m
  Min error:                   {metrics['min_error']:>10.1f} m
  Max error:                   {metrics['max_error']:>10.1f} m

ACCURACY BY NUMBER OF CELLS
"""
        
        for num_cells in sorted(metrics['accuracy_by_cells'].keys()):
            rmse_cells = metrics['accuracy_by_cells'][num_cells]
            summary += f"  {num_cells} cell(s):                    {rmse_cells:>10.1f} m RMSE\n"
        
        summary += f"""
QUALITY DISTRIBUTION
"""
        
        for quality, count in sorted(metrics['quality_counts'].items()):
            pct = 100 * count / metrics['samples']
            summary += f"  {quality:20s}: {count:3d} ({pct:5.1f}%)\n"
        
        good_quality_pct = 100 * metrics['good_quality_count'] / metrics['samples']
        
        summary += f"""
GEOMETRY QUALITY
  GDOP < 8 (good geometry):    {metrics['good_quality_count']:3d} ({good_quality_pct:5.1f}%)

WITHIN UNCERTAINTY BOUNDS
  Error < 200m:                {metrics['within_bounds_pct']:>6.1f}%

{'='*80}
INTERPRETATION
{'='*80}

CEP Thresholds:
  CEP < 100m  :  Excellent positioning accuracy
  CEP 100-300m:  Good positioning accuracy
  CEP > 300m  :  Fair positioning (may indicate distance calibration issues)

Your Results:
  Accuracy Level: {'EXCELLENT' if metrics['rmse'] < 100 else 'GOOD' if metrics['rmse'] < 300 else 'FAIR' if metrics['rmse'] < 800 else 'POOR'} ({metrics['rmse']:.1f} m RMSE)

Contributing Factors for High RMSE:
  1. Distance model calibration (see Phase 2.2)
  2. Tower location errors
  3. Multipath/NLOS conditions
  4. Antenna orientation/gain assumptions

{'='*80}
ANALYSIS PARAMETERS
{'='*80}

Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Context:    {self.base_name if self.base_name else 'default'}

{'='*80}
"""
        
        return summary
    
    def save_summary(self) -> None:
        summary = self.generate_summary()
        
        self.summary_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.summary_output, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"[OK] TXT summary saved: {self.summary_output}")
    
    
    def save_metrics_csv(self, context: str = None) -> None:

        metrics = self.compute_accuracy_metrics()

        distance_method = 'path_loss'
        if context and 'formula' in context.lower():
            distance_method = 'formula'

        csv_row = {
            'timestamp': datetime.now().isoformat(),
            'context': context if context else self.base_name if self.base_name else 'default',
            'distance_method': distance_method,
            'samples': metrics['samples'],
            'cep_m': round(metrics['cep'], 1) if not np.isnan(metrics['cep']) else np.nan,
            'r95_m': round(metrics['r95'], 1) if not np.isnan(metrics['r95']) else np.nan,
            'rmse_m': round(metrics['rmse'], 1) if not np.isnan(metrics['rmse']) else np.nan,
            'mean_error_m': round(metrics['mean_error'], 1) if not np.isnan(metrics['mean_error']) else np.nan,
            'std_error_m': round(metrics['std_error'], 1) if not np.isnan(metrics['std_error']) else np.nan,
            'min_error_m': round(metrics['min_error'], 1) if not np.isnan(metrics['min_error']) else np.nan,
            'max_error_m': round(metrics['max_error'], 1) if not np.isnan(metrics['max_error']) else np.nan,
        }
        
        # Add accuracy by cells
        for num_cells in [1, 3, 4, 5]:
            if num_cells in metrics['accuracy_by_cells']:
                csv_row[f'accuracy_{num_cells}cell_rmse_m'] = round(metrics['accuracy_by_cells'][num_cells], 1)
            else:
                csv_row[f'accuracy_{num_cells}cell_rmse_m'] = np.nan
        
        # Add quality counts with 'gdop' inserted into the key
        for quality in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'INSUFFICIENTCOORDS']:
            # Format: quality_gdop_EXCELLENT_count
            key_name = f'quality_gdop_{quality}_count'
            if quality in metrics['quality_counts']:
                csv_row[key_name] = metrics['quality_counts'][quality]
            else:
                csv_row[key_name] = 0
        
        # Add geometry and bounds
        csv_row['within_200m_pct'] = round(metrics['within_bounds_pct'], 1) if not np.isnan(metrics['within_bounds_pct']) else np.nan
        csv_row['gdop_lt8_count'] = metrics['good_quality_count']
        
        # Create DataFrame and save
        df_metrics = pd.DataFrame([csv_row])
        
        self.metrics_csv_output.parent.mkdir(parents=True, exist_ok=True)
        
        df_metrics.to_csv(self.metrics_csv_output, index=False)
        
        print(f"[OK] CSV metrics saved: {self.metrics_csv_output}")
    
    def run(self, context: str = None) -> Tuple[Optional[pd.DataFrame], Dict]:
        self.load_data()
        
        if self.ground_truth_available:
            self.validate()
            self.save_summary()
            self.save_metrics_csv(context=context)  # NEW: Save metrics CSV
        
        metrics = self.compute_accuracy_metrics()
        return self.df_validation, metrics


def validate_positions(results_file: str = 'trilateration_results.csv',
                      ground_truth_file: str = None,
                      base_name: str = None,  # CRITICAL: Pass this!
                      validation_output: str = 'validation_results.csv',
                      summary_output: str = 'accuracy_summary.txt',
                      context: str = None) -> Tuple[Optional[pd.DataFrame], Dict]:

    validator = GroundTruthValidator(
        results_file=results_file,
        ground_truth_file=ground_truth_file,
        base_name=base_name,  # CRITICAL: Pass base_name!
        validation_output=validation_output,
        summary_output=summary_output
    )
    
    return validator.run(context=context)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ground_validation.py results_file [ground_truth_file] [base_name] [context]")
        print("Example: python ground_validation.py trilateration_results.csv ground_truth_ixelle_4.csv ixelle_4 city")
        sys.exit(1)
    
    results = sys.argv[1]
    truth = sys.argv[2] if len(sys.argv) > 2 else None
    base_name = sys.argv[3] if len(sys.argv) > 3 else None
    context = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        validator = GroundTruthValidator(
            results_file=results,
            ground_truth_file=truth,
            base_name=base_name
        )
        
        print("Loading data...")
        validator.load_data()
        
        print("Validating positions...")
        validator.validate()
        
        print("Computing metrics...")
        metrics = validator.compute_accuracy_metrics()
        
        print("\n" + validator.generate_summary())
        
        if metrics['samples'] > 0:
            validator.save_summary()
            validator.save_metrics_csv(context=context)
            print(f"✓ Validation results saved to: {validator.validation_output}")
            print(f"✓ Accuracy summary (TXT) saved to: {validator.summary_output}")
            print(f"✓ Metrics (CSV) saved to: {validator.metrics_csv_output}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)