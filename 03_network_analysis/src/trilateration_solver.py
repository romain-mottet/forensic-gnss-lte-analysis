import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(pos1, pos2):
    lat1, lon1 = np.radians(pos1)
    lat2, lon2 = np.radians(pos2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in meters
    R = 6_371_000
    return R * c


def angle_diff(a1: float, a2: float) -> float:
    diff = (a2 - a1 + 180) % 360 - 180
    return abs(diff)


def bearing_from_tower_to_pos(tower_pos: np.ndarray, est_pos: np.ndarray) -> float:
    tower_lat, tower_lon = tower_pos
    est_lat, est_lon = est_pos
    
    dlat = np.radians(est_lat - tower_lat)
    dlon = np.radians(est_lon - tower_lon)
    
    y = np.sin(dlon) * np.cos(np.radians(est_lat))
    x = np.cos(np.radians(tower_lat)) * np.sin(np.radians(est_lat)) - \
        np.sin(np.radians(tower_lat)) * np.cos(np.radians(est_lat)) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(y, x)) % 360
    return bearing


def residuals_wls(pos, towers, distances, weights, tower_azimuths=None, serving_azimuth_idx=-1):
    """    
    Returns:
        Weighted residuals for each tower + optional azimuth penalty residual
    """
    residuals = []
    
    # Range residuals (existing)
    for i, (tower, d_measured, weight) in enumerate(zip(towers, distances, weights)):
        d_calc = haversine_distance(pos, tower)
        residuals.append(np.sqrt(weight) * (d_measured - d_calc))
    
    if (tower_azimuths is not None and 
        serving_azimuth_idx >= 0 and 
        len(tower_azimuths) > serving_azimuth_idx and 
        tower_azimuths[serving_azimuth_idx] > 0):
        
        serving_tower = towers[serving_azimuth_idx]
        serving_azimuth = tower_azimuths[serving_azimuth_idx]
        
        # Compute bearing from serving tower to candidate position
        candidate_bearing = bearing_from_tower_to_pos(serving_tower, pos)
        
        # Angular deviation from sector centerline
        angle_error = angle_diff(candidate_bearing, serving_azimuth)
        
        # Soft penalty: quadratic outside sector (±60°), zero inside
        if angle_error > 60:
            # Penalty grows quadratically with angular deviation beyond ±60°
            azimuth_penalty_deg2 = (angle_error - 60) ** 2 * 0.01  # Tunable weight
        else:
            azimuth_penalty_deg2 = 0.0
        
        # Scale penalty to match distance residuals (meters equivalent)
        # Convert degrees^2 → meters: rough approximation (1° ≈ 100m at mid-latitudes)
        azimuth_penalty_m = azimuth_penalty_deg2 * 100
        
        # Append as extra residual with unit weight
        residuals.append(np.sqrt(1.0) * azimuth_penalty_m)
    
    return np.array(residuals)


def compute_jacobian_numerical(residual_fn, pos, towers, distances, weights, 
                                tower_azimuths=None, serving_azimuth_idx=-1, epsilon=1e-5):
    jacobian = []
    
    # Compute base residuals
    residuals = residual_fn(pos, towers, distances, weights, tower_azimuths, serving_azimuth_idx)
    
    for i in range(len(residuals)):
        # Perturb latitude
        pos_lat_plus = pos.copy()
        pos_lat_plus[0] += epsilon
        residuals_lat_plus = residual_fn(pos_lat_plus, towers, distances, weights, 
                                          tower_azimuths, serving_azimuth_idx)
        d_lat = (residuals_lat_plus[i] - residuals[i]) / epsilon
        
        # Perturb longitude
        pos_lon_plus = pos.copy()
        pos_lon_plus[1] += epsilon
        residuals_lon_plus = residual_fn(pos_lon_plus, towers, distances, weights, 
                                          tower_azimuths, serving_azimuth_idx)
        d_lon = (residuals_lon_plus[i] - residuals[i]) / epsilon
        
        jacobian.append([d_lat, d_lon])
    
    return np.array(jacobian)


def compute_gdop(jacobian, weights):
    try:
        W = np.diag(weights)
        # Covariance matrix: (J^T W J)^-1
        JtWJ = jacobian.T @ W @ jacobian
        cov_matrix = np.linalg.inv(JtWJ)
        gdop = np.sqrt(np.trace(cov_matrix))
        return gdop
    except:
        return np.inf


def quality_from_gdop(gdop):
    """Assign quality flag based on GDOP value."""
    if gdop < 5:
        return 'EXCELLENT'
    elif gdop < 8:
        return 'GOOD'
    elif gdop < 12:
        return 'ACCEPTABLE'
    else:
        return 'POOR'


class TrilaterationSolver:
    
    def __init__(self, input_file: str = 'trilateration_input.csv',
                 output_file: str = 'trilateration_results.csv'):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df_input = None
        self.df_results = None
    
    def load_data(self) -> pd.DataFrame:
        """Load trilateration input from CSV."""
        print(f" Loading {self.input_file}...")
        self.df_input = pd.read_csv(self.input_file)
        print(f" ✓ Loaded {len(self.df_input)} rows, {self.df_input['timestamp'].nunique()} timestamps")
        return self.df_input
    
    def trilaterate_timestamp(self, timestamp, cells_df):

        # Extract data
        towers = cells_df[['tower_lat', 'tower_lon']].dropna().values
        indices_valid = cells_df[['tower_lat', 'tower_lon']].notna().all(axis=1).values
        
        distances = cells_df.loc[indices_valid, 'distance_m'].values
        uncertainties = cells_df.loc[indices_valid, 'uncertainty_m'].values
        cell_ids = cells_df.loc[indices_valid, 'cell_id'].values
        
        # Extract tower azimuths
        if 'tower_azimuth' in cells_df.columns:
            tower_azimuths = cells_df.loc[indices_valid, 'tower_azimuth'].fillna(0.0).values
            serving_azimuth_idx = 0  # First cell = serving (highest RSRP)
        else:
            tower_azimuths = None
            serving_azimuth_idx = -1  # Disabled
        
        # Compute weights (inverse variance)
        weights = 1.0 / (uncertainties ** 2)
        weights_norm = weights / weights.sum() * len(weights)  # Normalize
        
        num_cells = len(cells_df)
        num_valid = len(towers)
        
        # Handle case with < 3 valid coordinates
        if num_valid < 3:
            # Use best cell by RSRP
            best_idx = cells_df['rsrp_dbm'].idxmax()
            best_cell = cells_df.loc[best_idx]
            return {
                'timestamp': timestamp,
                'est_lat': best_cell['tower_lat'],
                'est_lon': best_cell['tower_lon'],
                'num_cells_used': 1,
                'num_cells_total': num_cells,
                'gdop': np.nan,
                'quality_flag': 'INSUFFICIENT_COORDS',
                'residual_error_m': np.nan,
                'cell_ids': str(int(best_cell['cell_id']))
            }
        
        # Initial guess: weighted average of tower positions
        init_pos = np.average(towers, axis=0, weights=weights_norm)
        
        # Solve using scipy least_squares with azimuth constraint
        try:
            result = least_squares(
                residuals_wls,
                init_pos,
                args=(towers, distances, weights_norm, tower_azimuths, serving_azimuth_idx),
                max_nfev=200,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )
            
            est_lat, est_lon = result.x
            
            # Compute GDOP
            jac = compute_jacobian_numerical(residuals_wls, result.x, towers, distances, 
                                              weights_norm, tower_azimuths, serving_azimuth_idx)
            gdop = compute_gdop(jac, weights_norm)
            
            # Final residuals
            final_residuals = residuals_wls(result.x, towers, distances, weights_norm, 
                                             tower_azimuths, serving_azimuth_idx)
            residual_error = np.sqrt(np.sum(final_residuals ** 2) / len(final_residuals))
            
            quality = quality_from_gdop(gdop)
            
        except Exception:
            # If optimization fails, return weighted average
            est_lat, est_lon = init_pos
            gdop = np.inf
            quality = 'OPTIMIZATION_FAILED'
            residual_error = np.nan
        
        return {
            'timestamp': timestamp,
            'est_lat': est_lat,
            'est_lon': est_lon,
            'num_cells_used': num_valid,
            'num_cells_total': num_cells,
            'gdop': gdop,
            'quality_flag': quality,
            'residual_error_m': residual_error,
            'cell_ids': ','.join(map(str, cell_ids.astype(int)))
        }
    
    def solve(self) -> pd.DataFrame:
        """Run trilateration solver on all timestamps."""
        if self.df_input is None:
            raise ValueError("No input data loaded. Call load_data() first.")
        
        print(f"\n Running trilateration...")
        results = []
        
        for i, (timestamp, group) in enumerate(self.df_input.groupby('timestamp')):
            result = self.trilaterate_timestamp(timestamp, group)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{self.df_input['timestamp'].nunique()} timestamps")
        
        self.df_results = pd.DataFrame(results)
        
        print(f"\n Saving to {self.output_file}...")
        self.df_results.to_csv(self.output_file, index=False)
        print(f" ✓ Saved {len(self.df_results)} results")
        
        return self.df_results
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if self.df_results is None or len(self.df_results) == 0:
            return {}
        
        valid = self.df_results[self.df_results['quality_flag'] != 'INSUFFICIENT_COORDS']
        
        return {
            'total_positions': len(self.df_results),
            'successful': len(valid),
            'gdop_mean': valid['gdop'].mean() if len(valid) > 0 else np.nan,
            'residual_mean': valid['residual_error_m'].mean() if len(valid) > 0 else np.nan,
            'excellent_count': (self.df_results['quality_flag'] == 'EXCELLENT').sum(),
            'good_count': (self.df_results['quality_flag'] == 'GOOD').sum(),
            'acceptable_count': (self.df_results['quality_flag'] == 'ACCEPTABLE').sum(),
            'poor_count': (self.df_results['quality_flag'] == 'POOR').sum(),
        }
    
    def print_summary(self) -> None:
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("TRILATERATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total positions: {summary['total_positions']}")
        print(f"Successful: {summary['successful']}")
        print(f"\n📊 GDOP distribution (geometry quality):")
        print(f"   Mean: {summary['gdop_mean']:.2f}")
        print(f"\n🎯 Quality flags:")
        print(f"   EXCELLENT: {summary['excellent_count']}")
        print(f"   GOOD: {summary['good_count']}")
        print(f"   ACCEPTABLE: {summary['acceptable_count']}")
        print(f"   POOR: {summary['poor_count']}")
        print(f"\n📏 Residual errors (fit quality):")
        print(f"   Mean: {summary['residual_mean']:.1f} m")
        print(f"\n📋 Sample results (first 5):")
        cols_display = ['timestamp', 'est_lat', 'est_lon', 'num_cells_used', 'gdop', 'quality_flag']
        print(self.df_results[cols_display].head().to_string(index=False))
        print("=" * 70)



if __name__ == "__main__":
    solver = TrilaterationSolver(
        input_file='trilateration_input.csv',
        output_file='trilateration_results.csv'
    )
    
    try:
        solver.load_data()
        solver.solve()
        solver.print_summary()
        print("\n✓ Phase 3.2 complete: trilateration_results.csv ready!")
    except Exception as e:
        print(f" Error: {e}")
        exit(1)
