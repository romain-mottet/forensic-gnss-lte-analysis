import pandas as pd
from pathlib import Path
from typing import List, Dict


def find_distance_csvs(prefix_reports_dir: Path) -> List[Dict]:
    all_csvs = []
    
    # Walk through session_reports directories
    for session_dir in prefix_reports_dir.glob("*_reports"):
        print(f"📁 Processing session: {session_dir.name}")
        
        # Walk ALL subdirectories (including context subdirs with different results)
        for csv_path in session_dir.rglob("Accuracy_distance_cell_summary_*.csv"):
            try:
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    continue
                
                # Extract metadata from filepath
                # Example: reports/lln_reports/lln_1_reports/city/Accuracy_distance_cell_summary_lln_1_city.csv
                prefix = prefix_reports_dir.name.replace("_reports", "")
                session = session_dir.name.replace("_reports", "")
                context_path = csv_path.parent.relative_to(session_dir)
                full_context = str(context_path).replace("/", "_").replace("\\", "_")
                
                # For each row in the CSV (per-cell metrics)
                for idx, row in df.iterrows():
                    row_dict = row.to_dict()
                    
                    # Add metadata
                    row_dict.update({
                        'prefix': prefix,
                        'session': session,
                        'context_path': str(context_path),
                        'full_context': full_context,
                        'csv_file': csv_path.name,
                    })
                    
                    all_csvs.append(row_dict)
                
                # Print progress
                try:
                    rel_path = csv_path.relative_to(prefix_reports_dir.parent.parent)
                    rel_str = f"reports/{rel_path}"
                except ValueError:
                    rel_str = str(csv_path)
                
                print(f" ✓ {rel_str} → {len(df)} cells")
                
            except Exception as e:
                print(f" Error reading {csv_path}: {e}")
    
    return all_csvs


def save_raw_distance_records(df_all: pd.DataFrame, output_dir: Path) -> Path:

    raw_path = output_dir / "01_raw_distance_records.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep all relevant columns
    record_cols = [
        'prefix', 'session', 'full_context', 'context_path', 'csv_file',
        'cell_id', 'sample_count', 'mae_m', 'rmse_m', 'cep_m', 'r95_m',
        'mean_error_m', 'std_error_m', 'min_error_m', 'max_error_m',
        'frequency_mhz', 'distance_method', 'primary_confidence', 'context'
    ]
    
    # Keep only columns that exist
    record_cols_present = [c for c in record_cols if c in df_all.columns]
    
    df_all[record_cols_present].to_csv(raw_path, index=False)
    
    print(f"📄 Saved {len(df_all)} distance records: {raw_path}")
    
    return raw_path


def build_per_prefix_distance_summaries(df_all: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    
    for prefix in prefixes:
        prefix_df = df_all[df_all['prefix'] == prefix]
        
        if prefix_df.empty:
            continue
        
        # Count unique (session, context) combinations
        n_sessions = prefix_df.groupby(['session', 'full_context']).ngroups
        
        summary_rows.append({
            'prefix': prefix,
            'session': f"{prefix}_SUMMARY",
            'full_context': "AGGREGATED",
            'n_records': int(len(prefix_df)),
            'n_cells': int(prefix_df['cell_id'].nunique()),
            'n_sessions': int(n_sessions),
            'avg_mae_m': round(prefix_df['mae_m'].mean(), 1),
            'avg_rmse_m': round(prefix_df['rmse_m'].mean(), 1),
            'avg_cep_m': round(prefix_df['cep_m'].mean(), 1),
            'avg_r95_m': round(prefix_df['r95_m'].mean(), 1),
        })
    
    return pd.DataFrame(summary_rows)


def build_global_distance_summary(df_all: pd.DataFrame) -> Dict:
    return {
        'prefix': 'GLOBAL',
        'session': 'GLOBAL_SUMMARY',
        'full_context': 'AGGREGATED',
        'n_records': int(len(df_all)),
        'n_cells': int(df_all['cell_id'].nunique()),
        'n_sessions': int(df_all.groupby(['prefix', 'session', 'full_context']).ngroups),
        'avg_mae_m': round(df_all['mae_m'].mean(), 1),
        'avg_rmse_m': round(df_all['rmse_m'].mean(), 1),
        'avg_cep_m': round(df_all['cep_m'].mean(), 1),
        'avg_r95_m': round(df_all['r95_m'].mean(), 1),
    }


def build_context_summaries(df_all: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    
    for (prefix, full_context), group in df_all.groupby(['prefix', 'full_context']):
        summary_rows.append({
            'prefix': prefix,
            'context': full_context,
            'n_records': int(len(group)),
            'n_cells': int(group['cell_id'].nunique()),
            'n_sessions': int(group['session'].nunique()),
            'avg_mae_m': round(group['mae_m'].mean(), 1),
            'avg_rmse_m': round(group['rmse_m'].mean(), 1),
            'avg_cep_m': round(group['cep_m'].mean(), 1),
            'avg_r95_m': round(group['r95_m'].mean(), 1),
        })
    
    return pd.DataFrame(summary_rows)


def build_method_summaries(df_all: pd.DataFrame) -> pd.DataFrame:

    summary_rows: List[Dict] = []
    
    for method, group in df_all.groupby('distance_method'):
        summary_rows.append({
            'distance_method': method,
            'n_records': int(len(group)),
            'n_cells': int(group['cell_id'].nunique()),
            'avg_mae_m': round(group['mae_m'].mean(), 1),
            'avg_rmse_m': round(group['rmse_m'].mean(), 1),
            'avg_cep_m': round(group['cep_m'].mean(), 1),
            'avg_r95_m': round(group['r95_m'].mean(), 1),
            'avg_sample_count': round(group['sample_count'].mean(), 0),
        })
    
    return pd.DataFrame(summary_rows)


def save_merged_distance_evaluation(df_all: pd.DataFrame, output_dir: Path) -> Path:
    merged_path = output_dir / "merged_distance_evaluation.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep all columns for maximum flexibility in analysis
    df_all.to_csv(merged_path, index=False)
    
    print(f"\n MERGED DISTANCE EVALUATION SAVED: {merged_path}")
    print(f"   Total records: {len(df_all):,}")
    print(f"   Unique cells: {df_all['cell_id'].nunique():,}")
    print(f"   Sessions: {df_all.groupby(['prefix', 'session']).ngroups}")
    print(f"   Contexts: {df_all['full_context'].nunique()}")
    
    return merged_path


def print_distance_summary(df_all: pd.DataFrame, output_dir: Path):

    print("\n" + "=" * 80)
    print("DISTANCE ACCURACY SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n OVERALL METRICS:")
    print(f"   Total records (per-cell evaluations): {len(df_all):,}")
    print(f"   Unique cells: {df_all['cell_id'].nunique():,}")
    print(f"   Unique sessions: {df_all.groupby(['prefix', 'session']).ngroups}")
    print(f"   Unique contexts: {df_all['full_context'].nunique()}")
    
    print(f"\n ACCURACY ACROSS ALL DATA:")
    print(f"   Mean MAE: {df_all['mae_m'].mean():.1f} m")
    print(f"   Mean RMSE: {df_all['rmse_m'].mean():.1f} m")
    print(f"   Mean CEP: {df_all['cep_m'].mean():.1f} m")
    print(f"   Mean R95: {df_all['r95_m'].mean():.1f} m")
    
    # By context
    print(f"\n BY CONTEXT:")
    context_summary = df_all.groupby('full_context')[['mae_m', 'rmse_m']].mean()
    print(context_summary.round(1).to_string())
    
    # By method
    print(f"\n BY DISTANCE METHOD:")
    method_summary = df_all.groupby('distance_method')[['mae_m', 'rmse_m']].mean()
    print(method_summary.round(1).to_string())
    
    # By prefix
    print(f"\n BY PREFIX:")
    prefix_summary = df_all.groupby('prefix')[['mae_m', 'rmse_m']].mean()
    print(prefix_summary.round(1).to_string())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    from pathlib import Path
    
    # For standalone testing
    script_dir = Path(__file__).parent
    reports_dir = script_dir.parent / "reports"
    output_dir = script_dir.parent / "summary"
    
    prefixes = ["ixelle", "lln", "waha"]
    all_records = []
    
    for prefix in prefixes:
        prefix_dir = reports_dir / f"{prefix}_reports"
        if prefix_dir.exists():
            records = find_distance_csvs(prefix_dir)
            all_records.extend(records)
    
    if all_records:
        df = pd.DataFrame(all_records)
        save_raw_distance_records(df, output_dir)
        save_merged_distance_evaluation(df, output_dir)
        print_distance_summary(df, output_dir)
    else:
        print(" No distance accuracy records found!")
