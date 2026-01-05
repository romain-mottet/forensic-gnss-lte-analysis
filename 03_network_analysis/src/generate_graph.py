import argparse
import pandas as pd
from pathlib import Path


from helpers.helpers_graphs_coverage import generate_coverage_comparison_graph
from helpers.helpers_graphs_distance import (
    generate_distance_mae_comparison_graph,
    generate_distance_error_distribution_graph
)
from helpers.helpers_graphs_trilateration import generate_trilateration_accuracy_graph
from helpers.helpers_graphs_trilateration_cells import generate_cells_vs_accuracy_scatter


SCRIPT_DIR = Path(__file__).parent
SUMMARY_DIR = SCRIPT_DIR.parent / "summary"


def generate_all_graphs(output_dir: Path = None) -> Path:
    """
    Generate all analysis graphs.
    
    Orchestrates the pipeline:
    1. Load summary data (coverage_hierarchical_complete.csv)
    2. Generate coverage comparison graph
    3. Load distance data (distance_summary_by_context.csv, merged_distance_evaluation.csv)
    4. Generate distance comparison graphs (bar + box plot)
    5. Load trilateration data (trilateration_summary_by_context.csv)
    6. Generate trilateration accuracy graph
    7. Load session and trilateration records
    8. Generate cells vs accuracy scatter plot
    
    Args:
        output_dir: Output directory (default: root/graphs)
        
    Returns:
        Path to graphs output directory
    """
    
    # Default to root/graphs (not src/graphs)
    if output_dir is None:
        output_dir = SCRIPT_DIR.parent / "graphs"  # src/../graphs = root/graphs
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GRAPH GENERATION PIPELINE")
    print("=" * 70)
    print(f" Summary root: {SUMMARY_DIR.absolute()}")
    print(f" Output root: {output_dir.absolute()}")
    print()
    
    # ===== Load coverage hierarchical data =====
    coverage_csv = SUMMARY_DIR / "coverage_hierarchical_complete.csv"
    
    if not coverage_csv.exists():
        print(f" ERROR: {coverage_csv} not found!")
        print(" Run generate_summary.py first to create coverage data.")
        return None
    
    try:
        df_coverage = pd.read_csv(coverage_csv)
        print(f" Loaded coverage data: {coverage_csv.name}")
        print(f"   Rows: {len(df_coverage)}")
    except Exception as e:
        print(f" Error loading coverage CSV: {e}")
        return None
    
    print()
    
    # ===== GRAPH 1: Coverage Comparison =====
    print(" Generating Graph 1: Coverage Comparison by Environment")
    try:
        generate_coverage_comparison_graph(
            df_coverage=df_coverage,
            output_dir=output_dir,
            output_filename="coverage_comparison.pdf"
        )
        print(f" Graph saved: coverage_comparison.pdf")
    except Exception as e:
        print(f" Error generating coverage comparison graph: {e}")
    
    print()
    
    # ===== Load distance evaluation data =====
    distance_summary_csv = SUMMARY_DIR / "distance_summary_by_context.csv"
    merged_distance_csv = SUMMARY_DIR / "merged_distance_evaluation.csv"
    
    if not distance_summary_csv.exists():
        print(f"  WARNING: {distance_summary_csv} not found!")
        print(" Skipping distance graphs. Run generate_summary.py first.")
        print()
    elif not merged_distance_csv.exists():
        print(f"  WARNING: {merged_distance_csv} not found!")
        print(" Skipping distance graphs. Run generate_summary.py first.")
        print()
    else:
        try:
            df_distance = pd.read_csv(distance_summary_csv)
            df_merged = pd.read_csv(merged_distance_csv)
            print(f"Loaded distance data: {distance_summary_csv.name}")
            print(f"   Rows: {len(df_distance)}")
            print(f"Loaded merged evaluation: {merged_distance_csv.name}")
            print(f"   Rows: {len(df_merged)}")
        except Exception as e:
            print(f" Error loading distance CSV: {e}")
            return None
        
        print()
        
        # ===== GRAPH 2: Distance MAE Comparison =====
        print(" Generating Graph 2: Distance MAE Comparison by Environment")
        try:
            generate_distance_mae_comparison_graph(
                df_distance=df_distance,
                output_dir=output_dir,
                output_filename="distance_mae_comparison.pdf"
            )
            print(f"Graph saved: distance_mae_comparison.pdf")
        except Exception as e:
            print(f"Error generating distance MAE comparison graph: {e}")
        
        print()
        
        # ===== GRAPH 3: Distance Error Distribution =====
        print(" Generating Graph 3: Distance Error Distribution (Box Plot)")
        try:
            generate_distance_error_distribution_graph(
                df_merged=df_merged,
                output_dir=output_dir,
                output_filename="distance_error_distribution.pdf"
            )
            print(f" Graph saved: distance_error_distribution.pdf")
        except Exception as e:
            print(f" Error generating distance error distribution graph: {e}")
        
        print()
    
    # ===== Load trilateration evaluation data =====
    trilateration_context_csv = SUMMARY_DIR / "trilateration_summary_by_context.csv"
    
    if not trilateration_context_csv.exists():
        print(f" WARNING: {trilateration_context_csv} not found!")
        print(" Skipping trilateration graph. Run generate_summary.py first.")
        print()
    else:
        try:
            df_trilat = pd.read_csv(trilateration_context_csv)
            print(f" Loaded trilateration data: {trilateration_context_csv.name}")
            print(f"   Rows: {len(df_trilat)}")
        except Exception as e:
            print(f" Error loading trilateration CSV: {e}")
            return None
        
        print()
        
        # ===== GRAPH 4: Trilateration Accuracy =====
        print(" Generating Graph 4: Trilateration Accuracy (Path Loss vs Formula)")
        try:
            generate_trilateration_accuracy_graph(
                df_context=df_trilat,
                output_dir=output_dir,
                output_filename="trilateration_accuracy.pdf"
            )
            print(f" Graph saved: trilateration_accuracy.pdf")
        except Exception as e:
            print(f"Error generating trilateration accuracy graph: {e}")
        
        print()
    
    # ===== Load trilateration cells vs accuracy data =====
    sessions_csv = SUMMARY_DIR / "01_raw_sessions.csv"
    trilat_records_csv = SUMMARY_DIR / "02_raw_trilateration_records.csv"
    
    if not sessions_csv.exists():
        print(f" WARNING: {sessions_csv} not found!")
        print(" Skipping cells vs accuracy graph. Run generate_summary.py first.")
        print()
    elif not trilat_records_csv.exists():
        print(f" WARNING: {trilat_records_csv} not found!")
        print(" Skipping cells vs accuracy graph. Run generate_summary.py first.")
        print()
    else:
        try:
            df_sessions = pd.read_csv(sessions_csv)
            df_trilat_records = pd.read_csv(trilat_records_csv)
            print(f" Loaded sessions data: {sessions_csv.name}")
            print(f"   Rows: {len(df_sessions)}")
            print(f" Loaded trilateration records: {trilat_records_csv.name}")
            print(f"   Rows: {len(df_trilat_records)}")
        except Exception as e:
            print(f" Error loading trilateration records CSV: {e}")
            return None
        
        print()
        
        # ===== GRAPH 5: Cells vs Accuracy Scatter =====
        print("📊 Generating Graph 5: Cells Found vs Trilateration Accuracy (Scatter)")
        try:
            generate_cells_vs_accuracy_scatter(
                df_sessions=df_sessions,
                df_trilat=df_trilat_records,
                output_dir=output_dir,
                output_filename="cells_vs_accuracy.pdf"
            )
            print(f" Graph saved: cells_vs_accuracy.pdf")
        except Exception as e:
            print(f" Error generating cells vs accuracy scatter graph: {e}")
        
        print()
    
    print("=" * 70)
    print(" GRAPH GENERATION COMPLETE")
    print("=" * 70)
    print(f" Output directory: {output_dir.absolute()}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate analysis graphs from summary data (from src/). "
            "Outputs to root/graphs by default."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: root/graphs)",
    )
    
    args = parser.parse_args()
    
    generate_all_graphs(Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()