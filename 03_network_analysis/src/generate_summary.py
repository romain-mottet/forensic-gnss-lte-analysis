import argparse
import pandas as pd
from pathlib import Path


from helpers.helpers_summary_coverage import (
    find_coverage_csvs,
    deduplicate_by_default_context,
    save_raw_sessions,
    build_per_prefix_summaries,
    build_global_summary,
    build_hierarchical_dataframe,
    save_hierarchical_summary,
    print_hierarchical_summary,
)


from helpers.helpers_summary_distance import (
    find_distance_csvs,
    save_raw_distance_records,
    build_per_prefix_distance_summaries,
    build_global_distance_summary,
    build_context_summaries,
    build_method_summaries,
    save_merged_distance_evaluation,
    print_distance_summary,
)


from helpers.helpers_summary_trilateration import (
    find_trilateration_csvs,
    save_raw_trilateration_records,
    build_context_summaries as build_trilateration_context_summaries,
    build_method_summaries as build_trilateration_method_summaries,
    build_location_summaries,
    build_session_availability,
    save_merged_trilateration_evaluation,
    print_trilateration_summary,
)


SCRIPT_DIR = Path(__file__).parent
REPORTS_DIR = SCRIPT_DIR.parent / "reports"  


def generate_coverage_summary(output_dir: Path = None) -> Path:
    """
    Generate complete hierarchical coverage summary.

    Orchestrates the pipeline:
    1. Find all coverage CSVs across prefix/session/context hierarchy
    2. Deduplicate by context (keep only DEFAULT)
    3. Save raw sessions
    4. Build per-prefix summaries
    5. Build global summary
    6. Combine into hierarchical DataFrame
    7. Save and print results

    Args:
        output_dir: Output directory (default: root/summary)

    Returns:
        Path to main hierarchical summary CSV
    """


    if output_dir is None:
        output_dir = SCRIPT_DIR.parent / "summary"  

    prefixes = ["ixelle", "lln", "waha"]
    all_coverage_data = []

    print("\n" + "=" * 80)
    print("PHASE 1: COVERAGE SUMMARY GENERATION")
    print("=" * 80)
    print("\n Scanning reports hierarchy from src/...")
    print(f" Reports root: {REPORTS_DIR.absolute()}")
    print(f" Output root: {output_dir.absolute()}")

    # ===== STEP 1: Find all coverage CSVs =====
    for prefix in prefixes:
        prefix_dir = REPORTS_DIR / f"{prefix}_reports"

        if not prefix_dir.exists():
            print(f" {prefix}_reports not found at {prefix_dir}")
            continue

        print(f"\n {prefix}_reports: {prefix_dir}")
        prefix_csvs = find_coverage_csvs(prefix_dir)
        all_coverage_data.extend(prefix_csvs)

    if not all_coverage_data:
        print(" No coverage_report CSV files found!")
        print(" Run the pipeline first and ensure coverage_report_*.csv exist.")
        return None

    print(f"\n Found {len(all_coverage_data)} coverage reports (before dedup)")

    # ===== STEP 2: Create DataFrame and deduplicate =====
    df_all = pd.DataFrame(all_coverage_data)
    df_all = deduplicate_by_default_context(df_all)

    # ===== STEP 3: Save raw deduplicated sessions =====
    save_raw_sessions(df_all, output_dir)

    # ===== STEP 4: Build summaries =====
    df_summary_prefix = build_per_prefix_summaries(df_all, prefixes)
    global_summary = build_global_summary(df_all)

    # Append global summary to prefix summaries
    df_summary = pd.concat(
        [df_summary_prefix, pd.DataFrame([global_summary])],
        ignore_index=True
    )

    # ===== STEP 5: Build hierarchical DataFrame =====
    final_df = build_hierarchical_dataframe(df_all, df_summary)

    # ===== STEP 6: Save hierarchical summary =====
    save_hierarchical_summary(final_df, output_dir)

    # ===== STEP 7: Print preview and statistics =====
    print_hierarchical_summary(final_df, df_summary)

    return output_dir / "coverage_hierarchical_complete.csv"


def generate_distance_summary(output_dir: Path = None) -> Path:
    """
    Generate complete merged distance accuracy evaluation database.

    Orchestrates the pipeline:
    1. Find ALL distance CSVs (keep all contexts - unlike coverage)
    2. Concatenate into single master database
    3. Save raw records
    4. Build context-based summaries
    5. Build method-based summaries
    6. Save merged evaluation CSV
    7. Print statistics

    Args:
        output_dir: Output directory (default: root/summary)

    Returns:
        Path to merged distance evaluation CSV
    """

    # Default to root/summary
    if output_dir is None:
        output_dir = SCRIPT_DIR.parent / "summary"

    prefixes = ["ixelle", "lln", "waha"]
    all_records = []

    print("\n" + "=" * 80)
    print("PHASE 2: DISTANCE ACCURACY SUMMARY GENERATION")
    print("=" * 80)
    print("\n Scanning reports hierarchy for distance accuracy CSVs...")
    print(f" Reports root: {REPORTS_DIR.absolute()}")
    print(f" Output root: {output_dir.absolute()}")
    print("\n NOTE: Keeping ALL contexts (city/town/default/formula)")
    print(" because accuracy varies significantly by context")

    # ===== STEP 1: Find all distance CSVs (NO deduplication!) =====
    for prefix in prefixes:
        prefix_dir = REPORTS_DIR / f"{prefix}_reports"

        if not prefix_dir.exists():
            print(f" {prefix}_reports not found at {prefix_dir}")
            continue

        print(f"\n {prefix}_reports: {prefix_dir}")
        prefix_records = find_distance_csvs(prefix_dir)
        all_records.extend(prefix_records)

    if not all_records:
        print(" No Accuracy_distance_cell_summary CSV files found!")
        print(" Run the pipeline with distance validation and ensure CSVs exist.")
        return None

    print(f"\n Found {len(all_records)} distance records (from all contexts)")

    # ===== STEP 2: Create DataFrame (NO deduplication - this is the key difference!) =====
    df_all = pd.DataFrame(all_records)

    # ===== STEP 3: Save raw records =====
    save_raw_distance_records(df_all, output_dir)

    # ===== STEP 4: Save merged database (MAIN OUTPUT) =====
    merged_path = save_merged_distance_evaluation(df_all, output_dir)

    # ===== STEP 5: Build and save context summaries =====
    print("\n Building context-based summaries...")
    df_context_summary = build_context_summaries(df_all)
    context_summary_path = output_dir / "distance_summary_by_context.csv"
    df_context_summary.to_csv(context_summary_path, index=False)
    print(f"✓ Saved context summary: {context_summary_path}")

    # ===== STEP 6: Build and save method summaries =====
    print("\n Building method-based summaries...")
    df_method_summary = build_method_summaries(df_all)
    method_summary_path = output_dir / "distance_summary_by_method.csv"
    df_method_summary.to_csv(method_summary_path, index=False)
    print(f"✓ Saved method summary: {method_summary_path}")

    # ===== STEP 7: Print statistics =====
    print_distance_summary(df_all, output_dir)

    return merged_path


def generate_trilateration_summary(output_dir: Path = None) -> Path:
    """
    Generate complete merged trilateration accuracy evaluation database.

    Orchestrates the pipeline:
    1. Find ALL accuracy_summary_*_metrics.csv files (keep all contexts)
    2. Concatenate into single master database
    3. Save raw records
    4. Build context-based summaries
    5. Build method-based summaries
    6. Build location-based summaries (by prefix: ixelle/lln/waha)
    7. Build session availability tracking (shows missing contexts per session)
    8. Save merged evaluation CSV
    9. Print statistics

    Args:
        output_dir: Output directory (default: root/summary)

    Returns:
        Path to merged trilateration evaluation CSV
    """


    if output_dir is None:
        output_dir = SCRIPT_DIR.parent / "summary"

    prefixes = ["ixelle", "lln", "waha"]
    all_records = []

    print("\n" + "=" * 80)
    print("PHASE 3: TRILATERATION ACCURACY SUMMARY GENERATION")
    print("=" * 80)
    print("\n Scanning reports hierarchy for trilateration accuracy CSVs...")
    print(f" Reports root: {REPORTS_DIR.absolute()}")
    print(f" Output root: {output_dir.absolute()}")
    print("\n NOTE: Keeping ALL contexts (city/town/default/formula/village)")
    print(" because accuracy varies significantly by context")

    # ===== STEP 1: Find all trilateration CSVs (NO deduplication!) =====
    for prefix in prefixes:
        prefix_dir = REPORTS_DIR / f"{prefix}_reports"

        if not prefix_dir.exists():
            print(f" {prefix}_reports not found at {prefix_dir}")
            continue

        print(f"\n🔍 {prefix}_reports: {prefix_dir}")
        prefix_records = find_trilateration_csvs(prefix_dir)
        all_records.extend(prefix_records)

    if not all_records:
        print(" No accuracy_summary_*_metrics CSV files found!")
        print(" Run trilateration solver and ensure accuracy_summary_*_metrics.csv exist.")
        return None

    print(f"\n Found {len(all_records)} trilateration records (from all contexts)")

    # ===== STEP 2: Create DataFrame (NO deduplication) =====
    df_all = pd.DataFrame(all_records)

    # ===== STEP 3: Save raw records =====
    save_raw_trilateration_records(df_all, output_dir)

    # ===== STEP 4: Save merged database (MAIN OUTPUT) =====
    merged_path = save_merged_trilateration_evaluation(df_all, output_dir)

    # ===== STEP 5: Build and save context summaries =====
    print("\n Building context-based summaries...")
    df_context_summary = build_trilateration_context_summaries(df_all)
    context_summary_path = output_dir / "trilateration_summary_by_context.csv"
    df_context_summary.to_csv(context_summary_path, index=False)
    print(f"✓ Saved context summary: {context_summary_path}")

    # ===== STEP 6: Build and save method summaries =====
    print("\n Building method-based summaries...")
    df_method_summary = build_trilateration_method_summaries(df_all)
    method_summary_path = output_dir / "trilateration_summary_by_method.csv"
    df_method_summary.to_csv(method_summary_path, index=False)
    print(f"✓ Saved method summary: {method_summary_path}")

    # ===== STEP 7: Build and save location summaries =====
    print("\n Building location-based summaries...")
    df_location_summary = build_location_summaries(df_all, prefixes)
    location_summary_path = output_dir / "trilateration_summary_by_location.csv"
    df_location_summary.to_csv(location_summary_path, index=False)
    print(f"✓ Saved location summary: {location_summary_path}")

    # ===== STEP 8: Build and save session availability =====
    print("\n Building session availability tracking...")
    df_session_avail = build_session_availability(df_all, prefixes)
    session_avail_path = output_dir / "trilateration_session_availability.csv"
    df_session_avail.to_csv(session_avail_path, index=False)
    print(f"✓ Saved session availability: {session_avail_path}")

    # ===== STEP 9: Print statistics =====
    print_trilateration_summary(df_all, output_dir)

    return merged_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate nested coverage, distance, AND trilateration accuracy reports (from src/). "
            "Coverage deduplicates by context. Distance and trilateration keep all contexts. "
            "Outputs to root/summary by default."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: root/summary)",
    )

    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip coverage summary generation",
    )

    parser.add_argument(
        "--skip-distance",
        action="store_true",
        help="Skip distance accuracy summary generation",
    )

    parser.add_argument(
        "--skip-trilateration",
        action="store_true",
        help="Skip trilateration accuracy summary generation",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not args.skip_coverage:
        print("\n Starting coverage summary generation...")
        cov_path = generate_coverage_summary(output_dir)
        if cov_path:
            print(f"\n Coverage summary ready: {cov_path}")

    if not args.skip_distance:
        print("\n Starting distance accuracy summary generation...")
        dist_path = generate_distance_summary(output_dir)
        if dist_path:
            print(f"\n Distance evaluation database ready: {dist_path}")

    if not args.skip_trilateration:
        print("\n Starting trilateration accuracy summary generation...")
        trilat_path = generate_trilateration_summary(output_dir)
        if trilat_path:
            print(f"\n Trilateration evaluation database ready: {trilat_path}")

    print("\n" + "=" * 80)
    print(" SUMMARY GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()