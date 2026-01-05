import sys
from pathlib import Path
from typing import Dict, Tuple

from src.config.parameters import (
    BASE_DIR,
    SMARTPHONE_DIR,
    WATCH_DIR,
    RESULTS_DIR,
    ALLOWED_OFFSET_SECONDS_PARAM,
    TEST_WINDOWS_PARAM,
    MAJOR_GAP_THRESHOLD_SECONDS_PARAM,
    NEAR_GAP_WINDOW_SECONDS_PARAM,
)

from src.gpx_to_csv import process_all_gpx_files
from src.find_optimal_match_window import format_results_table
from src.compare_data import process_location_pair, find_matching_files
from src.quality_analyzer import GPSQualityAnalyzer
from src.flag_comparison import run_for_all_files as flag_all_comparison_files
from src.unified_dataset import run_step_6_create_unified_dataset
from src.waypoint_distance_analyzer import run_waypoint_analysis
from src.device_accuracy_analyzer import run_device_accuracy_analysis
from src.location_accuracy_analyzer import run_location_accuracy_analysis
from src.cloud_accuracy_analyzer import run_cloud_accuracy_analysis
from src.phone_watch_agreement_analyzer import run_phone_watch_agreement_analysis
from src.visualization_analyzer import create_visualizations
from src.helpers.print_helper import PipelineMessages

from src.helpers.file_helper import discover_location_file_pairs
from src.helpers.window_helper import test_windows_for_pair, print_window_recommendations


def step_1_parse_gpx_files() -> bool:
    """Step 1: Parse all GPX files to CSV format."""
    PipelineMessages.step1_start()

    if not WATCH_DIR.exists():
        PipelineMessages.step1_error_watch_dir()
        return False

    PipelineMessages.step1_processing(str(WATCH_DIR.resolve()))
    summary = process_all_gpx_files(WATCH_DIR)

    if summary.get("total", 0) == 0:
        PipelineMessages.step1_no_gpx()
        return False

    PipelineMessages.step1_found_files(summary["total"])

    for i, result in enumerate(summary.get("results", []), 1):
        status = "✓" if result.get("success") else "✗"
        input_name = Path(result.get("input_file", "")).name if result.get("input_file") else "Unknown"

        if result.get("success"):
            output_name = Path(result.get("output_file", "")).name if result.get("output_file") else "Unknown"
            PipelineMessages.step1_file_result(
                i, summary["total"], status, input_name, output_name=output_name, trackpoints=result.get("trackpoints")
            )
        else:
            PipelineMessages.step1_file_result(
                i, summary["total"], status, input_name, error_msg=result.get("error", "Unknown error")
            )

    PipelineMessages.step1_summary(summary.get("total", 0), summary.get("successful", 0), summary.get("failed", 0))
    return summary.get("failed", 1) == 0


def step_2_find_optimal_window() -> Dict[Tuple[Path, Path], Dict]:
    PipelineMessages.step2_start()

    if not SMARTPHONE_DIR.exists() or not WATCH_DIR.exists():
        PipelineMessages.step2_error_dirs()
        return {}

    PipelineMessages.step2_testing_windows()

    # 1.1.1: Discover file pairs
    location_file_pairs = discover_location_file_pairs()
    if not location_file_pairs:
        PipelineMessages.step2_no_pairs()
        return {}

    all_pairs = []
    for location_name, pairs in location_file_pairs.items():
        PipelineMessages.step2_location(location_name)
        all_pairs.extend(pairs)

    # 1.1.2: Test each pair → richer results
    all_results = []
    file_pair_data: Dict[Tuple[Path, Path], Dict] = {}
    
    for smartphone_file, watch_file in all_pairs:
        parts = smartphone_file.stem.split("_")
        location = parts[2] if len(parts) >= 3 else "unknown"
        number = parts[3] if len(parts) >= 4 else "5"

        PipelineMessages.step2_testing_pair(location, number)

        result = test_windows_for_pair(smartphone_file, watch_file)
        
        if "error" not in result:
            all_results.append(result)
            file_pair_data[(smartphone_file, watch_file)] = result

            lines = format_results_table(result).split("\n")
            for line in lines:
                if line.strip():
                    print(f" {line}")

    if not all_results:
        PipelineMessages.step2_no_pairs()
        return {}

    # 1.1.3: Print recommendations
    print_window_recommendations(all_results)

    return file_pair_data 

def step_3_run_comparison(file_pair_data: Dict[Tuple[Path, Path], Dict]) -> bool:
    PipelineMessages.step3_start()

    if not file_pair_data:
        PipelineMessages.step3_error_no_window()
        return False

    if not SMARTPHONE_DIR.exists() or not WATCH_DIR.exists():
        PipelineMessages.step3_error_dirs()
        return False

    print()
    print("=" * 80)
    print("MERGE CONFIGURATION: Per-File-Pair Optimal Windows")
    print("=" * 80)

    for (spfile, _), data in sorted(file_pair_data.items()):
        print(f" {spfile.name:<35} {data['optimal_window']:.1f}s  {data['best_match_rate']:>5.1f}%")

    print("=" * 80)
    print()

    PipelineMessages.step3_processing_pairs(len(file_pair_data))

    all_pairs = list(file_pair_data.keys())
    successful = 0
    rejected = 0
    failed = 0

    for i, (spfile, watchfile) in enumerate(sorted(all_pairs), 1):
        pair_data = file_pair_data[(spfile, watchfile)]

        if pair_data.get('best_match_rate', 0) < 80:
            print(f"⚠️ Skipping {pair_data['location']}_{pair_data['session']}: {pair_data['best_match_rate']:.1f}% match rate too low")
            rejected += 1
            continue

        optimal_window = pair_data['optimal_window']
        result = process_location_pair(
            spfile, watchfile, RESULTS_DIR,
            time_window_seconds=optimal_window,
            allowed_offset_seconds=ALLOWED_OFFSET_SECONDS_PARAM,
        )

        if result.get("success"):
            match_rate = (
                (result["merged_count"] / result["smartphone_count"] * 100)
                if result.get("smartphone_count", 0) > 0 else 0
            )
            PipelineMessages.step3_pair_success(
                i, len(all_pairs), f"{pair_data['location']}_{pair_data['session']}",
                result.get("merged_count", 0), result.get("smartphone_count", 0), match_rate
            )
            successful += 1
        elif result.get("error") and "exceeds" in str(result.get("error")):
            PipelineMessages.step3_pair_rejected(
                i, len(all_pairs), f"{pair_data['location']}_{pair_data['session']}", result.get("time_offset")
            )
            rejected += 1
        else:
            PipelineMessages.step3_pair_error(i, len(all_pairs), f"{pair_data['location']}_{pair_data['session']}")
            failed += 1

    PipelineMessages.step3_summary(successful, rejected, failed, str(RESULTS_DIR.resolve()))
    return failed == 0


def step_4_run_quality_analysis() -> bool:
    PipelineMessages.step4_start()
    if not RESULTS_DIR.exists():
        PipelineMessages.step4_error_results_dir()
        return False
    try:
        analyzer = GPSQualityAnalyzer(
            results_dir=str(RESULTS_DIR),
            reports_dir=str(RESULTS_DIR.parent / "reports"),
        )
        analyzer.run_for_all_files()
        reports_path = RESULTS_DIR.parent / "reports"
        PipelineMessages.step4_summary_success(str(reports_path.resolve()))
        return True
    except Exception as e:
        PipelineMessages.step4_error(str(e))
        return False


def step_5_flag_comparison_data() -> bool:
    PipelineMessages.step5_start()
    if not RESULTS_DIR.exists():
        PipelineMessages.step5_error_results_dir()
        return False
    try:
        summary = flag_all_comparison_files(
            major_gap_threshold=MAJOR_GAP_THRESHOLD_SECONDS_PARAM,
            near_gap_window=NEAR_GAP_WINDOW_SECONDS_PARAM,
        )
        if summary.get("total", 0) == 0:
            print(" No comparison files found to flag")
            return True
        PipelineMessages.step5_processing_files(summary["total"])
        for idx, result in enumerate(summary.get("results", []), 1):
            if result.get("success"):
                PipelineMessages.step5_file_success(
                    idx, summary["total"], result.get("input_file", ""),
                    result.get("cache_duplicates", 0), result.get("near_major_gap", 0)
                )
            else:
                PipelineMessages.step5_file_error(
                    idx, summary["total"], result.get("input_file", ""),
                    result.get("error", "Unknown error")
                )
        PipelineMessages.step5_summary_success(
            summary.get("successful", 0), summary.get("total", 0), str(RESULTS_DIR.resolve())
        )
        return summary.get("failed", 1) == 0
    except Exception as e:
        PipelineMessages.step5_error(str(e))
        return False


def step_6_create_unified_dataset() -> bool:
    PipelineMessages.step6_start()
    try:
        return bool(run_step_6_create_unified_dataset())
    except Exception as e:
        PipelineMessages.step6_error(str(e))
        return False


def step_7_waypoint_analysis() -> bool:
    PipelineMessages.step7_start()
    success = run_waypoint_analysis()
    print()
    return bool(success)


def step_8_device_accuracy_analysis() -> bool:
    PipelineMessages.step8_start()
    success = run_device_accuracy_analysis()
    print()
    return bool(success)


def step_9_location_accuracy_analysis() -> bool:
    PipelineMessages.step9_start()
    success = run_location_accuracy_analysis()
    print()
    return bool(success)


def step_10_cloud_accuracy_analysis() -> bool:
    PipelineMessages.step10_start()
    success = run_cloud_accuracy_analysis()
    print()
    return bool(success)


def step_11_phone_watch_agreement() -> bool:
    PipelineMessages.step11_start()
    success = run_phone_watch_agreement_analysis()
    print()
    return bool(success)


def step_12_visualizations() -> bool:
    PipelineMessages.step12_start()
    success = create_visualizations()
    print()
    return bool(success)


def _extract_location_number(file_path: Path) -> Tuple[str, str]:
    stem = file_path.stem
    parts = stem.replace("ground_truth_", "").split("_")
    location = parts[0]
    number = parts[1] if len(parts) > 1 else "0"
    return location, number


def main() -> int:
    """Run the complete pipeline."""
    PipelineMessages.pipeline_start_sequence()

    success1 = step_1_parse_gpx_files()
    print()

    file_pair_data = step_2_find_optimal_window()
    if not file_pair_data:
        print(" Could not determine optimal windows for any file pairs - stopping")
        return 1
    print()

    success3 = step_3_run_comparison(file_pair_data)
    print()

    success4 = step_4_run_quality_analysis() if success3 else False
    print()
    success5 = step_5_flag_comparison_data() if success3 else False
    print()
    success6 = step_6_create_unified_dataset() if success3 else False
    print()
    success7 = step_7_waypoint_analysis() if success6 else False
    print()
    success8 = step_8_device_accuracy_analysis() if success7 else False
    print()
    success9 = step_9_location_accuracy_analysis() if success8 else False
    print()
    success10 = step_10_cloud_accuracy_analysis() if success9 else False
    print()
    success11 = step_11_phone_watch_agreement() if success10 else False
    print()
    success12 = step_12_visualizations() if success11 else False
    print()

    # Summary with richer Step 2 info
    PipelineMessages.pipeline_summary_header()
    PipelineMessages.pipeline_step_complete("1", "Parse GPX files → CSV", success1)
    
    avg_match = sum(data.get('best_match_rate', 0) for data in file_pair_data.values()) / len(file_pair_data)
    PipelineMessages.pipeline_step_complete(
        "2", "Find optimal time windows", True, f"{len(file_pair_data)} pairs @ {avg_match:.1f}%"
    )
    
    PipelineMessages.pipeline_step_complete("3", "Merge smartphone + watch data (optimized)", success3)
    PipelineMessages.pipeline_step_complete("4", "Analyze merged data quality", success4)
    PipelineMessages.pipeline_step_complete("5", "Flag comparison data quality issues", success5)
    PipelineMessages.pipeline_step_complete("6", "Create unified GPS dataset", success6)
    PipelineMessages.pipeline_step_complete("7", "Waypoint distance analysis", success7)
    PipelineMessages.pipeline_step_complete("8", "Global device accuracy metrics (CEP/R95)", success8)
    PipelineMessages.pipeline_step_complete("9", "GPS accuracy by location", success9)
    PipelineMessages.pipeline_step_complete("10", "GPS accuracy by cloud coverage", success10)
    PipelineMessages.pipeline_step_complete("11", "Phone-Watch agreement analysis", success11)
    PipelineMessages.pipeline_step_complete("12", "Thesis visualizations (PDF figures)", success12)

    if (success3 and success4 and success5 and success6 and success7 and 
        success8 and success9 and success10 and success11 and success12):
        PipelineMessages.pipeline_success()
        return 0

    PipelineMessages.pipeline_partial_failure()
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
