from src.config.parameters import (
    PHONE_WATCH_DISTANCE_THRESHOLD
)

class PipelineMessages:
    """Static class with methods for all pipeline messages."""

    # ========== STEP 1: Parse GPX files ==========

    @staticmethod
    def step1_start():
        print("\n" + "=" * 80)
        print("STEP 1: Parse GPX files to CSV")
        print("=" * 80)

    @staticmethod
    def step1_error_watch_dir():
        print("❌ ERROR: WATCH directory not found")

    @staticmethod
    def step1_processing(path: str):
        print(f"📂 Processing: {path}")

    @staticmethod
    def step1_no_gpx():
        print("⚠️  No GPX files found in WATCH directory")

    @staticmethod
    def step1_found_files(count: int):
        print(f"✓ Found {count} GPX files\n")

    @staticmethod
    def step1_file_result(i: int, total: int, status: str, input_name: str,
                          output_name: str = None, trackpoints: int = None, error_msg: str = None):
        if error_msg:
            print(f" {status} [{i}/{total}] {input_name}")
            print(f"       Error: {error_msg}")
        else:
            print(f" {status} [{i}/{total}] {input_name} → {output_name} ({trackpoints} trackpoints)")

    @staticmethod
    def step1_summary(total: int, successful: int, failed: int):
        print(f"\n✓ Step 1 complete: {successful}/{total} files processed successfully")
        if failed > 0:
            print(f"⚠️  {failed} file(s) failed")

    # ========== STEP 2: Find optimal time windows ==========

    @staticmethod
    def step2_start():
        print("\n" + "=" * 80)
        print("STEP 2: Find optimal time windows (per file pair)")
        print("=" * 80)

    @staticmethod
    def step2_error_dirs():
        print("❌ ERROR: SMARTPHONE or WATCH directory not found")

    @staticmethod
    def step2_testing_windows():
        print("🔍 Testing multiple time windows for each file pair...")

    @staticmethod
    def step2_location(location: str):
        print(f"\n📍 Location: {location}")

    @staticmethod
    def step2_testing_pair(location: str, number: str):
        print(f" Testing {location}_{number}:")

    @staticmethod
    def step2_no_pairs():
        print("⚠️  No matching smartphone/watch pairs found")

    @staticmethod
    def step2_recommendations_header():
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)

    @staticmethod
    def step2_recommended_window(window: float, votes: int, total: int):
        percentage = (votes / total * 100) if total > 0 else 0
        print(f"Most common optimal window: {window:.1f}s ({votes}/{total} datasets = {percentage:.0f}%)")

    @staticmethod
    def step2_recommendations_by_dataset_header():
        print("\nPer-dataset recommendations:")

    @staticmethod
    def step2_dataset_recommendation(sp_file: object, window: float, rate: float):
        filename = sp_file.name if hasattr(sp_file, 'name') else str(sp_file)
        print(f" • {filename:<40} → {window:.1f}s window (match rate: {rate:.1f}%)")

    @staticmethod
    def step2_footer():
        print("-" * 80)

    # ========== STEP 3: Merge smartphone and watch data ==========

    @staticmethod
    def step3_start():
        print("\n" + "=" * 80)
        print("STEP 3: Merge smartphone & watch data (using optimal windows)")
        print("=" * 80)

    @staticmethod
    def step3_error_no_window():
        print("❌ ERROR: No optimal windows available")

    @staticmethod
    def step3_error_dirs():
        print("❌ ERROR: SMARTPHONE or WATCH directory not found")

    @staticmethod
    def step3_processing_pairs(count: int):
        print(f"🔗 Processing {count} file pair(s)...\n")

    @staticmethod
    def step3_pair_success(i: int, total: int, identifier: str, merged: int, sp_count: int, match_rate: float):
        print(f" ✓ [{i}/{total}] {identifier}: {merged}/{sp_count} matched ({match_rate:.1f}%)")

    @staticmethod
    def step3_pair_rejected(i: int, total: int, identifier: str, offset: str):
        print(f" ⚠️  [{i}/{total}] {identifier}: REJECTED (offset {offset} exceeds ±1h)")

    @staticmethod
    def step3_pair_error(i: int, total: int, identifier: str):
        print(f" ✗ [{i}/{total}] {identifier}: ERROR")

    @staticmethod
    def step3_summary(successful: int, rejected: int, failed: int, output_path: str):
        print(f"\n✓ Step 3 complete: {successful} successful, {rejected} rejected, {failed} failed")
        print(f"📁 Output: {output_path}")

    # ========== STEP 4: Analyze quality ==========

    @staticmethod
    def step4_start():
        print("\n" + "=" * 80)
        print("STEP 4: Analyze merged data quality")
        print("=" * 80)

    @staticmethod
    def step4_error_results_dir():
        print("❌ ERROR: Results directory not found")

    @staticmethod
    def step4_summary_success(reports_path: str):
        print(f"✓ Step 4 complete: Quality analysis finished")
        print(f"📁 Reports: {reports_path}")

    @staticmethod
    def step4_error(error: str):
        print(f"❌ Step 4 ERROR: {error}")

    # ========== STEP 5: Flag comparison data ==========

    @staticmethod
    def step5_start():
        print("\n" + "=" * 80)
        print("STEP 5: Flag comparison data quality issues")
        print("=" * 80)

    @staticmethod
    def step5_error_results_dir():
        print("❌ ERROR: Results directory not found")

    @staticmethod
    def step5_processing_files(count: int):
        print(f"🚩 Flagging {count} comparison file(s)...\n")

    @staticmethod
    def step5_file_success(i: int, total: int, filename: str, cache_dups: int, near_gap: int):
        print(f" ✓ [{i}/{total}] {filename:<35} (cache_dup: {cache_dups}, near_gap: {near_gap})")

    @staticmethod
    def step5_file_error(i: int, total: int, filename: str, error: str):
        print(f" ✗ [{i}/{total}] {filename}")
        print(f"       Error: {error}")

    @staticmethod
    def step5_summary_success(successful: int, total: int, output_path: str):
        print(f"\n✓ Step 5 complete: {successful}/{total} files flagged successfully")
        print(f"📁 Output: {output_path}")

    @staticmethod
    def step5_error(error: str):
        print(f"❌ Step 5 ERROR: {error}")

    # ========== Pipeline summary ==========

    @staticmethod
    def pipeline_start_sequence():
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "GPS GROUND TRUTH COMPARISON PIPELINE".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")

    @staticmethod
    def pipeline_summary_header():
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)

    @staticmethod
    def pipeline_step_complete(step_num: int, description: str, success: bool, detail: str = None):
        status = "✓" if success else "✗"
        print(f"{status} Step {step_num}: {description}")
        if detail:
            print(f"           {detail}")

    @staticmethod
    def pipeline_success():
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + "✓ PIPELINE COMPLETED SUCCESSFULLY".center(78) + "║")
        print("╚" + "=" * 78 + "╝\n")

    @staticmethod
    def pipeline_partial_failure():
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + "⚠️  PIPELINE COMPLETED WITH ERRORS".center(78) + "║")
        print("╚" + "=" * 78 + "╝\n")
    # ========== STEP 6: Create unified dataset ==========

    @staticmethod
    def step6_start():
        print("\n" + "=" * 80)
        print("STEP 6: Create unified GPS dataset")
        print("=" * 80)

    @staticmethod
    def step6_summary(waypoints_count: int, gps_count: int, total_records: int, output_file: str):
        print(f"✓ Step 6 complete: Unified dataset created")
        print(f"  • Waypoints loaded: {waypoints_count}")
        print(f"  • GPS observations loaded: {gps_count}")
        print(f"  • Total unified records: {total_records}")
        print(f"📁 Output: {output_file}")

    @staticmethod
    def step6_error(error: str):
        print(f"❌ Step 6 ERROR: {error}")

    # ========== STEP 7: Waypoint distance analysis ==========
    @staticmethod
    def step7_start():
        print("\n" + "=" * 80)
        print("STEP 7: Waypoint distance analysis")
        print("=" * 80)

    @staticmethod
    def step7_error_waypoints():
        print("❌ ERROR: waypoints file not found")

    @staticmethod
    def step7_error_unified():
        print("❌ ERROR: unified dataset not found")

    @staticmethod
    def step7_summary_success(count: int, reports_path: str):
        print(f"✓ Step 7 complete: {count} session reports generated")
        print(f"📁 Reports: {reports_path}")

    @staticmethod
    def step7_error(error: str):
        print(f"❌ Step 7 ERROR: {error}")

    # ========== STEP 8: Device accuracy analysis ==========
    @staticmethod
    def step8_start():
        print("\n" + "=" * 80)
        print("STEP 8: Global device accuracy analysis (CEP, R95, RMSE)")
        print("=" * 80)

    @staticmethod
    def step8_no_data():
        print("❌ ERROR: No waypoint distance data available")
        print("   Run Steps 6-7 first")

    @staticmethod
    def step8_summary_success(sp_samples: int, watch_samples: int, output_file: str):
        print(f"✓ Step 8 complete: Accuracy metrics calculated")
        print(f" • Smartphone samples: {sp_samples}")
        print(f" • Watch samples:      {watch_samples}")
        print(f"📁 Output: {output_file}")

    @staticmethod
    def step8_error(error: str):
        print(f"❌ Step 8 ERROR: {error}")

    # ========== STEP 9: Location accuracy analysis ==========
    @staticmethod
    def step9_start():
        print("\n" + "=" * 80)
        print("STEP 9: GPS accuracy analysis by location")
        print("=" * 80)

    @staticmethod
    def step9_no_locations():
        print("❌ ERROR: No location directories found in data/SMARTPHONE/")

    @staticmethod
    def step9_discovered_locations(count: int, locations: list):
        print(f"📍 Discovered {count} locations: {', '.join(locations)}")

    @staticmethod
    def step9_location_no_data(location: str):
        print(f" ⚠️  {location}: No valid waypoint matches")

    @staticmethod
    def step9_summary_success(location_count: int, output_file: str, results: list):
        print(f"✓ Step 9 complete: {location_count} locations analyzed")
        print(f"📁 Output: {output_file}")
        for result in results:
            print(f" • {result['location']}: {result['smartphone_samples']} Smartphone, {result['watch_samples']} Watch samples")

    # ========== STEP 10: Cloud coverage accuracy analysis ==========
    @staticmethod
    def step10_start():
        print("\n" + "=" * 80)
        print("STEP 10: GPS accuracy analysis by cloud coverage")
        print("=" * 80)

    @staticmethod
    def step10_no_categories():
        print("❌ ERROR: No cloud_coverage values found in waypoints")

    @staticmethod
    def step10_discovered_categories(count: int, categories: list):
        print(f"☁️  Discovered {count} cloud categories: {', '.join(categories)}")

    @staticmethod
    def step10_category_no_data(category: str):
        print(f" ⚠️  Cloud {category}: No valid waypoint matches")

    @staticmethod
    def step10_summary_success(category_count: int, output_file: str, results: list):
        print(f"✓ Step 10 complete: {category_count} cloud categories analyzed")
        print(f"📁 Output: {output_file}")
        for result in results:
            print(f" • Cloud {result['category']}: {result['smartphone_samples']} Smartphone, {result['watch_samples']} Watch samples")

    # ========== STEP 11: Phone-Watch agreement analysis ==========
    @staticmethod
    def step11_start():
        print("\n" + "=" * 80)
        print("STEP 11: Phone-Watch GPS consistency analysis")
        print("=" * 80)

    @staticmethod
    def step11_error_no_unified():
        print("❌ ERROR: unified_gps_dataset.csv not found")

    @staticmethod
    def step11_error_no_waypoints():
        print("❌ ERROR: ground_truth_waypoints.csv not found")

    @staticmethod
    def step11_loading_data(waypoint_count: int):
        print(f"📍 Loading {waypoint_count} waypoints for ground truth matching")

    @staticmethod
    def step11_no_valid_data():
        print("❌ No valid phone/watch coordinate pairs found")

    @staticmethod
    def step11_summary_success(total: int, agreement_rate: float, mean_dist: float, output_file: str):
        print(f"✓ Step 11 complete: {total} timestamps analyzed")
        print(f"   Agreement ≤{PHONE_WATCH_DISTANCE_THRESHOLD}m: {agreement_rate}%")
        print(f"   Mean phone-watch distance: {mean_dist}m")
        print(f"📁 Output: {output_file}")

    # ========== STEP 12: Visualization generation ==========
    @staticmethod
    def step12_start():
        print("\n" + "=" * 80)
        print("STEP 12: Thesis visualization generation")
        print("=" * 80)

    @staticmethod
    def step12_complete(fig_count: int, output_dir: str):
        print(f"✓ Step 12 complete: {fig_count} publication-ready figures generated")
        print(f"📁 Output: {output_dir}")
        print("   Formats: figure_01_lln_dominance.pdf, figure_02_device_comparison.pdf, etc.")