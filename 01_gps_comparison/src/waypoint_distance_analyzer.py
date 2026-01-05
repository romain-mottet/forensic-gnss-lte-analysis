import statistics
from pathlib import Path
from src.helpers.waypoint_helper import get_session_waypoint_details
from src.config.parameters import WAYPOINT_TIME_WINDOW_SECONDS_PARAM


def format_session_report(session_data: Dict) -> str:
    """Format single session report - IDENTICAL to original output."""
    lines = []
    lines.append("=" * 85)
    lines.append(f"MEASUREMENT ACCURACY REPORT (≤{WAYPOINT_TIME_WINDOW_SECONDS_PARAM}s matches)")
    lines.append(f"Location: {session_data['location']} | Session: {session_data['session']}")
    lines.append("=" * 85)
    lines.append("")

    # Waypoint distances table - IDENTICAL format
    lines.append("[WAYPOINT DISTANCES]")
    lines.append(f"{'Waypoint':<15} {'ID':<4} {'Time Diff':<10} {'Smartphone':<12} {'Watch':<12}")
    lines.append("-" * 60)

    filtered_count = session_data.get('filtered_count', 0)
    total_waypoints = len(session_data.get('waypoints', [])) + filtered_count

    if filtered_count > 0:
        lines.append(f"* {filtered_count}/{total_waypoints} waypoints filtered (> {WAYPOINT_TIME_WINDOW_SECONDS_PARAM}s)")
        lines.append("")

    for wp in session_data['waypoints']:
        lines.append(
            f"{wp['waypoint_name']:<15} "
            f"{wp['waypoint_id']:<4} "
            f"{wp['time_diff_s']:<10.1f}s "
            f"{wp['smartphone_dist_m']:<12.1f}m "
            f"{wp['watch_dist_m']:<12.1f}m"
        )

    lines.append("")

    # Summary statistics - IDENTICAL format
    sp_dists = session_data['smartphone_distances']
    watch_dists = session_data['watch_distances']
    
    def format_stats(dists):
        if not dists:
            return "N/A", "N/A", "N/A", "N/A", "N/A"
        return (
            f"{statistics.mean(dists):.1f}",
            f"{min(dists):.1f}",
            f"{max(dists):.1f}",
            f"{statistics.median(dists):.1f}",
            f"{statistics.stdev(dists):.1f}" if len(dists) > 1 else "N/A"
        )

    sp_mean, sp_min, sp_max, sp_median, sp_stdev = format_stats(sp_dists)
    watch_mean, watch_min, watch_max, watch_median, watch_stdev = format_stats(watch_dists)

    lines.append("[SUMMARY STATISTICS]")
    lines.append(f"{'Device':<12} {'Mean':<8} {'Min':<8} {'Max':<8} {'Median':<8} {'Stdev':<8}")
    lines.append("-" * 60)
    lines.append(f"{'Smartphone':<12} {sp_mean:<8} {sp_min:<8} {sp_max:<8} {sp_median:<8} {sp_stdev:<8}")
    lines.append(f"{'Watch':<12} {watch_mean:<8} {watch_min:<8} {watch_max:<8} {watch_median:<8} {watch_stdev:<8}")
    lines.append("")
    lines.append(f"⚠️  FILTER: Only ≤{WAYPOINT_TIME_WINDOW_SECONDS_PARAM}s matches included ({filtered_count} filtered)")

    return "\n".join(lines)


def run_waypoint_analysis() -> bool:
    """Run Step 7: Waypoint distance analysis using waypoint_helper."""
    from src.config.parameters import DATA_DIR, RESULTS_DIR
    
    waypoints_file = DATA_DIR / "ground_truth_waypoints.csv"
    unified_file = RESULTS_DIR / "unified_gps_dataset.csv"
    
    if not waypoints_file.exists():
        print("❌ ERROR: waypoints file not found")
        return False
    
    if not unified_file.exists():
        print("❌ ERROR: unified dataset not found")
        return False
    
    try:
        # Get all unique location/session combinations
        locations = ['ixelle', 'lln', 'waha']
        sessions = [1, 2, 3, 4, 5]
        
        # Create reports directory structure
        reports_base = RESULTS_DIR.parent / "reports"
        reports_base.mkdir(exist_ok=True)
        
        successful = 0
        total_filtered = 0
        
        for location in locations:
            for session in sessions:
                session_results = get_session_waypoint_details(location, session)
                
                if session_results['sessions']:  # Only if data found
                    session_data = session_results['sessions'][0]
                    
                    report_path = (reports_base / location /
                                 f"measurement_{location}_{session}.txt")
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    report_text = format_session_report(session_data)
                    report_path.write_text(report_text, encoding='utf-8')
                    
                    successful += 1
                    total_filtered += session_data.get('filtered_count', 0)
        
        print(f"\n✓ Step 7 complete: {successful} session reports generated")
        print(f"📁 Reports: {reports_base.resolve()}")
        print(f"⏱️ Time window: ≤{WAYPOINT_TIME_WINDOW_SECONDS_PARAM}s matches only")
        if total_filtered > 0:
            print(f"🚫 {total_filtered} waypoints filtered (>{WAYPOINT_TIME_WINDOW_SECONDS_PARAM}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 7 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
