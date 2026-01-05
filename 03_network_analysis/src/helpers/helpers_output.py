from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


def format_parameter_table(params: Optional[Any]) -> str:
    """
    Format algorithm parameters as a readable table.
    
    Args:
        params: AlgorithmParams object (or None)
    
    Returns:
        Formatted string table of key parameters
    """
    if params is None:
        return "No parameters loaded"
    
    lines = []
    lines.append("=" * 80)
    lines.append("ALGORITHM PARAMETERS")
    lines.append("=" * 80)
    
    # Context
    context = getattr(params, 'context', 'unknown')
    lines.append(f"\nContext: {context.upper()}")
    lines.append("-" * 80)
    
    # Phase 2.2: Distance Estimation
    lines.append("\nPHASE 2.2: DISTANCE ESTIMATION")
    
    # Distance method
    distance_method = getattr(params, 'distance_calculation_method', 'unknown')
    lines.append(f"  Distance Method: {distance_method.upper()}")
    
    rsrp_threshold = getattr(params, 'rsrp_quality_threshold_dbm', 'N/A')
    lines.append(f"  RSRP Quality Threshold: {rsrp_threshold} dBm")
    
    max_unc = getattr(params, 'max_uncertainty_by_level', {})
    if max_unc and 'MEDIUM' in max_unc:
        lines.append(f"  Max Uncertainty (MEDIUM): {max_unc['MEDIUM']:.1f} m")
    if max_unc and 'HIGH' in max_unc:
        lines.append(f"  Max Uncertainty (HIGH): {max_unc['HIGH']:.1f} m")
    
    # Phase 2.3: Bearing Estimation
    lines.append("\nPHASE 2.3: BEARING ESTIMATION")
    bearing_conf = getattr(params, 'bearing_confidence_thresholds', {})
    if bearing_conf:
        lines.append(f"  Bearing Confidence Points: Configured")
    
    # Phase 3.1: Trilateration Input
    lines.append("\nPHASE 3.1: TRILATERATION INPUT PREPARATION")
    min_cells = getattr(params, 'min_cells_required', 'N/A')
    max_cells = getattr(params, 'max_cells_to_keep', 'N/A')
    lines.append(f"  Min Cells Required: {min_cells}")
    lines.append(f"  Max Cells to Keep: {max_cells}")
    
    # Phase 3.2: Trilateration Solver
    lines.append("\nPHASE 3.2: TRILATERATION SOLVER")
    convergence = getattr(params, 'convergence_threshold_m', 'N/A')
    max_gdop = getattr(params, 'max_gdop_accepted', 'N/A')
    lines.append(f"  Convergence Threshold: {convergence} m")
    lines.append(f"  Max GDOP Accepted: {max_gdop}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def print_pipeline_start(
    log_file: str,
    data_dir: str,
    start_time: str,
    context: Optional[str] = None,
    params: Optional[Any] = None
) -> None:

    print("\n" + "=" * 80)
    print("GEOLOCATION ANALYSIS PIPELINE (PHASES 1-3.3)")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Data dir: {data_dir}")
    print(f"Started: {start_time}")
    if context:
        context_display = context.upper() if context != "default" else "DEFAULT"
        print(f"Context: {context_display}")
    print("=" * 80)
    
    # Print parameter summary
    if params is not None:
        print()
        print(format_parameter_table(params))


def print_step(step_num: str, description: str) -> None:
    print(f"\n[{step_num}] {description}...")


def print_ok(message: str) -> None:
    print(f"  ✓ {message}")


def print_error(message: str) -> None:
    print(f"  ✗ ERROR: {message}")


def print_warning(message: str) -> None:
    print(f"  ⚠ {message}")


def print_pipeline_complete(
    end_time: str,
    data_dir: str,
    report_dir: str,
    output_files: Dict[str, str],
    context: Optional[str] = None
) -> None:
    """
    Print pipeline completion banner.
    
    Args:
        end_time: End time string
        data_dir: Data directory name
        report_dir: Report directory name
        output_files: Dict of phase → output filename
        context: Environment context used
    """
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Completed: {end_time}")
    if context:
        context_display = context.upper() if context != "default" else "DEFAULT"
        print(f"Context: {context_display}")
    print()
    print(f"Output files saved to: {data_dir}/")
    print(f"Reports saved to: {report_dir}/")
    print()
    print("Key output files:")
    for phase, filename in output_files.items():
        print(f"  {phase}: {filename}")
    print("=" * 80 + "\n")


def create_parameter_report_section(
    params: Optional[Any],
    context: Optional[str] = None
) -> str:
    """
    Create a parameter report section for inclusion in output files.
    
    Returns a formatted string suitable for appending to CSV reports or text files.
    
    Args:
        params: AlgorithmParams object
        context: Environment context
    
    Returns:
        Formatted report section as string
    """
    if params is None:
        return ""
    
    lines = []
    
    # Header
    lines.append("\n" + "=" * 90)
    lines.append("ANALYSIS PARAMETERS")
    lines.append("=" * 90)
    
    # Metadata
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    context_display = context.upper() if context and context != "default" else "DEFAULT"
    lines.append(f"Context: {context_display}")
    
    # Distance method
    distance_method = getattr(params, 'distance_calculation_method', 'unknown')
    lines.append(f"Distance Method: {distance_method.upper()}")
    
    # Phase-specific parameters
    lines.append("\n" + "-" * 90)
    lines.append("PHASE 2.2: DISTANCE ESTIMATION")
    lines.append("-" * 90)
    
    rsrp_threshold = getattr(params, 'rsrp_quality_threshold_dbm', 'N/A')
    lines.append(f"RSRP Quality Threshold (dBm): {rsrp_threshold}")
    
    min_uncertainty = getattr(params, 'min_uncertainty_m', 'N/A')
    lines.append(f"Min Uncertainty (m): {min_uncertainty}")
    
    max_unc = getattr(params, 'max_uncertainty_by_level', {})
    if max_unc:
        lines.append("\nMax Uncertainty by Signal Level:")
        for level in ['EXCELLENT', 'HIGH', 'GOOD', 'MEDIUM', 'WEAK', 'VERY_WEAK']:
            if level in max_unc:
                lines.append(f"  {level:<12}: {max_unc[level]:.1f} m")
    
    # Phase 2.3
    lines.append("\n" + "-" * 90)
    lines.append("PHASE 2.3: BEARING ESTIMATION")
    lines.append("-" * 90)
    
    rsrq_scores = getattr(params, 'rsrq_score_thresholds', {})
    if rsrq_scores:
        lines.append("RSRQ Score Thresholds:")
        for threshold, points in sorted(rsrq_scores.items(), reverse=True):
            lines.append(f"  RSRQ ≥ {threshold:>3} dB: {points:>3} points")
    
    bearing_conf = getattr(params, 'bearing_confidence_thresholds', {})
    if bearing_conf:
        lines.append("\nBearing Confidence Thresholds: Configured")
    
    # Phase 3.1
    lines.append("\n" + "-" * 90)
    lines.append("PHASE 3.1: TRILATERATION INPUT PREPARATION")
    lines.append("-" * 90)
    
    min_cells = getattr(params, 'min_cells_required', 'N/A')
    max_cells = getattr(params, 'max_cells_to_keep', 'N/A')
    lines.append(f"Min Cells Required: {min_cells}")
    lines.append(f"Max Cells to Keep: {max_cells}")
    lines.append(f"RSRP Quality Threshold (dBm): {rsrp_threshold}")
    
    # Phase 3.2
    lines.append("\n" + "-" * 90)
    lines.append("PHASE 3.2: TRILATERATION SOLVER")
    lines.append("-" * 90)
    
    weight_power = getattr(params, 'weight_by_uncertainty_power', 'N/A')
    convergence = getattr(params, 'convergence_threshold_m', 'N/A')
    max_iter = getattr(params, 'max_iterations', 'N/A')
    max_gdop = getattr(params, 'max_gdop_accepted', 'N/A')
    residual_weight = getattr(params, 'residual_weight', 'N/A')
    
    lines.append(f"Weight by Uncertainty Power: {weight_power}")
    lines.append(f"Convergence Threshold (m): {convergence}")
    lines.append(f"Max Iterations: {max_iter}")
    lines.append(f"Max GDOP Accepted: {max_gdop}")
    lines.append(f"Residual Weight: {residual_weight}")
    
    # Footer
    lines.append("\n" + "=" * 90)
    
    return "\n".join(lines)


def create_json_parameter_metadata(
    params: Optional[Any],
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create JSON-compatible parameter metadata for machine-parsable reporting.
    
    Args:
        params: AlgorithmParams object
        context: Environment context
    
    Returns:
        Dictionary with parameter metadata
    """
    if params is None:
        return {}
    
    metadata = {
        "context": context or "default",
        "generated_at": datetime.now().isoformat(),
        "distance_calculation_method": getattr(params, 'distance_calculation_method', 'unknown'),
        "phase_2_2": {
            "rsrp_quality_threshold_dbm": getattr(params, 'rsrp_quality_threshold_dbm', None),
            "min_uncertainty_m": getattr(params, 'min_uncertainty_m', None),
            "max_uncertainty_by_level": getattr(params, 'max_uncertainty_by_level', {}),
        },
        "phase_2_3": {
            "rsrq_score_thresholds": getattr(params, 'rsrq_score_thresholds', {}),
            "bearing_confidence_thresholds": "Configured" if getattr(params, 'bearing_confidence_thresholds', {}) else None,
        },
        "phase_3_1": {
            "min_cells_required": getattr(params, 'min_cells_required', None),
            "max_cells_to_keep": getattr(params, 'max_cells_to_keep', None),
            "rsrp_quality_threshold_dbm": getattr(params, 'rsrp_quality_threshold_dbm', None),
        },
        "phase_3_2": {
            "weight_by_uncertainty_power": getattr(params, 'weight_by_uncertainty_power', None),
            "convergence_threshold_m": getattr(params, 'convergence_threshold_m', None),
            "max_iterations": getattr(params, 'max_iterations', None),
            "max_gdop_accepted": getattr(params, 'max_gdop_accepted', None),
            "residual_weight": getattr(params, 'residual_weight', None),
        },
    }
    
    return metadata


def save_parameter_report(
    output_file: Path,
    params: Optional[Any],
    context: Optional[str] = None,
    append: bool = False
) -> None:
    """
    Save parameter report to a text file.
    
    Args:
        output_file: Path to output file
        params: AlgorithmParams object
        context: Environment context
        append: If True, append to file; if False, overwrite
    """
    report_content = create_parameter_report_section(params, context)
    if not report_content:
        return
    
    mode = 'a' if append else 'w'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(report_content)


def print_context_comparison_summary(
    results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Print a comparison summary of results across different contexts.
    
    Args:
        results: Dict mapping context -> metrics dict
                 Each dict should have: 'positions', 'gdop_mean', 'residual_mean',
                                       'cep', 'r95', 'rmse'
    
    Example:
        results = {
            'city': {'positions': 5, 'gdop_mean': 263.59, 'residual_mean': 441.5, ...},
            'village': {'positions': 41, 'gdop_mean': 81.19, 'residual_mean': 1616.6, ...},
            'town': {'positions': 41, 'gdop_mean': 0.00, 'residual_mean': 1735.5, ...},
        }
        print_context_comparison_summary(results)
    """
    print("\n" + "=" * 100)
    print("CONTEXT COMPARISON SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"\n{'Metric':<30} {'CITY':<20} {'VILLAGE':<20} {'TOWN':<20}")
    print("-" * 100)
    
    # Extract all unique keys
    all_keys = set()
    for context_results in results.values():
        all_keys.update(context_results.keys())
    
    # Print each metric
    for key in sorted(all_keys):
        city_val = results.get('city', {}).get(key, 'N/A')
        village_val = results.get('village', {}).get(key, 'N/A')
        town_val = results.get('town', {}).get(key, 'N/A')
        
        # Format values
        if isinstance(city_val, float):
            city_str = f"{city_val:.2f}"
            village_str = f"{village_val:.2f}" if isinstance(village_val, float) else str(village_val)
            town_str = f"{town_val:.2f}" if isinstance(town_val, float) else str(town_val)
        else:
            city_str = str(city_val)
            village_str = str(village_val)
            town_str = str(town_val)
        
        print(f"{key:<30} {city_str:<20} {village_str:<20} {town_str:<20}")
    
    print("\n" + "=" * 100)
