import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))


try:
    from helpers.helpers_config import Config
    from helpers.helpers_output import (
        print_step, print_ok, print_error,
        print_pipeline_start, print_pipeline_complete,
        save_parameter_report,
        create_json_parameter_metadata,
    )
    from helpers.helpers_validation import (
        validate_required_files, validate_output_file, get_csv_row_count,
    )
except ImportError as e:
    print(f"Error importing helpers: {e}")
    print("Make sure src/helpers/ directory exists with helper modules")
    sys.exit(1)


try:
    from config.algorithm_params import get_algorithm_params
except ImportError as e:
    print(f"Error importing algorithm_params: {e}")
    sys.exit(1)

try:
    from parse_logs import parse_network_logs
    from verify_coverage import verify_database_coverage
    from extract_signal_strength import extract_signal_strength
    from estimate_distance import estimate_distances
    from estimate_bearing import estimate_bearings
    from prepare_trilateration import TrilaterationInputPreparer
    from trilateration_solver import TrilaterationSolver
    from ground_validation import GroundTruthValidator
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    sys.exit(1)

def main() -> None:
    """Main pipeline orchestrator with context and formula support."""
    
    # Validate command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py [context] [subcontext]")
        print("")
        print("Arguments:")
        print(" log_file: Input log file (e.g., lln_1.txt)")
        print(" [context] Environment context:")
        print(" - city, village, town, default: Path loss models")
        print(" - formula: Regression-based distance estimation (NEW!)")
        print(" If not specified, uses 'default'")
        print(" [subcontext] Bearing/trilateration context (only for formula mode):")
        print(" - city, village, town, default")
        print(" If not specified with formula, uses 'default'")
        print("")
        print("Examples:")
        print(" python main.py lln_1.txt # Default context")
        print(" python main.py lln_2.txt city # City context")
        print(" python main.py waha_1.txt village # Village context")
        print(" python main.py msg_1.txt town # Town context")
        print(" python main.py lln_1.txt formula # Formula + default (NEW!)")
        print(" python main.py lln_5.txt formula town # Formula + town (NEW!)")
        sys.exit(1)

    log_file = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else "default"
    subcontext = sys.argv[3] if len(sys.argv) > 3 else None

    # FORMULA CONTEXT: Load coefficients from JSON
    if context.lower() == "formula":
        # Extract prefix from log filename
        log_path = Path(log_file)
        log_stem = log_path.stem # e.g., "lln_1"
        prefix = log_stem.split('_')[0] # e.g., "lln"

        # Default subcontext if not provided
        if not subcontext:
            subcontext = "default"

        print("=" * 70)
        print(f"FORMULA-BASED DISTANCE ESTIMATION")
        print("=" * 70)
        print(f"Log file: {log_file}")
        print(f"Detected prefix: {prefix}")
        print(f"Bearing/trilateration context: {subcontext.upper()}")
        print("")

        try:
            formula_data = Config.load_formula_coefficients(prefix)
            print(f"✓ Loaded formula coefficients for prefix: {prefix}")
            print(f" Model: {formula_data.get('model')}")
            print(f" Features: {', '.join(formula_data.get('features', []))}")
            perf = formula_data.get('performance_metrics', {})
            print(f" Training MAE: {perf.get('mae_mean_m', 0):.1f}m")
            print(f" Training RMSE: {perf.get('rmse_mean_m', 0):.1f}m")
            print(f" Training samples: {perf.get('train_samples', 0)}")
            print("")
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(1)
        except Exception as e:
            print(f" ERROR: Failed to load formula: {e}")
            sys.exit(1)

        # Get FormulaParams with loaded coefficients and subcontext
        algorithm_params = get_algorithm_params("formula", subcontext=subcontext, formula_data=formula_data)
        resolved_context = "formula"
        resolved_subcontext = subcontext
    else:
        # Regular context (city, village, town, default)
        formula_data = None
        algorithm_params = get_algorithm_params(context)
        resolved_context = context
        resolved_subcontext = None

    # Setup configuration using helper
    paths = Config.setup_for_log(log_file, context=resolved_context, subcontext=resolved_subcontext)

    # Override algorithm_params in paths
    paths["algorithm_params"] = algorithm_params
    paths["context"] = resolved_context
    paths["subcontext"] = resolved_subcontext

    # Validate required files using helper
    validate_required_files(paths)

    # Print start information using helper
    start_time = datetime.now().strftime('%H:%M:%S')

    print_pipeline_start(
        log_file=log_file,
        data_dir=paths['log_data_dir'].name,
        start_time=start_time,
        context=algorithm_params.context,
        params=algorithm_params
    )

    print(f"\n[INFO] Environment Context: {algorithm_params.context.upper()}")
    if algorithm_params:
        print(f"[INFO] Algorithm Parameters loaded for context: {algorithm_params.context}")
    if resolved_context == "formula":
        print(f"[INFO] Distance calculation method: REGRESSION FORMULA")
        print(f"[INFO] Bearing/Trilateration context: {resolved_subcontext.upper()}")
    else:
        print(f"[INFO] Distance calculation method: PATH LOSS MODEL")

    # ====================================================================
    # PHASE 1: Data Extraction
    # ====================================================================
    print_step("1/9", "Parsing network log (Phase 1.1)")
    try:
        parse_network_logs(
            input_file=str(paths["input_log"]),
            output_file=str(paths["parsed_csv"]),
        )
        print_ok(f"Output: {paths['parsed_csv'].name}")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    print_step("2/9", "Verifying database coverage (Phase 1.2)")
    try:
        verify_database_coverage(
            log_file=str(paths["parsed_csv"]),
            towers_file=str(paths["towers_db"]),
            report_file=str(paths["coverage_report"]),
            status_file=str(paths["towers_status"]),
        )
        print_ok(f"Report: {paths['coverage_report'].name}")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    # ====================================================================
    # PHASE 2: Signal Analysis
    # ====================================================================
    print_step("3/9", "Extracting signal strength data (Phase 2.1)")
    try:
        # For formula mode, pass combined context "formula_city" instead of just "formula"
        if resolved_context == "formula" and resolved_subcontext:
            extract_context = f"formula_{resolved_subcontext}"
        else:
            extract_context = resolved_context

        result_2_1 = extract_signal_strength(
            parsed_log_file=str(paths["parsed_csv"]),
            output_dir=str(paths["log_data_dir"]),
            base_name=paths["basename"],
            context=extract_context, # ← Pass combined context here
            verbose=False
        )

        if not result_2_1.get("success"):
            print_error(result_2_1.get("error", "Unknown error"))
            sys.exit(1)

        print_ok(f"Signal records: {result_2_1['signal_records']}")
        print_ok(f"Metadata records: {result_2_1['metadata_records']}")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    print_step("4/9", "Estimating distances from signal strength (Phase 2.2)")
    try:
        estimate_distances(
            signal_metadata_path=paths["signal_metadata_csv"],
            signal_data_path=paths["signal_data_csv"],
            neighbor_signals_path=paths["neighbor_signals_csv"],
            pci_path=paths["pci_db"],
            towers_path=paths["towers_db"],
            output_path=paths["distance_estimates_csv"],
            algorithm_params=algorithm_params,
        )

        if not validate_output_file(paths["distance_estimates_csv"], "Distance estimation"):
            sys.exit(1)

        row_count = get_csv_row_count(paths["distance_estimates_csv"])
        print_ok(f"Distance estimates: {row_count} rows")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    # PHASE 2.2.5: Distance Ground Truth Validation
    print_step("4.5/9", "Validating distance estimates vs ground truth (Phase 2.2.5)")
    try:
        from validate_distance_ground_truth import compute_distance_accuracy
        
        distance_accuracy_report = paths["report_dir"] / f"Accuracy_distance_cell_summary_{paths['basename']}_{paths['context']}.txt"
        
        metrics = compute_distance_accuracy(
            distance_estimates_path=paths["distance_estimates_csv"],
            ground_truth_path=paths["ground_truth_csv"],
            pci_path=paths["pci_db"],
            towers_path=paths["towers_db"],
            output_path=distance_accuracy_report,
            context=context
        )
        
        print_ok(f"Distance validation report: {distance_accuracy_report.name}")
        print_ok(f"Samples: {metrics.get('samples', 0)}")
        print_ok(f"MAE: {metrics.get('mae', 0):.1f}m")
        print_ok(f"RMSE: {metrics.get('rmse', 0):.1f}m")
    except Exception as e:
        print_error(f"Distance validation failed: {str(e)}")
        # Continue pipeline (non-blocking)

    print_step("5/9", "Estimating bearing angles (Phase 2.3)")
    try:
        estimate_bearings(
            distance_estimates_path=paths["distance_estimates_csv"],
            signal_data_path=paths["signal_data_csv"],
            signal_metadata_path=paths["signal_metadata_csv"],
            towers_path=paths["towers_db"],
            pci_path=paths["pci_db"],
            output_path=paths["bearing_estimates_csv"],
        )

        if not validate_output_file(paths["bearing_estimates_csv"], "Bearing estimation"):
            sys.exit(1)

        row_count = get_csv_row_count(paths["bearing_estimates_csv"])
        print_ok(f"Bearing estimates: {row_count} rows")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    # ====================================================================
    # PHASE 3: Geolocation Solving & Validation
    # ====================================================================
    print_step("6/9", "Preparing trilateration input (Phase 3.1)")
    try:
        preparer = TrilaterationInputPreparer(
            input_file=str(paths["distance_estimates_csv"]),
            output_file=str(paths["trilateration_input_csv"]),
            params=algorithm_params,
        )

        preparer.load_data()
        preparer.prepare_trilateration_input()
        preparer.save_output()

        summary = preparer.get_summary()
        print_ok(f"Output: {paths['trilateration_input_csv'].name}")
        print_ok(f"Rows: {summary['rows']} ({summary['timestamps']} timestamps)")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    print_step("7/9", "Solving device positions (Phase 3.2 - Weighted Trilateration)")
    try:
        solver = TrilaterationSolver(
            input_file=str(paths["trilateration_input_csv"]),
            output_file=str(paths["trilateration_results_csv"])
        )

        solver.load_data()
        solver.solve()

        tri_summary = solver.get_summary()
        print_ok(f"Output: {paths['trilateration_results_csv'].name}")
        print_ok(f"Positions: {tri_summary['total_positions']}")
        print_ok(f"Average GDOP: {tri_summary['gdop_mean']:.2f}")
        print_ok(f"Average residual: {tri_summary['residual_mean']:.1f}m")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    # ====================================================================
    # PHASE 3.3: Ground Truth Validation
    # ====================================================================
    print_step("8/9", "Validating against ground truth (Phase 3.3)")
    try:
        validator = GroundTruthValidator(
            results_file=str(paths["trilateration_results_csv"]),
            ground_truth_file=str(paths["ground_truth_csv"]) if paths["ground_truth_csv"] else None,
            base_name=paths["basename"],
            validation_output=str(paths["validation_results_csv"]),
            summary_output=str(paths["accuracy_summary_txt"]),
        )

        # NEW: Single line does it all (loads, validates, saves TXT, saves CSV)
        validation_df, val_summary = validator.run(context=paths["context"])

        print_ok(f"Validation: validation_results.csv")
        print_ok(f"Samples: {val_summary.get('samples', 0)}")
        print_ok(f"CEP (50th %ile): {val_summary.get('cep', 0):.1f}m")
        print_ok(f"R95 (95th %ile): {val_summary.get('r95', 0):.1f}m")
        print_ok(f"RMSE: {val_summary.get('rmse', 0):.1f}m")
        print_ok(f"Summary: accuracy_summary.txt")
        print_ok(f"Metrics CSV: accuracy_summary_metrics.csv")  # NEW
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    print_step("9/9", "Generating final reports and metadata")
    try:
        # ====================================================================
        # ENHANCED: APPEND PARAMETERS TO REPORT
        # ====================================================================
        accuracy_summary_file = Path(paths["accuracy_summary_txt"])

        # Append parameter report to accuracy summary
        save_parameter_report(
            output_file=accuracy_summary_file,
            params=algorithm_params,
            context=algorithm_params.context,
            append=True
        )

        # Optional: Save machine-readable JSON metadata
        metadata_dict = create_json_parameter_metadata(algorithm_params, algorithm_params.context)

        # Add distance method and subcontext info for formula mode
        if resolved_context == "formula":
            metadata_dict["distance_method"] = "formula"
            metadata_dict["bearing_trilateration_context"] = resolved_subcontext
        else:
            metadata_dict["distance_method"] = "path_loss"
            metadata_dict["bearing_trilateration_context"] = resolved_context

        metadata_file = paths["analysis_metadata_json"]
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2)

        print_ok(f"✓ Parameter report appended to accuracy_summary.txt")
        print_ok(f"✓ Machine-readable metadata saved to analysis_metadata.json")
    except Exception as e:
        print_error(f"Final report generation failed: {str(e)}")

    # ====================================================================
    # PIPELINE COMPLETION
    # ====================================================================
    end_time = datetime.now().strftime('%H:%M:%S')

    # Output files mapping
    output_files = {
        "Phase 2.2": "distance_estimates.csv",
        "Phase 2.2.5": f"Accuracy_distance_cell_summary_{paths['basename']}_{paths['context']}.txt",
        "Phase 2.3": "bearing_estimates.csv",
        "Phase 3.2": "trilateration_results.csv",
        "Phase 3.3": "validation_results.csv",
        "Accuracy": "accuracy_summary.txt",
    }

    print_pipeline_complete(
        end_time=end_time,
        data_dir=paths['log_data_dir'].name,
        report_dir=paths['report_dir'].name,
        output_files=output_files,
        context=algorithm_params.context
    )

    # Formula-specific summary
    if resolved_context == "formula":
        print("")
        print("=" * 70)
        print("FORMULA-BASED DISTANCE ESTIMATION SUMMARY")
        print("=" * 70)
        print(f"Formula prefix: {prefix}")
        print(f"Formula model: {formula_data.get('model')}")
        print(f"Bearing/Trilateration context: {resolved_subcontext.upper()}")
        print(f"Training MAE: {perf.get('mae_mean_m', 0):.1f}m")
        print(f"Validation CEP: {val_summary.get('cep', 0):.1f}m")
        print(f"Validation R95: {val_summary.get('r95', 0):.1f}m")
        print("=" * 70)

if __name__ == "__main__":
    main()
