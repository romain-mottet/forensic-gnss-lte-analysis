import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from helpers.helpers_config import Config

# Available contexts
AVAILABLE_CONTEXTS = ['default', 'city', 'town', 'village']
AVAILABLE_SUBCONTEXTS = ['default', 'city', 'town', 'village']


def get_log_files(prefix: Optional[str] = None, log_name: Optional[str] = None) -> List[str]:
    if log_name:
        log_path = Config.RAW_LOGS_DIR / log_name
        if not log_path.exists():
            print(f" ERROR: Log file not found: {log_name}")
            sys.exit(1)
        return [log_name]
    
    logs_by_prefix = Config.list_logs_by_prefix()
    if not logs_by_prefix:
        print(f" ERROR: No log files found in {Config.RAW_LOGS_DIR}")
        sys.exit(1)
    
    if prefix:
        if prefix not in logs_by_prefix:
            print(f" ERROR: No logs found for prefix '{prefix}'")
            print(f"   Available prefixes: {', '.join(logs_by_prefix.keys())}")
            sys.exit(1)
        return logs_by_prefix[prefix]
    
    # Return all logs
    all_logs = []
    for prefix_logs in logs_by_prefix.values():
        all_logs.extend(prefix_logs)
    return sorted(all_logs)


def get_contexts(context_filter: Optional[str] = None) -> List[str]:
    if context_filter:
        if context_filter not in AVAILABLE_CONTEXTS:
            print(f" ERROR: Invalid context '{context_filter}'")
            print(f"   Available contexts: {', '.join(AVAILABLE_CONTEXTS)}")
            sys.exit(1)
        return [context_filter]
    
    return AVAILABLE_CONTEXTS


def get_subcontexts(subcontext_filter: Optional[str] = None) -> List[str]:
    if subcontext_filter:
        if subcontext_filter not in AVAILABLE_SUBCONTEXTS:
            print(f" ERROR: Invalid subcontext '{subcontext_filter}'")
            print(f"   Available subcontexts: {', '.join(AVAILABLE_SUBCONTEXTS)}")
            sys.exit(1)
        return [subcontext_filter]
    
    return AVAILABLE_SUBCONTEXTS


def check_formula_exists(log_file: str) -> bool:

    prefix = Config.extract_prefix(log_file)
    formulas_dir = Path(__file__).parent / "data" / "formulas"
    formula_file = formulas_dir / f"{prefix}_formula.json"
    return formula_file.exists()


def run_pipeline(log_file: str, context: str, subcontext: Optional[str] = None) -> Dict:
    start_time = datetime.now()
    
    # Build command
    cmd = [sys.executable, 'main.py', log_file, context]
    if subcontext:
        cmd.append(subcontext)
        display_context = f"{context}+{subcontext}"
    else:
        display_context = context
    
    print(f"\n{'='*80}")
    print(f"▶  Running: {log_file} with context={display_context}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n SUCCESS: {log_file} [{display_context}] completed in {duration:.1f}s")
            return {
                'log_file': log_file,
                'context': context,
                'subcontext': subcontext,
                'display_context': display_context,
                'status': 'SUCCESS',
                'duration': duration,
                'error': None
            }
        else:
            print(f"\n FAILED: {log_file} [{display_context}] (exit code: {result.returncode})")
            return {
                'log_file': log_file,
                'context': context,
                'subcontext': subcontext,
                'display_context': display_context,
                'status': 'FAILED',
                'duration': duration,
                'error': f"Exit code: {result.returncode}"
            }
    
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n ERROR: {log_file} [{display_context}] - {str(e)}")
        return {
            'log_file': log_file,
            'context': context,
            'subcontext': subcontext,
            'display_context': display_context,
            'status': 'ERROR',
            'duration': duration,
            'error': str(e)
        }


def aggregate_accuracy_summaries(report_dir: Path, output_file: str) -> None:
    """Aggregate all accuracy_summary_*.txt files into a single comprehensive summary."""
    print(f"\n Scanning {report_dir} for accuracy summaries...")
    
    summary_files = sorted(report_dir.rglob("accuracy_summary_*.txt"))
    
    if not summary_files:
        print(f"    No accuracy_summary_*.txt files found in {report_dir}")
        return
    
    output_path = report_dir / output_file
    
    with output_path.open("w", encoding="utf-8") as out:
        out.write("=" * 90 + "\n")
        out.write("BATCH ACCURACY SUMMARY - ALL CONTEXTS COMBINED\n")
        out.write("=" * 90 + "\n\n")
        
        for idx, summary_file in enumerate(summary_files, 1):
            rel_path = summary_file.relative_to(report_dir)
            
            out.write("=" * 80 + "\n")
            out.write(f"RUN {idx}/{len(summary_files)}: {rel_path}\n")
            out.write("=" * 80 + "\n\n")
            out.write(summary_file.read_text(encoding="utf-8"))
            out.write("\n" * 3)
        
        out.write("=" * 90 + "\n")
        out.write(f"AGGREGATED {len(summary_files)} ACCURACY SUMMARIES FROM {report_dir.name}\n")
        out.write("=" * 90 + "\n")
    
    print(f"    Aggregated {len(summary_files)} summaries → {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print summary of batch run results."""
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    total_time = sum(r['duration'] for r in results)
    
    print(f"\nTotal runs: {total}")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"    Errors: {errors}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per run: {total_time/total:.1f}s")
    
    if failed > 0 or errors > 0:
        print(f"\n{'='*80}")
        print("FAILED/ERROR RUNS:")
        print(f"{'='*80}")
        for r in results:
            if r['status'] in ['FAILED', 'ERROR']:
                print(f"  {r['status']}: {r['log_file']} [{r['display_context']}]")
                if r['error']:
                    print(f"    Error: {r['error']}")
    
    print(f"\n{'='*80}")


def main():
    """Main batch runner."""
    parser = argparse.ArgumentParser(
        description='Batch runner for geolocation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ALL logs with ALL contexts
  python batch_run.py
  
  # Regular contexts (current behavior)
  python batch_run.py --prefix lln                    # lln_* with city, town, village, default
  python batch_run.py --prefix lln --context city     # lln_* with city only
  python batch_run.py --log lln_5.txt                 # Single file, ALL contexts
  
  # Formula-only mode (NEW!)
  python batch_run.py --prefix lln --formula-only                  # lln_* with formula+all_subcontexts
  python batch_run.py --prefix lln --formula-only --subcontext city  # lln_* with formula+city only
  
  # Combined mode (NEW!)
  python batch_run.py --prefix lln --include-formula               # lln_* with ALL regular + ALL formula
  python batch_run.py --prefix lln --include-formula --subcontext city  # lln_* with ALL regular + formula+city
"""
    )
    
    parser.add_argument('--prefix', type=str, default=None,
                        help='Run only logs with this prefix (e.g., ixelle, lln, waha)')
    parser.add_argument('--context', type=str, default=None, choices=AVAILABLE_CONTEXTS,
                        help='Run only with this context (default, city, town, village)')
    parser.add_argument('--log', type=str, default=None,
                        help='Run only this specific log file (e.g., ixelle_4.txt)')
    parser.add_argument('--formula-only', action='store_true',
                        help='Run ONLY formula mode with subcontexts (skips regular contexts)')
    parser.add_argument('--include-formula', action='store_true',
                        help='Include formula variants in addition to regular contexts')
    parser.add_argument('--subcontext', type=str, default=None, choices=AVAILABLE_SUBCONTEXTS,
                        help='Filter subcontexts for formula mode (default, city, town, village)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without actually running')
    
    args = parser.parse_args()
    
    # Validate conflicting arguments
    if args.formula_only and args.include_formula:
        print(" ERROR: Cannot use --formula-only and --include-formula together")
        sys.exit(1)
    
    if args.subcontext and not (args.formula_only or args.include_formula):
        print(" ERROR: --subcontext can only be used with --formula-only or --include-formula")
        sys.exit(1)
    
    # Get log files
    log_files = get_log_files(prefix=args.prefix, log_name=args.log)
    
    # Build run configurations
    run_configs = []
    
    if args.formula_only:
        # Formula-only mode: only formula variants
        subcontexts = get_subcontexts(args.subcontext)
        
        for log_file in log_files:
            if not check_formula_exists(log_file):
                prefix = Config.extract_prefix(log_file)
                print(f"  WARNING: Skipping {log_file} - no formula found for prefix '{prefix}'")
                continue
            
            for subcontext in subcontexts:
                run_configs.append({
                    'log_file': log_file,
                    'context': 'formula',
                    'subcontext': subcontext
                })
    
    elif args.include_formula:
        # Combined mode: regular contexts + formula variants
        contexts = get_contexts(args.context)
        subcontexts = get_subcontexts(args.subcontext)
        
        for log_file in log_files:
            # Add regular contexts
            for context in contexts:
                run_configs.append({
                    'log_file': log_file,
                    'context': context,
                    'subcontext': None
                })
            
            # Add formula variants (only if formula exists)
            if check_formula_exists(log_file):
                for subcontext in subcontexts:
                    run_configs.append({
                        'log_file': log_file,
                        'context': 'formula',
                        'subcontext': subcontext
                    })
            else:
                prefix = Config.extract_prefix(log_file)
                print(f"  WARNING: Skipping formula for {log_file} - no formula found for prefix '{prefix}'")
    
    else:
        # Regular mode: only regular contexts (current behavior)
        contexts = get_contexts(args.context)
        
        for log_file in log_files:
            for context in contexts:
                run_configs.append({
                    'log_file': log_file,
                    'context': context,
                    'subcontext': None
                })
    
    # Print configuration
    print("="*80)
    print("BATCH RUN CONFIGURATION")
    print("="*80)
    print(f"Log files to process: {len(log_files)}")
    for log in log_files:
        print(f"  • {log}")
    
    print(f"\nRun configurations: {len(run_configs)}")
    for cfg in run_configs[:10]:  # Show first 10
        display = f"{cfg['context']}+{cfg['subcontext']}" if cfg['subcontext'] else cfg['context']
        print(f"  • {cfg['log_file']} [{display}]")
    if len(run_configs) > 10:
        print(f"  ... and {len(run_configs) - 10} more")
    
    print(f"\nTotal runs: {len(run_configs)}")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN] - No actual execution")
        return
    
    response = input("\nProceed with batch run? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print(" Batch run cancelled")
        return
    
    # Execute runs
    start_time = datetime.now()
    results = []
    
    for i, cfg in enumerate(run_configs, 1):
        print(f"\n{'#'*80}")
        print(f"RUN {i}/{len(run_configs)}")
        print(f"{'#'*80}")
        
        result = run_pipeline(cfg['log_file'], cfg['context'], cfg['subcontext'])
        results.append(result)
    
    end_time = datetime.now()
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*80)
    print("AGGREGATING ACCURACY SUMMARIES")
    print("="*80)
    
    aggregated = 0
    reports_root = Config.REPORTS_DIR 
    
    for prefix_reports_dir in reports_root.glob("*_reports"):
        if prefix_reports_dir.is_dir():
            for basename_reports_dir in prefix_reports_dir.glob("*_reports"):
                if basename_reports_dir.is_dir():
                    summary_files = list(basename_reports_dir.rglob("accuracy_summary_*.txt"))
                    
                    if summary_files:
                        output_file = f"batch_accuracy_summary_{basename_reports_dir.name}.txt"
                        aggregate_accuracy_summaries(basename_reports_dir, output_file)
                        aggregated += 1
                        print(f"    Found {len(summary_files)} summaries in {basename_reports_dir.name} (including formula)")

    if aggregated == 0:
        print("     No accuracy_summary_*.txt files found in reports/ directories")
    else:
        print(f"\n SUCCESS: Aggregated {aggregated} report directories with accuracy summaries!")
    
    # Save batch run summary
    batch_report_dir = Path('reports') / 'batch_report'
    batch_report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_part = args.log.replace('.txt', '') if args.log else (args.prefix or 'ALL')
    
    if args.formula_only:
        mode_part = 'formula_only'
    elif args.include_formula:
        mode_part = 'with_formula'
    else:
        mode_part = 'regular'
    
    context_part = args.context if args.context else 'ALL'
    subcontext_part = args.subcontext if args.subcontext else 'ALL'
    
    summary_file = batch_report_dir / f'batch_run_{session_part}_{mode_part}_{context_part}_{subcontext_part}_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write("BATCH RUN SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Session filter: {session_part}\n")
        f.write(f"Mode: {mode_part}\n")
        f.write(f"Context filter: {context_part}\n")
        if args.formula_only or args.include_formula:
            f.write(f"Subcontext filter: {subcontext_part}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {(end_time - start_time).total_seconds():.1f}s\n\n")
        
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Success: {sum(1 for r in results if r['status'] == 'SUCCESS')}\n")
        f.write(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}\n")
        f.write(f"Errors: {sum(1 for r in results if r['status'] == 'ERROR')}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("="*80 + "\n")
        for r in results:
            f.write(f"{r['status']:8s} | {r['log_file']:20s} | {r['display_context']:15s} | {r['duration']:6.1f}s")
            if r['error']:
                f.write(f" | {r['error']}")
            f.write("\n")
    
    print(f"\n📄 Run summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
