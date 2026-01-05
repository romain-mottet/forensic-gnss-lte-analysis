import argparse
import sys
import traceback

def _import_runner(nonlinear: bool = False):
    """Unified import for both linear and nonlinear."""
    try:
        from src.distance_model import run_context  # type: ignore
        if nonlinear:
            # For nonlinear, pass flag to run_context
            def runner(ctx): return run_context(ctx, nonlinear=True)
        else:
            def runner(ctx): return run_context(ctx, nonlinear=False)
        return runner
    except ModuleNotFoundError:
        raise ImportError("src/distance_model.py not found or missing run_context")

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("context", help="Context folder under ./data (e.g. waha, lln)")
    p.add_argument("--nonlinear", action="store_true", help="Run nonlinear (polynomial+interaction) search")
    args = p.parse_args(argv[1:])
    
    try:
        runner = _import_runner(args.nonlinear)
        out = runner(args.context)
        
        print(f"Context: {args.context}")
        print(f"Mode: {'nonlinear' if args.nonlinear else 'linear'}")
        print(f"Rows used: {out.get('n_rows')}")
        print(f"Tower joins: {out.get('tower_join_counts')}")
        print(f"Saved summary: {out.get('summary')}")
        print(f"Saved report: {out.get('report')}")
        return 0
        
    except Exception as e:
        print("ERROR:", repr(e))
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
