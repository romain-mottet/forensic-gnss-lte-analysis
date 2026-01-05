import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from src.config.parameters import (
    TIMESTAMP_FORMAT_PARAM,
    GAP_THRESHOLD_SECONDS_PARAM,
)

TIMESTAMP_FORMAT = TIMESTAMP_FORMAT_PARAM
GAP_THRESHOLD_SECONDS = GAP_THRESHOLD_SECONDS_PARAM


class GPSQualityAnalyzer:
    """Analyzes GPS comparison data for quality issues."""

    def __init__(self, results_dir: str = "results", reports_dir: str = "reports"):
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        self._init_reports_dir()

    def _init_reports_dir(self) -> None:
        """Create reports/ directory with same substructure as results/."""
        if not self.results_dir.exists():
            raise FileNotFoundError(f"results directory not found: {self.results_dir}")
        for loc_dir in self.results_dir.iterdir():
            if loc_dir.is_dir():
                (self.reports_dir / loc_dir.name).mkdir(parents=True, exist_ok=True)


    def _parse_timestamp(self, s: str) -> Optional[datetime]:
        try:
            return datetime.strptime(s, TIMESTAMP_FORMAT)
        except Exception:
            return None

    def _load_csv(self, path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows, (reader.fieldnames or [])


    def _find_duplicate_timestamps(self, timestamps: List[str]) -> Dict[str, List[int]]:
        """
        Returns: {timestamp: [index1, index2, ...], ...}
        """
        idx_by_ts: Dict[str, List[int]] = {}
        for i, ts in enumerate(timestamps):
            idx_by_ts.setdefault(ts, []).append(i)
        return {ts: idxs for ts, idxs in idx_by_ts.items() if len(idxs) > 1}


    def _find_gaps(self, timestamps: List[str]) -> List[Dict]:
        """
        Returns list of gap dictionaries.
        """
        gaps: List[Dict] = []
        if len(timestamps) < 2:
            return gaps

        for i in range(1, len(timestamps)):
            t_prev = self._parse_timestamp(timestamps[i - 1])
            t_curr = self._parse_timestamp(timestamps[i])
            if t_prev is None or t_curr is None:
                continue
            delta = (t_curr - t_prev).total_seconds()
            if delta >= GAP_THRESHOLD_SECONDS:
                gaps.append(
                    {
                        "prev_index": i - 1,
                        "index": i,
                        "prev_timestamp": timestamps[i - 1],
                        "curr_timestamp": timestamps[i],
                        "gap_seconds": delta,
                    }
                )
        return gaps
    
    def analyze_file(self, csv_path: Path) -> Dict:

        location = csv_path.parent.name
        filename = csv_path.name

        try:
            rows, fieldnames = self._load_csv(csv_path)
        except Exception as e:
            return {
                "success": False,
                "location": location,
                "filename": filename,
                "error": f"Failed to read CSV: {e}",
            }

        if not rows:
            return {
                "success": False,
                "location": location,
                "filename": filename,
                "error": "Empty CSV (no data rows)",
            }

        timestamps = [r.get("timestamp", "") for r in rows]

        # Find duplicates
        duplicates = self._find_duplicate_timestamps(timestamps)
        num_dup_timestamps = len(duplicates)
        num_dup_rows = sum(len(idxs) for idxs in duplicates.values())

        # Find gaps
        gaps = self._find_gaps(timestamps)
        num_gaps = len(gaps)
        max_gap = max((g["gap_seconds"] for g in gaps), default=0.0)

        # Calculate duration
        t0 = self._parse_timestamp(timestamps[0])
        t1 = self._parse_timestamp(timestamps[-1])
        duration_sec = (t1 - t0).total_seconds() if t0 and t1 else 0.0

        return {
            "success": True,
            "location": location,
            "filename": filename,
            "rows": rows,
            "fieldnames": fieldnames,
            "timestamps": timestamps,
            "total_records": len(rows),
            "duration_seconds": duration_sec,
            "duplicates": duplicates,
            "num_dup_timestamps": num_dup_timestamps,
            "num_dup_rows": num_dup_rows,
            "gaps": gaps,
            "num_gaps": num_gaps,
            "max_gap_seconds": max_gap,
        }


    def _format_report(self, result: Dict) -> str:
        """Format analysis result as text report."""
        if not result["success"]:
            return f"ERROR analyzing {result['filename']}: {result['error']}\n"

        lines: List[str] = []

        lines.append("=" * 80)
        lines.append(f"GPS QUALITY REPORT: {result['filename']}")
        lines.append(f"Location: {result['location']}")
        lines.append("=" * 80)
        lines.append("")

        # Basic statistics
        lines.append("[BASIC]")
        lines.append(f"Total records: {result['total_records']}")
        dur_min = result["duration_seconds"] / 60 if result["duration_seconds"] else 0.0
        lines.append(
            f"Duration: {result['duration_seconds']:.1f} seconds ({dur_min:.1f} minutes)"
        )
        lines.append("")

        # Duplicates section
        lines.append("=" * 80)
        lines.append("[DUPLICATE TIMESTAMPS]")
        lines.append("=" * 80)
        if result["num_dup_timestamps"] == 0:
            lines.append("No duplicate timestamps found.")
        else:
            lines.append(
                f"Found {result['num_dup_timestamps']} timestamps with duplicates "
                f"({result['num_dup_rows']} rows total)."
            )
            lines.append("")
            for ts, idxs in sorted(result["duplicates"].items()):
                lines.append(f"- Timestamp: {ts}")
                lines.append(f"  Row indices: {idxs}")
                lines.append("")

        # Gaps section
        lines.append("=" * 80)
        lines.append(f"[TIME GAPS >= {GAP_THRESHOLD_SECONDS} s]")
        lines.append("=" * 80)
        if result["num_gaps"] == 0:
            lines.append("No gaps above threshold.")
        else:
            lines.append(
                f"Found {result['num_gaps']} gaps >= {GAP_THRESHOLD_SECONDS} s "
                f"(max = {result['max_gap_seconds']:.1f} s)."
            )
            lines.append("")
            for g in result["gaps"]:
                lines.append(
                    f"- Gap from row {g['prev_index']} to {g['index']}: "
                    f"{g['gap_seconds']:.1f} s"
                )
                lines.append(f"  prev: {g['prev_timestamp']}")
                lines.append(f"  next: {g['curr_timestamp']}")
                lines.append("")

        # Summary section
        lines.append("=" * 80)
        lines.append("[SUMMARY]")
        lines.append("=" * 80)
        issues: List[str] = []
        if result["num_dup_timestamps"]:
            issues.append(f"{result['num_dup_timestamps']} duplicate timestamp(s)")
        if result["num_gaps"]:
            issues.append(f"{result['num_gaps']} gap(s) >= {GAP_THRESHOLD_SECONDS}s")

        if issues:
            lines.append("Issues detected:")
            for s in issues:
                lines.append(f"- {s}")
        else:
            lines.append("No duplicate timestamps or large gaps detected.")
        lines.append("")

        return "\n".join(lines)

    def _report_path_for(self, csv_path: Path) -> Path:
        """
        Convert CSV path to report path.

        results/lln/comparison_lln_5.csv ->
            reports/lln/comparison_lln_5_quality_report.txt
        """
        location = csv_path.parent.name
        fname = csv_path.name.replace(".csv", "_quality_report.txt")
        return self.reports_dir / location / fname

    def run_for_all_files(self) -> None:
        print("=" * 80)
        print("GPS QUALITY ANALYSIS")
        print("=" * 80)

        summary_rows: List[Dict] = []

        # Process each location directory
        for loc_dir in sorted(self.results_dir.iterdir()):
            if not loc_dir.is_dir():
                continue

            print(f"\nLocation: {loc_dir.name}")
            print("-" * 80)

            # Process each comparison file, but SKIP flagged files
            for csv_path in sorted(loc_dir.glob("comparison_*.csv")):
                # Skip flagged comparison files (comparison_flag_*.csv)
                if "_flag_" in csv_path.name:
                    continue

                print(f" Analyzing {csv_path.name} ... ", end="", flush=True)

                result = self.analyze_file(csv_path)
                report_text = self._format_report(result)
                report_path = self._report_path_for(csv_path)

                # Ensure directory exists and write report
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report_text, encoding="utf-8")

                print("done")

                # Add to summary
                if result["success"]:
                    summary_rows.append(
                        {
                            "file": csv_path.name,
                            "location": loc_dir.name,
                            "records": result["total_records"],
                            "dup_ts": result["num_dup_timestamps"],
                            "gaps": result["num_gaps"],
                        }
                    )

        # Print summary table
        if summary_rows:
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"{'Location':<10} {'File':<30} {'Records':>8} {'DupTS':>6} {'Gaps':>6}")
            print("-" * 80)
            for row in summary_rows:
                print(
                    f"{row['location']:<10} {row['file']:<30} "
                    f"{row['records']:>8} {row['dup_ts']:>6} {row['gaps']:>6}"
                )

            print(f"\nReports written to: {self.reports_dir.resolve()}")
            print("=" * 80 + "\n")


def main():
    """Entry point - run analysis with default directories."""
    try:
        analyzer = GPSQualityAnalyzer(results_dir="results", reports_dir="reports")
        analyzer.run_for_all_files()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nMake sure you run this script from the project root:")
        print("  python src/quality_analyzer.py")


if __name__ == "__main__":
    main()
