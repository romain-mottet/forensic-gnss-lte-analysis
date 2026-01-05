from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _extract_context_from_filename(prefix: str, session: str, csv_path: Path) -> str:

    stem = csv_path.stem  

    if stem.startswith("accuracy_summary_"):
        stem = stem[len("accuracy_summary_") :]

    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]


    if stem.startswith(prefix + "_"):
        stem = stem[len(prefix) + 1 :]  # 1_city OR 1_formula_city

    # session is like "lln_1" -> session_suffix is "1"
    session_suffix = session
    if session.startswith(prefix + "_"):
        session_suffix = session[len(prefix) + 1 :]

    # Remove leading "{session_suffix}_"
    if stem.startswith(session_suffix + "_"):
        stem = stem[len(session_suffix) + 1 :]

    return stem if stem else "unknown"


def find_trilateration_csvs(prefix_reports_dir: Path) -> List[Dict]:

    all_rows: List[Dict] = []

    for session_dir in prefix_reports_dir.glob("*_reports"):
        prefix = prefix_reports_dir.name.replace("_reports", "")
        session = session_dir.name.replace("_reports", "")

        for csv_path in session_dir.rglob("accuracy_summary_*_metrics.csv"):
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                context = _extract_context_from_filename(prefix=prefix, session=session, csv_path=csv_path)

                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    row_dict.update(
                        {
                            "prefix": prefix,
                            "session": session,
                            "context": context,
                            "csv_file": csv_path.name,
                        }
                    )
                    all_rows.append(row_dict)

            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    return all_rows


def save_raw_trilateration_records(df_all: pd.DataFrame, output_dir: Path) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "02_raw_trilateration_records.csv"

    record_cols = [
        "prefix",
        "session",
        "context",
        "csv_file",
        "timestamp",
        "distance_method",
        "samples",
        "cep_m",
        "r95_m",
        "rmse_m",
        "mean_error_m",
        "std_error_m",
        "min_error_m",
        "max_error_m",
        "accuracy_1cell_rmse_m",
        "accuracy_3cell_rmse_m",
        "accuracy_4cell_rmse_m",
        "accuracy_5cell_rmse_m",
        "quality_gdop_EXCELLENT_count",
        "quality_gdop_GOOD_count",
        "quality_gdop_ACCEPTABLE_count",
        "quality_gdop_POOR_count",
        "quality_gdop_INSUFFICIENTCOORDS_count",
        "within_200m_pct",
        "gdop_lt8_count",
    ]

    cols_present = [c for c in record_cols if c in df_all.columns]
    df_all[cols_present].to_csv(raw_path, index=False)
    return raw_path


def _avg_metrics(group: pd.DataFrame) -> Dict:
    metric_cols = [
        "cep_m",
        "r95_m",
        "rmse_m",
        "mean_error_m",
        "std_error_m",
        "min_error_m",
        "max_error_m",
        "within_200m_pct",
    ]

    out: Dict = {}
    for col in metric_cols:
        if col in group.columns:
            v = group[col].mean()
            if pd.notna(v):
                out[f"avg_{col}"] = round(float(v), 1)
    return out


def build_context_summaries(df_all: pd.DataFrame) -> pd.DataFrame:

    rows: List[Dict] = []

    for (prefix, context), group in df_all.groupby(["prefix", "context"]):
        r = {
            "location": prefix,
            "context": context,
            "n_sessions": int(group["session"].nunique()) if "session" in group.columns else 0,
            "n_records": int(len(group)),
            "n_samples": int(group["samples"].sum()) if "samples" in group.columns else 0,
        }
        r.update(_avg_metrics(group))
        rows.append(r)

    return pd.DataFrame(rows)


def build_method_summaries(df_all: pd.DataFrame) -> pd.DataFrame:

    rows: List[Dict] = []

    if "distance_method" not in df_all.columns:
        return pd.DataFrame(rows)

    for method, group in df_all.groupby("distance_method"):
        method_name = "unknown" if pd.isna(method) else str(method)
        r = {
            "distance_method": method_name,
            "n_sessions": int(group["session"].nunique()) if "session" in group.columns else 0,
            "n_contexts": int(group["context"].nunique()) if "context" in group.columns else 0,
            "n_records": int(len(group)),
            "n_samples": int(group["samples"].sum()) if "samples" in group.columns else 0,
        }
        r.update(_avg_metrics(group))
        rows.append(r)

    return pd.DataFrame(rows)


def build_location_summaries(df_all: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:

    rows: List[Dict] = []

    for prefix in prefixes:
        group = df_all[df_all["prefix"] == prefix]
        if group.empty:
            continue

        r = {
            "location": prefix,
            "n_sessions": int(group["session"].nunique()) if "session" in group.columns else 0,
            "n_contexts": int(group["context"].nunique()) if "context" in group.columns else 0,
            "n_records": int(len(group)),
            "n_samples": int(group["samples"].sum()) if "samples" in group.columns else 0,
        }
        r.update(_avg_metrics(group))
        rows.append(r)

    return pd.DataFrame(rows)


def build_session_availability(df_all: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:

    if "context" not in df_all.columns:
        return pd.DataFrame()

    all_contexts = sorted([c for c in df_all["context"].dropna().unique().tolist()])

    rows: List[Dict] = []
    for prefix in prefixes:
        pref_df = df_all[df_all["prefix"] == prefix]
        if pref_df.empty:
            continue

        for session in sorted(pref_df["session"].dropna().unique().tolist()):
            sess_df = pref_df[pref_df["session"] == session]
            available = set(sess_df["context"].dropna().unique().tolist())

            row = {"location": prefix, "session": session}
            for ctx in all_contexts:
                row[ctx] = 1 if ctx in available else 0

            row["n_available_contexts"] = int(sum(row[ctx] for ctx in all_contexts))
            row["n_missing_contexts"] = int(len(all_contexts) - row["n_available_contexts"])
            rows.append(row)

    # Keep a stable column order
    cols = ["location", "session", "n_available_contexts", "n_missing_contexts"] + all_contexts
    return pd.DataFrame(rows)[cols]


def save_merged_trilateration_evaluation(df_all: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "trilateration_merged_evaluation.csv"
    df_all.to_csv(merged_path, index=False)
    return merged_path


def print_trilateration_summary(df_all: pd.DataFrame, output_dir: Path) -> None:

    print("=" * 80)
    print("TRILATERATION ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Records: {len(df_all):,}")
    if "prefix" in df_all.columns and "session" in df_all.columns:
        print(f"Unique sessions: {df_all.groupby(['prefix', 'session']).ngroups}")
    if "context" in df_all.columns:
        print(f"Unique contexts: {df_all['context'].nunique()}")
    if "samples" in df_all.columns:
        print(f"Total samples: {int(df_all['samples'].sum()):,}")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    reports_dir = script_dir.parent / "reports"
    output_dir = script_dir.parent / "summary"

    prefixes = ["ixelle", "lln", "waha"]
    all_records: List[Dict] = []

    for prefix in prefixes:
        prefix_dir = reports_dir / f"{prefix}_reports"
        if prefix_dir.exists():
            all_records.extend(find_trilateration_csvs(prefix_dir))

    if not all_records:
        print("No trilateration accuracy records found.")
        raise SystemExit(0)

    df = pd.DataFrame(all_records)

    save_raw_trilateration_records(df, output_dir)
    save_merged_trilateration_evaluation(df, output_dir)

    build_context_summaries(df).to_csv(output_dir / "trilateration_summary_by_context.csv", index=False)
    build_method_summaries(df).to_csv(output_dir / "trilateration_summary_by_method.csv", index=False)
    build_location_summaries(df, prefixes).to_csv(output_dir / "trilateration_summary_by_location.csv", index=False)
    build_session_availability(df, prefixes).to_csv(output_dir / "trilateration_session_availability.csv", index=False)

    print_trilateration_summary(df, output_dir)
