import pandas as pd
from pathlib import Path
from typing import List, Dict


def find_coverage_csvs(prefix_reports_dir: Path) -> List[Dict]:
    all_csvs = []

    # Walk through session_reports directories
    for session_dir in prefix_reports_dir.glob("*_reports"):
        print(f"📁 Processing session: {session_dir.name}")

        # Walk ALL subdirectories (city/, default/, formula/city/, etc.)
        for csv_path in session_dir.rglob("coverage_report_*.csv"):
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                row = df.iloc[0].to_dict()

                # Metadata from paths
                prefix = prefix_reports_dir.name.replace("_reports", "")
                session = session_dir.name.replace("_reports", "")
                context_path = csv_path.parent.relative_to(session_dir)
                full_context = str(context_path).replace("/", "_").replace("\\", "_")

                row.update(
                    {
                        "prefix": prefix,
                        "session": session,
                        "context_path": str(context_path),
                        "full_context": full_context,
                        "csv_file": csv_path.name,
                    }
                )

                all_csvs.append(row)

                try:
                    rel_path = csv_path.relative_to(prefix_reports_dir.parent.parent)
                    rel_str = f"reports/{rel_path}"
                except ValueError:
                    rel_str = str(csv_path)

                print(
                    f"   ✓ {rel_str} → overall={row.get('overall_coverage_pct', 'N/A')}"
                )
            except Exception as e:
                print(f"   Error reading {csv_path}: {e}")

    return all_csvs


def deduplicate_by_default_context(df_all: pd.DataFrame) -> pd.DataFrame:

    print("\n Deduplicating: keeping only DEFAULT context per session...")

    df_all["is_default_ctx"] = df_all["full_context"].str.contains(
        "default", case=False, na=False
    )

    filtered_rows = []
    for (prefix, session), group in df_all.groupby(["prefix", "session"]):
        # Prefer a default-context row if available
        default_rows = group[group["is_default_ctx"]]
        if not default_rows.empty:
            filtered_rows.append(default_rows.iloc[0])
            chosen_ctx = "DEFAULT"
        else:
            # Fallback: take the first row (e.g., city) if no default found
            filtered_rows.append(group.iloc[0])
            chosen_ctx = group.iloc[0]["full_context"]
        print(f"   ✓ {prefix}_{session}: using context={chosen_ctx}")

    df_result = pd.DataFrame(filtered_rows).reset_index(drop=True)
    df_result.drop(columns=["is_default_ctx"], inplace=True)

    print(f"After deduplication: {len(df_result)} unique sessions")
    return df_result


def save_raw_sessions(df_all: pd.DataFrame, output_dir: Path) -> Path:
    raw_path = output_dir / "01_raw_sessions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_cols = [
        "prefix",
        "session",
        "full_context",
        "csv_file",
        "log_file",
        "total_cells",
        "found_cells",
        "missing_cells",
        "overall_coverage_pct",
        "serving_coverage_pct",
        "neighbor_coverage_pct",
    ]
    # Keep only the columns that actually exist to avoid KeyError
    session_cols_present = [c for c in session_cols if c in df_all.columns]

    df_all[session_cols_present].to_csv(raw_path, index=False)
    print(f"📄 Saved {len(df_all)} unique sessions: {raw_path}")

    return raw_path


def build_per_prefix_summaries(df_all: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    summary_rows: List[Dict] = []

    for prefix in prefixes:
        prefix_df = df_all[df_all["prefix"] == prefix]
        if prefix_df.empty:
            continue

        summary_rows.append(
            {
                "prefix": prefix,
                "session": f"{prefix}_SUMMARY",
                "full_context": "AGGREGATED",
                "n_sessions": int(len(prefix_df)),
                "total_cells": int(prefix_df["total_cells"].sum()),
                "avg_overall_cov_pct": round(
                    prefix_df["overall_coverage_pct"].mean(), 1
                ),
                "avg_serving_cov_pct": round(
                    prefix_df["serving_coverage_pct"].mean(), 1
                ),
                "avg_neighbor_cov_pct": round(
                    prefix_df["neighbor_coverage_pct"].mean(), 1
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def build_global_summary(df_all: pd.DataFrame) -> Dict:
    return {
        "prefix": "GLOBAL",
        "session": "GLOBAL_SUMMARY",
        "full_context": "AGGREGATED",
        "n_sessions": int(len(df_all)),
        "total_cells": int(df_all["total_cells"].sum()),
        "avg_overall_cov_pct": round(df_all["overall_coverage_pct"].mean(), 1),
        "avg_serving_cov_pct": round(df_all["serving_coverage_pct"].mean(), 1),
        "avg_neighbor_cov_pct": round(df_all["neighbor_coverage_pct"].mean(), 1),
    }


def build_hierarchical_dataframe(
    df_all: pd.DataFrame, df_summary: pd.DataFrame
) -> pd.DataFrame:

    # Session-level rows: n_sessions = 1 for each deduplicated session
    df_sessions_hier = df_all[
        [
            "prefix",
            "session",
            "full_context",
            "total_cells",
            "overall_coverage_pct",
            "serving_coverage_pct",
            "neighbor_coverage_pct",
            "csv_file",
        ]
    ].rename(
        columns={
            "overall_coverage_pct": "avg_overall_cov_pct",
            "serving_coverage_pct": "avg_serving_cov_pct",
            "neighbor_coverage_pct": "avg_neighbor_cov_pct",
        }
    )
    df_sessions_hier["n_sessions"] = 1  # each row is one unique session

    # Make summary rows have same columns
    summary_cols = [
        "prefix",
        "session",
        "full_context",
        "n_sessions",
        "total_cells",
        "avg_overall_cov_pct",
        "avg_serving_cov_pct",
        "avg_neighbor_cov_pct",
    ]
    df_summary_hier = df_summary[summary_cols].copy()
    df_summary_hier["csv_file"] = ""  # no single file associated

    # Combine: sessions first, then summaries
    final_df = pd.concat(
        [df_sessions_hier, df_summary_hier], ignore_index=True, sort=False
    )

    # Mark row types
    final_df["row_type"] = "SESSION"
    final_df.loc[
        final_df["session"].str.contains("SUMMARY", na=False), "row_type"
    ] = "SUMMARY"

    return final_df


def save_hierarchical_summary(final_df: pd.DataFrame, output_dir: Path) -> Path:
    main_path = output_dir / "coverage_hierarchical_complete.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(main_path, index=False)

    print(f"\n HIERARCHICAL SUMMARY SAVED: {main_path}")
    return main_path


def print_hierarchical_summary(final_df: pd.DataFrame, df_summary: pd.DataFrame):

    print("\n Preview (first 20 rows):")
    preview_cols = [
        "prefix",
        "session",
        "full_context",
        "row_type",
        "n_sessions",
        "avg_overall_cov_pct",
    ]
    print(final_df[preview_cols].head(20).to_string(index=False))

    print("\n Prefix-level summary:")
    print(
        df_summary[
            ["prefix", "n_sessions", "avg_overall_cov_pct"]
        ].to_string(index=False)
    )
