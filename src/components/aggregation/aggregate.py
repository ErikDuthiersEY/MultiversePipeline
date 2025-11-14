import argparse
from pathlib import Path
from functools import reduce
import pandas as pd

def load_all_metric_parquets(score_dirs: list[Path]) -> list[pd.DataFrame]:
    """
    Load all .parquet files in the given directories.
    Each file must contain a 'model' column and metric columns.
    """
    files = []
    for d in score_dirs:
        d = Path(d)
        files.extend(sorted(d.glob("*.parquet")))

    if not files:
        raise FileNotFoundError(
            f"No parquet metric files found in: {', '.join(str(d) for d in score_dirs)}"
        )

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            raise ValueError(f"'model' column missing in {f}")
        dfs.append(df)

    return dfs

def merge_on_model(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all metric DataFrames on 'model' using outer joins.
    Assumes metric column names are unique or compatible.
    """
    if len(dfs) == 1:
        return dfs[0]

    def _merge(left, right):
        return pd.merge(left, right, on="model", how="outer")

    return reduce(_merge, dfs)


def compute_cross_task_adaptability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-task adaptability over a specific set of metrics.
    """
    metric_names = [
        "reasoning_open",
        "reasoning_close",
        "summarization",
        "retrieval",
        "refusal_correctness",
    ]
    target_cols = [f"{m}_avg" for m in metric_names]
    existing_cols = [c for c in target_cols if c in df.columns]

    if not existing_cols:
        return df

    vals = df[existing_cols]

    mean_scores = vals.mean(axis=1, skipna=True)
    var_scores = vals.var(axis=1, skipna=True)

    df = df.copy()
    df["cross_task_mean"] = mean_scores
    df["cross_task_variance"] = var_scores

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    score_dirs = [Path(p) for p in args.scores_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_dfs = load_all_metric_parquets(score_dirs)

    merged = merge_on_model(metric_dfs)

    merged = compute_cross_task_adaptability(merged)

    out_csv = out_dir / "final_report.csv"
    merged.to_csv(out_csv, index=False)

    print(f"[OK] Wrote aggregated report to {out_csv}")
    print(f"Columns in report: {list(merged.columns)}")


if __name__ == "__main__":
    main()