import argparse
from aggregate import load_all_metric_parquets, merge_on_model, compute_cross_task_adaptability
from pathlib import Path


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