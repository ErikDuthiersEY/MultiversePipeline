import argparse
import pandas as pd
from pathlib import Path

from llm_evaluated_metrics import load_cfg, make_client, compute_bias_scores, compute_hallucinations_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config)
    client = make_client(cfg)

    raw_path = raw_dir / "raw_output.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_output.parquet not found in {raw_dir}")

    raw_df = pd.read_parquet(raw_path)


    bias_scores_path = out_dir / "bias_raw_scores.parquet"
    existing_bias = pd.read_parquet(bias_scores_path) if bias_scores_path.exists() else None
    bias_agg_df = compute_bias_scores(raw_df, cfg, client, existing_bias, bias_scores_path)

    hall_scores_path = out_dir / "hallucination_raw_scores.parquet"
    existing_hall = pd.read_parquet(hall_scores_path) if hall_scores_path.exists() else None
    hall_agg_df = compute_hallucinations_scores(raw_df, cfg, client, existing_hall, hall_scores_path)


    obj_df = bias_agg_df.merge(hall_agg_df, on="model", how="outer")
    obj_df = obj_df[["model", "hallucination_avg", "bias_avg"]]

    obj_scores_path = out_dir / "obj_scores.parquet"
    obj_df.to_parquet(obj_scores_path, index=False)
    print(f"[OK] Wrote final metrics to {obj_scores_path}")
    print(obj_df)


if __name__ == "__main__":
    main()