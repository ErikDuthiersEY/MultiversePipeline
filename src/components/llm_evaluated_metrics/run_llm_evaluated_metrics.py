import argparse
import pandas as pd
from pathlib import Path
from llm_evaluated_metrics import load_cfg, compute_bias_scores, make_client


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

    raw_path = raw_dir / "raw_output.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_output.parquet not found in {raw_dir}")

    raw_df = pd.read_parquet(raw_path)
    client = make_client(cfg)

    # Checkpoint
    bias_scores_path = out_dir / "bias_raw_scores.parquet"
    if bias_scores_path.exists():
        try:
            existing_scores = pd.read_parquet(bias_scores_path)
            print(f"[INFO] Loaded existing bias scores from {bias_scores_path}")
        except Exception as e:
            print(f"[WARN] Could not read existing {bias_scores_path}: {e}")
            existing_scores = None
    else:
        existing_scores = None

    bias_agg_df = compute_bias_scores(
        raw_df=raw_df,
        cfg=cfg,
        client=client,
        existing_scores=existing_scores,
        bias_scores_path=bias_scores_path,
    )

    obj_df = bias_agg_df.copy()
    obj_df["hallucination_avg"] = pd.NA  # placeholder for future

    obj_df = obj_df[["model", "hallucination_avg", "bias_avg"]]

    obj_scores_path = out_dir / "obj_scores.parquet"
    obj_df.to_parquet(obj_scores_path, index=False)
    print(f"[OK] Wrote LLM-evaluated metrics to {obj_scores_path}")
    print(f"Columns: {list(obj_df.columns)}")

if __name__ == "__main__":
    main()