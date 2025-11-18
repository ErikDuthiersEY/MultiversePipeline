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

    bias_df = compute_bias_scores(raw_df, cfg, client)

    obj_df = bias_df.copy()

    # Add hallucination columns now (empty), so schema is future-proof
    if "hallucination_avg" not in obj_df.columns:
        obj_df["hallucination_avg"] = pd.NA

    desired_cols = ["model", "hallucination_avg","bias_avg"]
    
    obj_df = obj_df[[c for c in desired_cols if c in obj_df.columns]]

    out_path = out_dir / "obj_scores.parquet"
    obj_df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote LLM-evaluated metrics to {out_path}")
    print(f"Columns: {list(obj_df.columns)}")


if __name__ == "__main__":
    main()