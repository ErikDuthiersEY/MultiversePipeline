import argparse, time, json 
from pathlib import Path 

import pandas as pd 
import yaml

def load_cfg(path: str) -> dict: 
    with open(path, "r", encoding="utf-8") as f: 
        cfg = yaml.safe_load(f) if path.endswith((".yml", ".yaml")) else json.load(f)

def call_model(prompt: str, cfg: dict) -> str: 
    """
    """      

    return {"response": "This is a placeholder response."}


def main(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    datasets_dir = Path(args.datasets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(datasets_dir.glob("*.parquet"))
    if not parquet_files: 
        print(f"No parquet files found in {datasets_dir}")
        return 
    
    for f in parquet_files: 
        task = f.stem
        df = pd.read_parquet(f)

        rows = []
        for _, r in df.iterrows():
            prompt_id = str(r["prompt_id"])
            prompt = str(r["prompt"])

            t0 = time.time()
            res = call_model(prompt, cfg)
            latency_ms = int((time.time() - t0) * 1000)

            rows.append({
                "model": cfg.get("model_id", "unknown-model"),
                "task": task,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "output": res.get("output", ""),
                "latency_ms": latency_ms,
                "finish_reason": res.get("finish_reason", "")
            })

        out_path = out_dir / f"raw_{task}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[OK] Wrote {out_path}")

if __name__ == "__main__": 
    main()