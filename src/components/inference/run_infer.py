import argparse, time
import pandas as pd 
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from infer import load_cfg, call_model, QPSThrottle, make_client

def main():
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
    
    cfg_inf = cfg["inference"]
    n_workers = int(cfg_inf["n_workers"])
    qps_limit = float(cfg_inf["qps_limit"])
    client = make_client(cfg)
    throttle = QPSThrottle(qps_limit)

    sample_size = cfg_inf["sample_size_per_task"]

    out_path = out_dir / "raw_output.parquet"

    #checkpoint 
    existing_df = None 
    if out_path.exists():
        try:
            existing_df = pd.read_parquet(out_path)
            print(f"Found existing global output file: {out_path}")
        except Exception as e:
            print(f"Warning: could not read existing {out_path} ({e}). "
                  f"Re-running all prompts for all datasets.")
            existing_df = None

    all_new_rows = []

    for f in parquet_files: 
        task = f.stem
        if task != "reasoning_open":
            continue
        print(f"\n=== Processing dataset: {task} ===")

        df = pd.read_parquet(f)

        if sample_size is not None:
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                print(f"[DEV MODE] Sampling {sample_size} rows for {task}")

        #checkpoint per task
        done_ids = set() 
        if existing_df is not None and not existing_df.empty:
            done_ids = set(
                existing_df.loc[existing_df["task"] == task, "prompt_id"]
                .astype(str)
                .tolist()
            )
            print(f"Already completed prompts for {task}: {len(done_ids)}")

        if done_ids:
            df = df[~df["id"].astype(str).isin(done_ids)]

        if df.empty: 
            print("No remaining prompts to run for this dataset.")
            continue
                
        total = len(df)
        print(f"Total prompts: {total}")
        print(f"Using {n_workers} workers, QPS limit ≈ {qps_limit} req/s\n")

        completed = 0

        def worker(prompt_id: str, prompt: str) -> dict:
            """
            Function executed by each thread.
            """
            throttle.wait()

            t0 = time.time()
            res = call_model(client, prompt, cfg)
            latency_ms = int((time.time() - t0) * 1000)

            return {
                "model": res["model"],
                "task": task,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "output": res["output"],
                "latency_ms": latency_ms,
                "input_tokens": int(res["input_tokens"]),
                "output_tokens": int(res["output_tokens"]),
                "finish_reason": res["finish_reason"],
                "error": res["error"],
            }

        futures = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for _, r in df.iterrows():
                prompt_id = str(r["id"])
                prompt = str(r["prompt"])
                futures.append(
                    executor.submit(worker, prompt_id, prompt)
                )
            
            for future in as_completed(futures):
                completed += 1
                all_new_rows.append(future.result())
                print(f"Completed {completed}/{total} prompts", end="\r")

        if not all_new_rows and existing_df is not None:
            print("\nNo new rows to add. Keeping existing raw_output.parquet as-is.")
            return

        new_df = pd.DataFrame(all_new_rows) if all_new_rows else pd.DataFrame()

        if existing_df is not None and not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_parquet(out_path, index=False)
        print(f"\n[OK] Wrote {out_path} ({len(final_df)} total rows)\n")

if __name__ == "__main__": 
    main()