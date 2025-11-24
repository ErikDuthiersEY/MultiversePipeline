import argparse, time
import pandas as pd 
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from infer import load_cfg, call_model, QPSThrottle, make_client, call_compactifai_model

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
    task_system_prompts = cfg_inf["system_prompts"]

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
        print(f"\n=== Processing dataset: {task} ===")

        task_system_prompt = task_system_prompts[task]

        df = pd.read_parquet(f)

        if task == cfg_inf["variation_sensitivity_task_name"]:
            df = df.sort_values("id").reset_index(drop=True)

        if sample_size is not None and len(df) > sample_size:
            if task == cfg_inf["variation_sensitivity_task_name"]:
                # Take the first N rows in order so groups remain intact
                df = df.head(sample_size)
                print(f"[DEV MODE] Taking first {sample_size} rows for {task} (preserve grouping)")
            else:
                # Other tasks can still use random sampling
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

        def worker(prompt_id: str, prompt: str) -> list[dict]:
            """
            Function executed by each thread.
            For each prompt, call BOTH:
              - baseline model (Azure OpenAI)
              - compressed model (CompactifAI)
            Returns a list of two row dicts.
            """
            rows = []
            baseline_model_id = cfg["baseline"]["model_id"]
            compressed_model_id = cfg["compactifai"]["model_id"]

            # Baseline model (Azure)
            throttle.wait()
            t0 = time.time()
            res_base = call_compactifai_model(prompt, cfg, baseline_model_id, task_system_prompt)
            latency_base_ms = int((time.time() - t0) * 1000)

            rows.append(
                {
                    "model": res_base["model"],
                    "task": task,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "output": res_base["output"],
                    "latency_ms": latency_base_ms,
                    "input_tokens": int(res_base["input_tokens"]),
                    "output_tokens": int(res_base["output_tokens"]),
                    "finish_reason": res_base["finish_reason"],
                    "error": res_base["error"],
                }
            )

            # Compressed model (CompactifAI)
            throttle.wait()
            t1 = time.time()
            res_comp = call_compactifai_model(prompt, cfg, compressed_model_id, task_system_prompt)
            latency_comp_ms = int((time.time() - t1) * 1000)

            rows.append(
                {
                    "model": res_comp["model"],
                    "task": task,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "output": res_comp["output"],
                    "latency_ms": latency_comp_ms,
                    "input_tokens": int(res_comp["input_tokens"]),
                    "output_tokens": int(res_comp["output_tokens"]),
                    "finish_reason": res_comp["finish_reason"],
                    "error": res_comp["error"],
                }
            )

            return rows
        
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
                rows_for_prompt = future.result()  
                all_new_rows.extend(rows_for_prompt)
                print(f"Completed {completed}/{total} prompts (both models)", end="\r")

        if not all_new_rows and existing_df is not None:
            print("\nNo new rows to add. Keeping existing raw_output.parquet as-is.")
            return

        new_df = pd.DataFrame(all_new_rows) if all_new_rows else pd.DataFrame()

        if not new_df.empty:
            new_df = (
                new_df
                .sort_values(["task", "prompt_id", "model"])
                .reset_index(drop=True)
            )

        if existing_df is not None and not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_parquet(out_path, index=False)
        print(f"\n[OK] Wrote {out_path} ({len(final_df)} total rows)\n")

if __name__ == "__main__": 
    main()