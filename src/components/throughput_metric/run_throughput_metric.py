import argparse
from pathlib import Path
import pandas as pd
from throughput_metric import load_cfg, make_client, QPSThrottle, run_throughput_for_model


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    datasets_dir = Path(args.datasets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   

    parquet_files = sorted(datasets_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[THROUGHPUT] No parquet files found in {datasets_dir}")
        return
    
    tp_cfg = cfg["throughput"]
    inf_cfg = cfg["inference"]

    requests_per_model = int(tp_cfg["requests_per_model"])
    qps_limit = float(tp_cfg["qps_limit"])
    n_runs = int(tp_cfg["n_runs"])

    worker_counts = [int(w) for w in tp_cfg["worker_counts"]]

    dataset_task = tp_cfg["dataset_task"]
    if dataset_task:
        task_file = None
        for f in parquet_files:
            if f.stem == dataset_task:
                task_file = f
                break
        if task_file is None:
            print(f"[THROUGHPUT] Dataset task '{dataset_task}' not found. "
                  f"Falling back to first dataset.")
            task_file = parquet_files[0]
    else:
        task_file = parquet_files[0]

    task_name = task_file.stem
    print(f"[THROUGHPUT] Using dataset '{task_name}' for load generation: {task_file}")

    df = pd.read_parquet(task_file)
    if "prompt" not in df.columns:
        raise ValueError(f"[THROUGHPUT] Dataset {task_file} has no 'prompt' column.")
    prompts = df["prompt"].astype(str).tolist()
    if not prompts:
        print(f"[THROUGHPUT] No prompts found in dataset {task_file}")
        return
    
    system_prompts = inf_cfg["system_prompts"]
    task_system_prompt = system_prompts[task_name]
    
    client = make_client(cfg)
    throttle = QPSThrottle(qps_limit)

    run_rows = []
    for n_workers in worker_counts:
        for run_id in range(1, n_runs + 1):
            # Baseline model
            run_rows.append(
                run_throughput_for_model(
                    model_kind="baseline",
                    prompts=prompts,
                    cfg=cfg,
                    client=client,
                    throttle=throttle,
                    requests_per_model=requests_per_model,
                    n_workers=n_workers,
                    system_prompt=task_system_prompt,
                    run_id=run_id
                )
            )

            # Compressed model
            run_rows.append(
                run_throughput_for_model(
                    model_kind="compressed",
                    prompts=prompts,
                    cfg=cfg,
                    client=client,
                    throttle=throttle,
                    requests_per_model=requests_per_model,
                    n_workers=n_workers,
                    system_prompt=task_system_prompt,
                    run_id=run_id
                )
            )

    df_runs = pd.DataFrame(run_rows)
    runs_path = out_dir / "thput_runs.parquet"
    df_runs.to_parquet(runs_path, index=False)
    print(f"[THROUGHPUT] Wrote per-run throughput results to {runs_path}")

    if df_runs.empty:
        print("[THROUGHPUT] No runs recorded, nothing to aggregate.")
        return

    agg = df_runs.groupby(["model", "worker_count"]).agg(
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        throughput_rps_mean=("throughput_rps", "mean"),
        throughput_rps_std=("throughput_rps", "std"),
        latency_avg_ms_mean=("latency_avg_ms", "mean"),
        latency_avg_ms_std=("latency_avg_ms", "std"),
        latency_p95_ms_mean=("latency_p95_ms", "mean"),
        latency_p95_ms_std=("latency_p95_ms", "std"),
    ).reset_index()

    worker_counts = sorted(agg["worker_count"].unique())
    print(f"[THROUGHPUT] Aggregating metrics for worker_counts={worker_counts}")

    wide_df = None
    for wc in worker_counts:
        sub = agg[agg["worker_count"] == wc].copy()
        sub = sub.drop(columns=["worker_count"])

        rename_map = {
            col: f"{col}_w{wc}" for col in sub.columns if col != "model"
        }
        sub = sub.rename(columns=rename_map)

        if wide_df is None:
            wide_df = sub
        else:
            wide_df = wide_df.merge(sub, on="model", how="outer")

    final_df = wide_df

    out_path = out_dir / "obj_scores.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote aggregated throughput scores to {out_path}")
    print(f"[THROUGHPUT] Aggregated rows:\n{final_df}")


if __name__ == "__main__":
    main()