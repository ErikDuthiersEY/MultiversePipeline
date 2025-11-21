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
    n_workers = int(tp_cfg["n_workers"])
    qps_limit = float(tp_cfg["qps_limit"])
    n_runs = int(tp_cfg["n_runs"])

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

    agg = df_runs.groupby("model").agg(
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        throughput_rps_mean=("throughput_rps", "mean"),
        throughput_rps_std=("throughput_rps", "std"),
        latency_avg_ms_mean=("latency_avg_ms", "mean"),
        latency_avg_ms_std=("latency_avg_ms", "std"),
        latency_p95_ms_mean=("latency_p95_ms", "mean"),
        latency_p95_ms_std=("latency_p95_ms", "std"),
    ).reset_index()

    final_df = agg.rename(
        columns={
            "success_rate_mean": "thput_success_rate",
            "throughput_rps_mean": "thput_throughput_rps",
            "latency_avg_ms_mean": "thput_latency_avg_ms",
            "latency_p95_ms_mean": "thput_latency_p95_ms",
        }
    )

    out_path = out_dir / "thput_scores.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote aggregated throughput scores to {out_path}")
    print(f"[THROUGHPUT] Aggregated rows:\n{final_df}")


if __name__ == "__main__":
    main()