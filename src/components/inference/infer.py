import argparse, time, json, yaml
import pandas as pd 
from pathlib import Path 
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_cfg(path: str) -> dict: 
    with open(path, "r", encoding="utf-8") as f: 
        cfg = yaml.safe_load(f) if path.endswith((".yml", ".yaml")) else json.load(f)
    return cfg

def make_client(cfg: dict) -> AzureOpenAI:
    """
    Create a reusable Azure OpenAI client.
    """
    infer_cfg = cfg["inference"]

    client = AzureOpenAI(
        api_key=infer_cfg["azure_api_key"],
        api_version=infer_cfg["azure_api_version"],
        azure_endpoint=infer_cfg["azure_endpoint"],
        timeout=infer_cfg["timeout_s"],
    )
    return client

def call_model(client: AzureOpenAI, prompt: str, cfg: dict) -> dict:
    """
    Azure OpenAI Chat Completions call.
    Returns: dict(output, input_tokens, output_tokens, finish_reason, error)
    """

    infer_cfg = cfg["inference"]
    
    try:
        resp = client.chat.completions.create(
            model=infer_cfg["model_id"],                
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=infer_cfg["max_tokens"],
        )
        
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or ""
        usage = getattr(resp, "usage", None)

        return {
            "model": infer_cfg["model_id"],
            "output": content,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish_reason": finish_reason,
            "error": ""
        }
    
    except Exception as e:
        return {
            "model": infer_cfg["model_id"],
            "output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "error",
            "error": str(e)
        }

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
    
    n_workers = int(cfg["inference"]["n_workers"])
    client = make_client(cfg)
    
    for f in parquet_files: 
        task = f.stem
        print(f"\n=== Processing dataset: {task} ===")
        df = pd.read_parquet(f)
        total = len(df)
        print(f"Total prompts: {total}")
        print(f"Using {n_workers} workers\n")

        rows = []
        completed = 0

        def worker(prompt_id: str, prompt: str) -> dict:
            """
            Function executed by each thread.
            """
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
                rows.append(future.result())
                print(f"Completed {completed}/{total} prompts", end="\r")
                
        out_path = out_dir / f"raw_output_{task}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[OK] Wrote {out_path}")

if __name__ == "__main__": 
    main()