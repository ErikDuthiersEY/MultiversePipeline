import argparse
import time
import json
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from openai import AzureOpenAI
import numpy as np
import threading
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_cfg(path: str) -> dict: 
    with open(path, "r", encoding="utf-8") as f: 
        cfg = yaml.safe_load(f) if path.endswith((".yml", ".yaml")) else json.load(f)
    return cfg

def make_client(cfg: dict) -> AzureOpenAI:
    """
    Create a reusable Azure OpenAI client.
    """
    azure_cfg = cfg["azure"]
    infer_cfg = cfg["inference"]

    client = AzureOpenAI(
        api_key=azure_cfg["api_key"],
        api_version=azure_cfg["api_version"],
        azure_endpoint=azure_cfg["endpoint"],
        timeout=infer_cfg["timeout_s"],
    )
    return client

class QPSThrottle:
    """Simple global QPS limiter shared by all threads in this process."""
    def __init__(self, qps: float):
        self.qps = max(qps, 0.0001)
        self.min_interval = 1.0 / self.qps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.time()

def call_baseline_model(client: AzureOpenAI, prompt: str, cfg: dict, system_prompt: str) -> dict:
    """
    Azure OpenAI Chat Completions call.
    Returns: dict(output, input_tokens, output_tokens, finish_reason, error)
    """

    infer_cfg = cfg["inference"]
    base_cfg = cfg["baseline"]

    model = base_cfg["model_id"]

    messages = []
    if system_prompt: 
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        resp = client.chat.completions.create(
            model=model,                
            messages=messages,
            max_completion_tokens=infer_cfg["max_tokens"]
            #temperature=infer_cfg["temperature"],
        )
        
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or ""
        usage = getattr(resp, "usage", None)

        return {
            "model": model,
            "output": content,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish_reason": finish_reason,
            "error": ""
        }
    
    except Exception as e:
        return {
            "model": model,
            "output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "error",
            "error": str(e)
        }
    
def call_compactifai_model(prompt: str, cfg: dict, system_prompt: str | None = None) -> dict:
    """
    Compressed model call via CompactifAI HTTP API.
    Returns: dict(model, output, input_tokens, output_tokens, finish_reason, error)
    """
    
    infer_cfg = cfg["inference"]
    compact_cfg = cfg["compactifai"]

    api_key = compact_cfg["api_key"]
    url = compact_cfg["url"]
    model_id = compact_cfg["model_id"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": infer_cfg["temperature"],
        "max_tokens": infer_cfg["max_tokens"],
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=infer_cfg["timeout_s"],
            verify=False
        )
        
        resp.raise_for_status()
        body = resp.json()

        choice = body["choices"][0]
        message = choice["message"]
        content = message["content"]
        finish_reason = choice.get("finish_reason") or choice.get("stop_reason", "")
        usage = body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        model_name = body.get("model", model_id)

        return {
            "model": model_name,
            "output": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "error": "",
        }

    except Exception as e:
        return {
            "model": model_id,
            "output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "error",
            "error": str(e),
        }


def run_throughput_for_model(
    model_kind: str,
    prompts: list[str],
    cfg: dict,
    client: AzureOpenAI,
    throttle: QPSThrottle,
    requests_per_model: int,
    n_workers: int,
    system_prompt: str | None,
    run_id: int,
) -> dict:
    """
    Run a simple throughput test for a single model (baseline or compressed).

    model_kind: "baseline" or "compressed"
    """

    if not prompts: 
        raise ValueError("No prompts available for throughput test.")
    
    if model_kind == "baseline":
        call_fn = lambda p: call_baseline_model(client, p, cfg, system_prompt)
        report_model_name = cfg["baseline"]["model_id"]
    else: 
        call_fn = lambda p: call_compactifai_model(p, cfg, system_prompt)
        report_model_name = cfg["compactifai"]["model_id"]


    total_requests_target = int(requests_per_model)
    latencies_ms: list[int] = []
    success = 0 
    failure = 0 

    def worker(prompt: str) -> dict:
        throttle.wait()
        t0 = time.time()
        res = call_fn(prompt)
        dt_ms = int((time.time() - t0) * 1000)
        return {
            "latency_ms": dt_ms,
            "error": res["error"],
        }

    print(f"\n[THROUGHPUT] Run {run_id} | Model={report_model_name} | "
          f"Requests target={total_requests_target} | Workers={n_workers}")
    
    num_prompts = len(prompts)

    t_start = time.time() 

    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor: 
        for i in range(total_requests_target):
            prompt = prompts[i % num_prompts]
            futures.append(executor.submit(worker, prompt))

        completed = 0 
        for fut in as_completed(futures): 
            result = fut.result() 
            completed += 1 
            if result["error"]:
                failure += 1
            else: 
                success += 1 
                latencies_ms.append(result["latency_ms"])

            print(f"[THROUGHPUT] Run {run_id} | {report_model_name}: "
                  f"{completed}/{total_requests_target} requests done", end="\r")
            
    elapsed_s = max(time.time() - t_start, 1e-6)
    total_requests = success + failure
    throughput_rps = total_requests / elapsed_s if total_requests > 0 else 0.0
    success_rate = success / total_requests if total_requests > 0 else 0.0 

    if latencies_ms: 
        latency_avg_ms = float(np.mean(latencies_ms))
        latency_p95_ms = float(np.percentile(latencies_ms, 95))
    else: 
        latency_avg_ms = None 
        latency_p95_ms = None

    print(f"\n[THROUGHPUT] Run {run_id} done for model={report_model_name}: "
            f"success={success}, failure={failure}, "
            f"throughput_rps={throughput_rps:.3f}, "
            f"latency_avg_ms={latency_avg_ms}, latency_p95_ms={latency_p95_ms}")


    return {
        "model": report_model_name,
        "run_id": run_id,
        "success_rate": success_rate,
        "throughput_rps": throughput_rps,
        "latency_avg_ms": latency_avg_ms,
        "latency_p95_ms": latency_p95_ms,
    }