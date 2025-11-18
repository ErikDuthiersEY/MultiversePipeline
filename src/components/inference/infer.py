import time, json, yaml
import threading
from openai import AzureOpenAI

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

def call_model(client: AzureOpenAI, prompt: str, cfg: dict) -> dict:
    """
    Azure OpenAI Chat Completions call.
    Returns: dict(output, input_tokens, output_tokens, finish_reason, error)
    """

    infer_cfg = cfg["inference"]
    global_cfg = cfg["global"]
    
    try:
        resp = client.chat.completions.create(
            model=global_cfg["base_model_id"],                
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=infer_cfg["max_tokens"],
        )
        
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or ""
        usage = getattr(resp, "usage", None)

        return {
            "model": global_cfg["base_model_id"],
            "output": content,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish_reason": finish_reason,
            "error": ""
        }
    
    except Exception as e:
        return {
            "model": global_cfg["base_model_id"],
            "output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "error",
            "error": str(e)
        }

