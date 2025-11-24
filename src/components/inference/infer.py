import time, json, yaml
import threading
from openai import AzureOpenAI
import requests
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

def call_model(client: AzureOpenAI, prompt: str, cfg: dict, system_prompt: str) -> dict:
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
    
def call_compactifai_model(prompt: str, cfg: dict, model_id: str, system_prompt: str | None = None) -> dict:
    """
    Compressed model call via CompactifAI HTTP API.
    Returns: dict(model, output, input_tokens, output_tokens, finish_reason, error)
    """
    
    infer_cfg = cfg["inference"]
    compact_cfg = cfg["compactifai"]

    api_key = compact_cfg["api_key"]
    url = compact_cfg["url"]

    temperature = infer_cfg["temperature"]
    max_tokens = infer_cfg["max_tokens"]
    timeout = infer_cfg["timeout_s"]

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
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=timeout,
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
