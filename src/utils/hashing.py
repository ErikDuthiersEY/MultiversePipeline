import hashlib
from typing import Any

def compute_run_hash(**kwargs: Any) -> str:
    """Hash único para run basado en params."""
    hash_input = str(sorted(kwargs.items()))
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]