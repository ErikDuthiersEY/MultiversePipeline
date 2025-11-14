import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from typing import Dict, Any

from src.utils.hashing import compute_run_hash
from src.components.rule_based_metrics.rule_based_metrics import compute_rule_based_metrics

def main():
    parser = argparse.ArgumentParser(description="Run rule-based metrics computation.")
    parser.add_argument("--config", default="configs/rule_based_metrics.yaml", help="Path to rule_based_metrics.yaml")
    parser.add_argument("--raw_path", default="outputs/raw_output.parquet", help="Path to raw_output.parquet from inference")  
    parser.add_argument("--datasets", default="datasets/processed", help="Path to datasets dir")
    parser.add_argument("--out_dir", default="outputs", help="Output dir for obj_scores")  # <-- CAMBIADO: Ahora "outputs" directo, sin /metrics
    parser.add_argument("--dataset_version", default="v1", help="Dataset version")
    parser.add_argument("--code_version", default="v0.1", help="Code version (e.g., from git)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Compute run_hash  # Desactivado para output fijo sin hash
    # model_id = "multi" 
    # config_hash = config.get("hash", "auto")
    # run_hash = compute_run_hash(
    #     code_version=args.code_version,
    #     config_hash=config_hash,
    #     model_id=model_id,
    #     dataset_version=args.dataset_version
    # )
    # print(f"Generated run_hash: {run_hash}")

    # Paths  # <-- AJUSTADO: Output fijo sin {run_hash}
    raw_outputs_path = Path(args.raw_path)
    obj_scores_path = Path(args.out_dir) / "obj_scores.parquet"  # Ahora: outputs/obj_scores.parquet
    datasets_root = Path(args.datasets)

    # Check if already exists  # Desactivado para siempre sobrescribir
    # if obj_scores_path.exists():
    #     print(f"obj_scores.parquet already exists at {obj_scores_path}. Skipping computation.")
    #     return

    # Ensure output dir exists
    obj_scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Run computation
    compute_rule_based_metrics(
        raw_path=str(raw_outputs_path),
        datasets_root=str(datasets_root),
        out_path=str(obj_scores_path),
        config=config,
        tasks=config.get("tasks", []),
        rouge_n=config.get("rouge_n", 2),
        use_deepeval=config.get("use_deepeval", True),
        compute_averages=config.get("compute_averages", True),
        verbose=config.get("verbose_mode", False)
    )
    print(f"Rule-based metrics computed and saved to {obj_scores_path}")  # Sin run_hash en print

if __name__ == "__main__":
    main()