import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import yaml
from typing import Dict, Any

#from src.utils.hashing import compute_run_hash
from rule_based_metrics import compute_rule_based_metrics

def main():
    print("=== Starting Rule-Based Metrics Component ===")

    parser = argparse.ArgumentParser(description="Run rule-based metrics computation.")
    parser.add_argument("--config", default="configs/rule_based_metrics.yaml", help="Path to rule_based_metrics.yaml")  
    parser.add_argument("--raw_dir", default="outputs", help="Input dir for raw_output.parquet from inference")
    parser.add_argument("--raw_path", default=None, help="Direct path to raw_output.parquet (overrides raw_dir)") 
    parser.add_argument("--datasets", default="datasets/processed", help="Path to datasets dir")
    parser.add_argument("--out_dir", default="outputs", help="Output dir for obj_scores.parquet")  
    parser.add_argument("--dataset_version", default="latest", help="Dataset version")
    parser.add_argument("--code_version", default="v0.1", help="Code version (e.g., from git)")
    args = parser.parse_args()

    print(f"[DEBUG] Loaded arguments: {args}")
   
    config_path = Path(args.config)
    print(f"[DEBUG] Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    
    print("[DEBUG] Config loaded successfully.")
    print(f"[DEBUG] Config contents: {config}")

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

    # Paths  # Usa raw_dir para input, out_dir fijo sin {run_hash}
    if args.raw_path:
        raw_outputs_path = Path(args.raw_path) 
        print(f"[DEBUG] Using explicit raw_path: {raw_outputs_path}")
    else:
        raw_outputs_path = Path(args.raw_dir) / "raw_output.parquet"  
        print(f"[DEBUG] Using raw_dir path: {raw_outputs_path}")

    obj_scores_path = Path(args.out_dir) / "obj_scores.parquet"  
    print(f"[DEBUG] Metrics will be saved to: {obj_scores_path}")

    datasets_root = Path(args.datasets)
    print(f"[DEBUG] Datasets directory: {datasets_root}")

    # Check if already exists  # Desactivado para siempre sobrescribir
    # if obj_scores_path.exists():
    #     print(f"obj_scores.parquet already exists at {obj_scores_path}. Skipping computation.")
    #     return

    obj_scores_path.parent.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] Ensured output directory exists.")
    rlb_config = config["rule_based_metrics"]
    compute_rule_based_metrics(
        raw_path=str(raw_outputs_path),
        datasets_root=str(datasets_root),
        out_path=str(obj_scores_path),
        config=config,
        tasks=rlb_config["tasks"],
        rouge_n=rlb_config["rouge_n"],
        use_deepeval=rlb_config["use_deepeval"],
        compute_averages=rlb_config["compute_averages"],
        verbose=rlb_config["verbose_mode"]
    )
    print(f"Rule-based metrics computed and saved to {obj_scores_path}")

if __name__ == "__main__":
    main()