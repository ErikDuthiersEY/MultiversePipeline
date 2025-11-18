import sys
from pathlib import Path

# Ajuste de path para proyecto grande
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import yaml
from typing import Dict, Any

from rule_based_metrics import compute_rule_based_metrics


def main():
    print("=== Starting Rule-Based Metrics Component ===")

    parser = argparse.ArgumentParser(description="Run rule-based metrics computation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Ruta al config principal (configs/config.yaml)")
    parser.add_argument("--raw_dir", default="outputs", help="Carpeta con raw_output.parquet")
    parser.add_argument("--raw_path", default=None, help="Ruta directa a raw_output.parquet")
    parser.add_argument("--datasets", default="datasets/processed", help="Carpeta con datasets de referencia")
    parser.add_argument("--out_dir", default="outputs", help="Carpeta de salida")
    args = parser.parse_args()

    print(f"[DEBUG] Argumentos cargados: {args}")

    config_path = Path(args.config)
    print(f"[DEBUG] Cargando config desde: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    print("[DEBUG] Config cargado correctamente.")
    print(f"[DEBUG] Claves principales: {list(config.keys())}")

    rlb_config = config.get("rule_based_metrics")
    if not rlb_config:
        raise KeyError("No se encontró la sección 'rule_based_metrics' en el config.yaml")

    # Paths
    raw_outputs_path = Path(args.raw_path) if args.raw_path else Path(args.raw_dir) / "raw_output.parquet"
    obj_scores_path = Path(args.out_dir) / "obj_scores.parquet"
    datasets_root = Path(args.datasets)

    print(f"[DEBUG] Input raw: {raw_outputs_path}")
    print(f"[DEBUG] Output métricas: {obj_scores_path}")
    print(f"[DEBUG] Datasets root: {datasets_root}")

    obj_scores_path.parent.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] Carpeta de salida asegurada.")

    compute_rule_based_metrics(
        raw_path=str(raw_outputs_path),
        datasets_root=str(datasets_root),
        out_path=str(obj_scores_path),
        config=config,
        tasks=rlb_config["tasks"],
        rouge_n=rlb_config.get("rouge_n", 2),
        use_deepeval=rlb_config.get("use_deepeval", True),
        compute_averages=rlb_config.get("compute_averages", True),
        verbose=rlb_config.get("verbose_mode", False)
    )

    print("=== Rule-Based Metrics Component finalizado ===")


if __name__ == "__main__":
    main()