"""
CLI runner for rule-based metrics computation.
Loads configuration, resolves paths, and invokes the pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure project root on sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rule_based_pipeline import compute_rule_based_metrics  # noqa: E402
import logging

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based metrics computation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Ruta al config principal (configs/config.yaml)")
    parser.add_argument("--raw_dir", default="outputs", help="Carpeta con raw_output.parquet")
    parser.add_argument("--raw_path", default=None, help="Ruta directa a raw_output.parquet")
    parser.add_argument("--datasets", default="datasets/processed", help="Carpeta con datasets de referencia")
    parser.add_argument("--out_dir", default="outputs", help="Carpeta de salida")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    # Configure logging early; verbose is derived after reading config
    configure_logging(verbose=False)
    logger.info("=== Starting Rule-Based Metrics Component ===")
    logger.debug("Argumentos cargados: %s", args)
    logger.debug("Cargando config desde: %s", config_path)

    config = load_config(config_path)
    logger.debug("Config cargado correctamente. Claves principales: %s", list(config.keys()))

    rlb_config = config.get("rule_based_metrics")
    if not rlb_config:
        raise KeyError("No se encontró la sección 'rule_based_metrics' en el config.yaml")

    # Re-configure logging if verbose_mode is set in config
    configure_logging(verbose=rlb_config.get("verbose_mode", False))

    raw_outputs_path = Path(args.raw_path) if args.raw_path else Path(args.raw_dir) / "raw_output.parquet"
    obj_scores_path = Path(args.out_dir) / "obj_scores.parquet"
    datasets_root = Path(args.datasets)

    logger.info("Input raw: %s", raw_outputs_path)
    logger.info("Output métricas: %s", obj_scores_path)
    logger.info("Datasets root: %s", datasets_root)

    obj_scores_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Carpeta de salida asegurada.")

    compute_rule_based_metrics(
        raw_path=str(raw_outputs_path),
        datasets_root=str(datasets_root),
        out_path=str(obj_scores_path),
        config=config,
        tasks=rlb_config["tasks"],
        compute_averages=rlb_config.get("compute_averages", True),
        verbose=rlb_config.get("verbose_mode", False),
    )

    logger.info("=== Rule-Based Metrics Component finalizado ===")


if __name__ == "__main__":
    main()
