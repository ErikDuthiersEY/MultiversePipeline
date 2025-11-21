"""
Pipeline utilities for rule-based metrics:
- Azure embeddings helper
- Dataset loading
- Task processing
- Top-level compute_rule_based_metrics
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from openai import AzureOpenAI

DEBUG = True

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== AZURE EMBEDDINGS ====================

def get_azure_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Obtiene embeddings usando Azure OpenAI text-embedding-ada-002.
    """
    azure_config = config["azure"]

    client = AzureOpenAI(
        api_key=azure_config["api_key"],
        api_version=azure_config["api_version"],
        azure_endpoint=azure_config["endpoint"]
    )

    batch_size = 100
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [text if text.strip() else " " for text in batch]

        try:
            response = client.embeddings.create(
                input=batch,
                model=azure_config["embeddings_id"]
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if DEBUG:
                logger.info(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(batch)} textos procesados")

        except Exception as e:
            logger.warning(f"Error obteniendo embeddings para batch {i//batch_size}: {e}")
            embeddings.extend([[0.0] * 1536] * len(batch))

    return np.array(embeddings)


# ==================== DATASET LOADING ====================

def load_dataset(task: str, datasets_root: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Carga y estandariza el dataset de referencia usando la configuracion.
    """
    rlb = config.get("rule_based_metrics", {})

    filename = rlb.get("dataset_files", {}).get(task)
    if not filename:
        raise ValueError(
            f"Task '{task}' no tiene 'dataset_files' definido en config.yaml. "
            f"Tasks disponibles: {list(rlb.get('dataset_files', {}).keys())}"
        )

    file_path = Path(datasets_root) / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {file_path}\n"
            f"Verifica que el archivo existe en {datasets_root}"
        )

    df = pd.read_parquet(file_path)
    df["id"] = df["id"].astype(str)

    if task == "variation_sensitivity":
        if "group_id" not in df.columns:
            df["group_id"] = df["id"].str.replace(r"_orig|_para|_typo", "", regex=True)
        df["group_id"] = df["group_id"].astype(str)

    mapping = rlb.get("column_mapping", {}).get(task, {})
    if mapping:
        df = df.rename(columns=mapping)

    required = rlb.get("required_columns", {}).get(task, [])
    if task == "variation_sensitivity":
        required = ["id", "prompt", "group_id"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{task}' ({filename}) falta columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    logger.info(f"Dataset '{task}' cargado correctamente con {len(df)} filas")
    return df


# ==================== TASK PROCESSING ====================

def process_task(
    raw_df_task: pd.DataFrame,
    ref_df: pd.DataFrame,
    task: str,
    config: Dict[str, Any]
) -> Tuple[str, float, float]:
    """
    Procesa una tarea especifica y calcula sus metricas.
    """
    import rule_based_metrics as metrics  # Lazy import to avoid circular dependency

    raw_df_task = raw_df_task.copy()

    if task == "variation_sensitivity":
        scores = metrics.compute_semantic_consistency(raw_df_task, config)
        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

    elif task == "refusal_correctness":
        merged = raw_df_task.merge(
            ref_df[["id", "safety_label"]],
            left_on="prompt_id",
            right_on="id",
            how="inner"
        )
        unsafe_df = merged[merged["safety_label"] == "unsafe"]

        if len(unsafe_df) == 0:
            logger.warning("No hay prompts unsafe para evaluar refusal correctness")
            scores = np.array([])
        else:
            scores = metrics.compute_refusal_correctness(unsafe_df)

        if len(scores) > 0 and np.any(~np.isnan(scores)):
            mean_score = float(np.nanmean(scores))
            std_score = float(np.nanstd(scores))
        else:
            mean_score = 0.0
            std_score = 0.0

    else:
        raw_df_task["prompt_id"] = raw_df_task["prompt_id"].astype(str)

        if task == "summarization":
            reference_col = "reference_summary" if "reference_summary" in ref_df.columns else "reference"

            required_cols = ["id", reference_col]
            if "prompt" in ref_df.columns:
                required_cols.append("prompt")
            if "category" in ref_df.columns:
                required_cols.append("category")

            merged = raw_df_task.merge(
                ref_df[required_cols],
                left_on="prompt_id",
                right_on="id",
                how="inner",
                suffixes=('_raw', '_ref')
            )

            if "reference_summary" in merged.columns:
                merged = merged.rename(columns={"reference_summary": "reference"})

        else:
            merged = raw_df_task.merge(
                ref_df[["id", "reference", "category"]],
                left_on="prompt_id",
                right_on="id",
                how="inner"
            )

        if len(merged) == 0:
            logger.warning(f"No hay coincidencias entre raw_output y dataset de referencia para {task}")
            scores = np.array([])
        else:
            predictions = merged["output"].fillna("").tolist()

            if task == "reasoning_close":
                categories = merged["category"].fillna("").tolist()
                references = merged["reference"].fillna("").tolist()
                scores = metrics.compute_exact_match(predictions, references, categories)

            elif task == "reasoning_open":
                references = merged["reference"].fillna("").tolist()
                scores = metrics.compute_semantic_similarity(predictions, references, config)

            elif task == "summarization":
                references = merged["reference"].fillna("").tolist()

                if "prompt_ref" in merged.columns:
                    sources = merged["prompt_ref"].fillna("").tolist()
                    if DEBUG:
                        logger.debug("Usando 'prompt_ref' como articulo fuente")
                elif "prompt_raw" in merged.columns:
                    sources = merged["prompt_raw"].fillna("").tolist()
                    if DEBUG:
                        logger.debug("Usando 'prompt_raw' como articulo fuente")
                elif "prompt" in merged.columns:
                    sources = merged["prompt"].fillna("").tolist()
                    if DEBUG:
                        logger.debug("Usando 'prompt' como articulo fuente")
                else:
                    raise KeyError(
                        f"No se encontro columna 'prompt' en el merge para summarization.\n"
                        f"Columnas disponibles: {list(merged.columns)}"
                    )

                scores = metrics.compute_rouge(predictions, references, sources, config)

            else:
                raise ValueError(
                    f"Task '{task}' no esta soportada. "
                    f"Tasks validas: reasoning_close, reasoning_open, summarization, "
                    f"variation_sensitivity, refusal_correctness"
                )

        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

    return task, mean_score, std_score


# ==================== MAIN COMPUTATION FUNCTION ====================

def compute_rule_based_metrics(
    raw_path: str,
    datasets_root: str,
    out_path: str,
    config: Dict[str, Any],
    tasks: List[str],
    compute_averages: bool = True,
    verbose: bool = False
) -> None:
    """
    Funcion principal para calcular todas las metricas rule-based.
    """
    logger.info("=== Iniciando calculo de metricas rule-based con Azure embeddings ===")

    raw_path_obj = Path(raw_path)
    if not raw_path_obj.exists():
        raise FileNotFoundError(f"Archivo raw_output no encontrado: {raw_path}")

    raw_df = pd.read_parquet(raw_path)

    if verbose:
        logger.info("\n=== DEBUG: Loaded raw_df ===")
        logger.info(f"[raw_df] shape: {raw_df.shape}")
        logger.info(f"[raw_df] cols: {list(raw_df.columns)}")
        logger.info(f"\n{raw_df.head(5).to_string()}")
        logger.info("=" * 50)

    required_cols = ["model", "task", "prompt_id", "output"]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"raw_output.parquet falta columnas requeridas: {missing_cols}\n"
            f"Columnas disponibles: {list(raw_df.columns)}"
        )

    raw_df["prompt_id"] = raw_df["prompt_id"].astype(str)
    results = []
    models = raw_df["model"].unique()

    logger.info(f"Modelos a procesar: {list(models)}")
    logger.info(f"Tasks a procesar: {tasks}")

    for model in models:
        logger.info("\n" + "=" * 60)
        logger.info(f"Procesando modelo: {model}")
        logger.info("=" * 60)

        model_results = {"model": model}
        model_raw = raw_df[raw_df["model"] == model]

        for task in tasks:
            logger.info(f"\n-> Procesando task: {task}")

            task_raw = model_raw[model_raw["task"] == task]
            if task_raw.empty:
                logger.warning(f"No hay datos para task '{task}' en modelo '{model}'. Saltando.")
                continue

            try:
                ref_df = load_dataset(task, datasets_root, config)

                if verbose:
                    logger.info(f"\n=== DEBUG: ref_df para task '{task}' ===")
                    logger.info(f"[ref_df] shape: {ref_df.shape}")
                    logger.info(f"[ref_df] cols: {list(ref_df.columns)}")
                    logger.info(f"\n{ref_df.head(3).to_string()}")
                    logger.info("=" * 50)

                task_key, mean_score, std_score = process_task(task_raw, ref_df, task, config)

                if compute_averages:
                    model_results[f"{task_key}_avg"] = round(float(mean_score), 4)
                    model_results[f"{task_key}_delta"] = round(float(std_score), 4)

                logger.info(f"OK {task_key}: {mean_score:.4f} +- {std_score:.4f}")

            except Exception as e:
                logger.error(f"Error procesando {task} para {model}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue

        results.append(model_results)

    if not results:
        raise ValueError(
            "No se generaron resultados. Verifica:\n"
            "1. raw_output.parquet contiene datos\n"
            "2. Las tasks estan configuradas correctamente\n"
            "3. Los datasets de referencia existen"
        )

    out_df = pd.DataFrame(results)
    metric_cols = sorted([c for c in out_df.columns if c != "model"])
    out_df = out_df[["model"] + metric_cols]

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("Metricas rule-based calculadas correctamente")
    logger.info(f"Archivo guardado en: {out_path}")
    logger.info("=" * 60)

    if verbose:
        logger.info("\n=== Resultado final ===")
        logger.info(f"\n{out_df.to_string(index=False)}")
        logger.info("=" * 50)
