import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import unicodedata
import string

from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(task: str, datasets_root: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Carga y estandariza el dataset de referencia usando solo el config.yaml."""
    rlb = config.get("rule_based_metrics", {})

    # --- Archivo ---
    filename = rlb.get("dataset_files", {}).get(task)
    if not filename:
        raise ValueError(f"[ERROR] Task '{task}' no tiene 'dataset_files' definido en config.yaml")

    file_path = Path(datasets_root) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset no encontrado: {file_path}")

    df = pd.read_parquet(file_path)
    df["id"] = df["id"].astype(str)

    # --- Renombrado de columnas ---
    mapping = rlb.get("column_mapping", {}).get(task, {})
    if mapping:
        old_cols = [col for col in mapping.keys() if col in df.columns]
        if old_cols:
            print(f"[DEBUG] Task '{task}' → renombrando columnas: {old_cols} → 'reference'")
            df = df.rename(columns=mapping)

    # --- Validación de columnas requeridas ---
    required = rlb.get("required_columns", {}).get(task, [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Dataset '{task}' ({filename}) falta columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    print(f"[DEBUG] Dataset '{task}' cargado correctamente → {len(df)} filas")
    return df


def compute_exact_match(predictions: List[str], references: List[str], use_deepeval: bool = True) -> np.ndarray:
    def normalize(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s)
        s = s.lower()
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.replace("\n", " ").replace("\r", " ")
        s = "".join(c for c in s if c not in string.punctuation)
        s = " ".join(s.split())
        return s

    scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize(pred)
        ref_norm = normalize(ref)
        metric = ExactMatchMetric(threshold=1.0)
        tc = LLMTestCase(input="", actual_output=pred_norm, expected_output=ref_norm)
        metric.measure(tc)
        scores.append(metric.score)

    return np.array(scores)


def compute_semantic_similarity(predictions: List[str], references: List[str]) -> np.ndarray:
    all_texts = predictions + references
    vectorizer = TfidfVectorizer(stop_words=None, max_features=500)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    sim_matrix = cosine_similarity(tfidf_matrix[:len(predictions)], tfidf_matrix[len(predictions):])
    return np.diag(sim_matrix)


def compute_rouge(predictions: List[str], references: List[str]) -> np.ndarray:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result['rougeL'].fmeasure)
    return np.array(scores)


def compute_semantic_consistency(raw_df: pd.DataFrame, ref_df: pd.DataFrame) -> np.ndarray:
    def normalize_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s)
        s = s.lower()
        s = s.replace("\n", " ").replace("\r", " ")
        s = " ".join(s.split())
        return s

    raw_df = raw_df.copy()
    raw_df["output"] = raw_df["output"].fillna("").apply(normalize_text)
    all_outputs = raw_df["output"].tolist()

    if len(all_outputs) == 0:
        return np.array([])

    vectorizer = TfidfVectorizer(stop_words=None, max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(all_outputs)

    scores = []
    for _, row in ref_df.iterrows():
        orig_p = row["prompt_original"]
        para_p = row["prompt_paraphrase"]
        typo_p = row["prompt_typo_variant"]

        orig_row = raw_df[raw_df["prompt"] == orig_p]
        para_row = raw_df[raw_df["prompt"] == para_p]
        typo_row = raw_df[raw_df["prompt"] == typo_p]

        if not (len(orig_row) == 1 and len(para_row) == 1 and len(typo_row) == 1):
            scores.append(0.0)
            continue

        orig_idx = orig_row.index[0]
        para_idx = para_row.index[0]
        typo_idx = typo_row.index[0]

        sim_para = cosine_similarity(tfidf_matrix[orig_idx:orig_idx+1], tfidf_matrix[para_idx:para_idx+1])[0][0]
        sim_typo = cosine_similarity(tfidf_matrix[orig_idx:orig_idx+1], tfidf_matrix[typo_idx:typo_idx+1])[0][0]
        avg_sim = (sim_para + sim_typo) / 2
        scores.append(avg_sim)

    return np.array(scores)


def process_task(raw_df_task: pd.DataFrame, ref_df: pd.DataFrame, task: str, rouge_n: int, use_deepeval: bool) -> Tuple[str, float, float]:
    if task == "sensitivity_variations":
        scores = compute_semantic_consistency(raw_df_task, ref_df)
    else:
        raw_df_task["prompt_id"] = raw_df_task["prompt_id"].astype(str)
        merged = raw_df_task.merge(ref_df[["id", "reference"]], left_on="prompt_id", right_on="id", how="inner")
        if len(merged) == 0:
            scores = np.array([])
        else:
            predictions = merged["output"].fillna("").tolist()
            references = merged["reference"].fillna("").tolist()

            if task == "reasoning_closed":
                scores = compute_exact_match(predictions, references, use_deepeval)
            elif task == "reasoning_open":
                scores = compute_semantic_similarity(predictions, references)
            elif task == "summarization":
                scores = compute_rouge(predictions, references)
            else:
                raise ValueError(f"Task no soportada: {task}")

    mean_score = np.mean(scores) if len(scores) > 0 else 0.0
    std_score = np.std(scores) if len(scores) > 1 else 0.0
    return task, mean_score, std_score


def compute_rule_based_metrics(
    raw_path: str,
    datasets_root: str,
    out_path: str,
    config: Dict[str, Any],
    tasks: List[str],
    rouge_n: int,
    use_deepeval: bool,
    compute_averages: bool = True,
    verbose: bool = False
) -> None:
    print("\n=== Iniciando cálculo de métricas rule-based ===")
    raw_df = pd.read_parquet(raw_path)

    if verbose:
        print("\n=== DEBUG: Loaded raw_df ===")
        print(f"[raw_df] shape: {raw_df.shape}")
        print(f"[raw_df] cols: {list(raw_df.columns)}")
        print(raw_df.head(5))
        print("==============================\n")

    raw_df["prompt_id"] = raw_df["prompt_id"].astype(str)
    results = []
    models = raw_df["model"].unique()

    for model in models:
        if verbose:
            print(f"\n--- Procesando modelo: {model} ---")

        model_results = {"model": model}
        model_raw = raw_df[raw_df["model"] == model]

        for task in tasks:
            if verbose:
                print(f"\nProcesando task: {task} para modelo {model}")

            task_raw = model_raw[model_raw["task"] == task]
            if task_raw.empty:
                if verbose:
                    print(f"[DEBUG] No hay datos para task '{task}' en modelo '{model}'. Saltando.")
                continue

            ref_df = load_dataset(task, datasets_root, config)

            if verbose:
                print(f"=== DEBUG: ref_df para task '{task}' ===")
                print(f"[ref_df] shape: {ref_df.shape}")
                print(f"[ref_df] cols: {list(ref_df.columns)}")
                print(ref_df.head(3))
                print("============================================\n")

            task_key, mean_score, std_score = process_task(task_raw, ref_df, task, rouge_n, use_deepeval)

            if compute_averages:
                model_results[f"{task_key}_avg"] = round(float(mean_score), 4)
                model_results[f"{task_key}_delta"] = round(float(std_score), 4)

            if verbose:
                print(f"→ {task_key}: {mean_score:.4f} ± {std_score:.4f}")

        results.append(model_results)

    if not results:
        raise ValueError("No se generaron resultados. Revisa raw_output.parquet y las tasks.")

    out_df = pd.DataFrame(results)
    metric_cols = sorted([c for c in out_df.columns if c != "model"])
    out_df = out_df[["model"] + metric_cols]

    out_df.to_parquet(out_path, index=False)
    print(f"\nMétricas rule-based calculadas y guardadas en:\n→ {out_path}")
    if verbose:
        print("\n=== Resultado final ===")
        print(out_df.to_string(index=False))
        print("========================\n")