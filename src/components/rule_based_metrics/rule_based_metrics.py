import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml  

from deepeval.metrics import ExactMatchMetric 
from rouge_score import rouge_scorer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 


def load_dataset(task: str, datasets_root: str) -> pd.DataFrame:
    """Load the reference dataset for a given task."""
    dataset_map = {
        "reasoning_closed": "reasoning_close.parquet",  
        "reasoning_open": "reasoning_open.parquet",
        "summarization": "summarization.parquet",
        "sensitivity_variations": "variation_sensivity_with_typos.parquet"  
    }
    if task not in dataset_map:
        raise ValueError(f"Unsupported task: {task}")
    
    file_path = Path(datasets_root) / dataset_map[task]
    df = pd.read_parquet(file_path)

    df['id'] = df['id'].astype(str)
    

    if task == "reasoning_closed":
        if 'gold_answer' in df.columns:
            df = df.rename(columns={'gold_answer': 'reference'})
    elif task == "reasoning_open":
        if 'reference_answer' in df.columns:
            df = df.rename(columns={'reference_answer': 'reference'})
    elif task == "summarization":
        if 'reference_summary' in df.columns:
            df = df.rename(columns={'reference_summary': 'reference'})
    elif task == "sensitivity_variations":

        pass
    
    if task == "sensitivity_variations":
        required_cols = ['id', 'prompt_original', 'prompt_paraphrase', 'prompt_typo_variant']
    else:
        required_cols = ['id', 'reference'] 
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset {task} missing required columns: {missing_cols}. Available: {list(df.columns)}")
    
    return df

def compute_exact_match(predictions: List[str], references: List[str], use_deepeval: bool = True) -> np.ndarray:
    """Compute Exact Match scores using DeepEval or simple string match."""
    if use_deepeval:
        metric = ExactMatchMetric(threshold=0.0, model=None)  
        scores = []
        for pred, ref in zip(predictions, references):
            metric.measure(test_case={'input': '', 'actual_output': pred, 'expected_output': ref})  
            scores.append(1.0 if metric.success else 0.0)
            metric.reset()  
        return np.array(scores)
    else:
        def normalize(s: str) -> str:
            return " ".join(s.lower().split())
        scores = [1.0 if normalize(p) == normalize(r) else 0.0 for p, r in zip(predictions, references)]
        return np.array(scores)

def compute_semantic_similarity(predictions: List[str], references: List[str]) -> np.ndarray:
    """Compute semantic similarity using TF-IDF + cosine (local alternative to BERTScore)."""
    all_texts = predictions + references
    vectorizer = TfidfVectorizer(stop_words=None, max_features=500)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    sim_matrix = cosine_similarity(tfidf_matrix[:len(predictions)], tfidf_matrix[len(predictions):])
    return np.diag(sim_matrix)

def compute_rouge(predictions: List[str], references: List[str], n: int = 2) -> np.ndarray:
    """Compute ROUGE-n (F1) scores."""
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred) 
        scores.append(score[f'rouge{n}'].fmeasure)
    return np.array(scores)

def compute_semantic_consistency(raw_df: pd.DataFrame, ref_df: pd.DataFrame) -> np.ndarray:
    """Compute Semantic Consistency Score (cosine sim between original and variant outputs using TF-IDF).
    Pairs: original vs paraphrase, original vs typo (avg per row)."""
    scores = []
    

    all_outputs = raw_df['output'].fillna('').tolist()
    if len(all_outputs) == 0:
        return np.array([])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_outputs)
    
    for _, row in ref_df.iterrows():

        orig_prompt = row['prompt_original']
        para_prompt = row['prompt_paraphrase']
        typo_prompt = row['prompt_typo_variant']
        
        orig_row = raw_df[raw_df['prompt'] == orig_prompt]
        para_row = raw_df[raw_df['prompt'] == para_prompt]
        typo_row = raw_df[raw_df['prompt'] == typo_prompt]
        
        if len(orig_row) != 1 or len(para_row) != 1 or len(typo_row) != 1:
            scores.append(0.0)  
            continue
        
        orig_output = orig_row.iloc[0]['output']
        para_output = para_row.iloc[0]['output']
        typo_output = typo_row.iloc[0]['output']
        
 
        orig_idx = raw_df[raw_df['prompt'] == orig_prompt].index[0]
        para_idx = raw_df[raw_df['prompt'] == para_prompt].index[0]
        typo_idx = raw_df[raw_df['prompt'] == typo_prompt].index[0]
        

        sim_para = cosine_similarity(tfidf_matrix[orig_idx:orig_idx+1], tfidf_matrix[para_idx:para_idx+1])[0][0]
  
        sim_typo = cosine_similarity(tfidf_matrix[orig_idx:orig_idx+1], tfidf_matrix[typo_idx:typo_idx+1])[0][0]
        

        avg_sim = (sim_para + sim_typo) / 2
        scores.append(avg_sim)
    
    return np.array(scores)

def process_task(raw_df_task: pd.DataFrame, ref_df: pd.DataFrame, task: str, rouge_n: int, use_deepeval: bool) -> Tuple[str, np.ndarray, np.ndarray]:
    """Process metrics for a single task, return task, scores, errors (std)."""
    if task == "sensitivity_variations":

        scores = compute_semantic_consistency(raw_df_task, ref_df)
    else:

        raw_df_task['prompt_id'] = raw_df_task['prompt_id'].astype(str)

        merged = raw_df_task.merge(ref_df[['id', 'reference']], left_on='prompt_id', right_on='id', how='inner')
        if len(merged) == 0:
            scores = np.array([])
        else:
            predictions = merged['output'].fillna('').tolist()
            references = merged['reference'].fillna('').tolist()
            
            if task == "reasoning_closed":
                scores = compute_exact_match(predictions, references, use_deepeval)
            elif task == "reasoning_open":
                scores = compute_semantic_similarity(predictions, references)
            elif task == "summarization":
                scores = compute_rouge(predictions, references, rouge_n)
            else:
                raise ValueError(f"Unsupported task: {task}")
    
    if len(scores) == 0:
        mean_score = 0.0
        std_score = 0.0
    else:
        mean_score = np.mean(scores)
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
    """Main function to compute rule-based metrics."""

    raw_df = pd.read_parquet(raw_path)
    
    if verbose:
            print("\n=== DEBUG: Loaded raw_df ===")
            print(f"[raw_df] shape: {raw_df.shape}")
            print(f"[raw_df] cols: {list(raw_df.columns)}")
            print(raw_df.head(5))
            print("==============================\n")

    raw_df['prompt_id'] = raw_df['prompt_id'].astype(str)

    results = []
    models = raw_df['model'].unique()
    
    for model in models:
        if verbose:
            print(f"\n--- Processing model: {model} ---")

        model_results = {'model': model}
        model_raw = raw_df[raw_df['model'] == model]
        
        for task in tasks:
            if verbose:
                print(f"\nProcessing task: {task} for model {model}")
            
            task_raw = model_raw[model_raw['task'] == task]
            if len(task_raw) == 0:
                if verbose:
                    print(f"[DEBUG] No data found for task '{task}' in model '{model}'. Skipping.")
                continue
            
            ref_df = load_dataset(task, datasets_root)

            if verbose:
                print(f"=== DEBUG: Loaded ref_df for task: {task} ===")
                print(f"[ref_df] shape: {ref_df.shape}")
                print(f"[ref_df] cols: {list(ref_df.columns)}")
                print(ref_df.head(5))
                print("============================================\n")
            
            task_key, mean_score, std_score = process_task(task_raw, ref_df, task, rouge_n, use_deepeval)
            
            if compute_averages:
                model_results[f"{task_key}_avg"] = mean_score
                model_results[f"{task_key}_delta"] = std_score
        
        results.append(model_results)
    

    if not results:
        raise ValueError("No results computed; check inputs.")
    
    out_df = pd.DataFrame(results)

    metric_cols = [col for col in out_df.columns if col != 'model']
    metric_cols.sort()
    out_df = out_df[['model'] + metric_cols]
    
    out_df.to_parquet(out_path, index=False)
    if verbose:
        print("\n=== DEBUG: Final metrics dataframe ===")
        print(out_df.head())
        print(f"Saved metrics to {out_path}")
        print("======================================\n")