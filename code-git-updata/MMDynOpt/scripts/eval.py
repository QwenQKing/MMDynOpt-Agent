#!/usr/bin/env python3

import re
import unicodedata
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

DEFAULT_RES_DIR = "eval_output_all"

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def f1_score(pred: str, gold_list) -> float:
    if isinstance(gold_list, str):
        gt_list = [gold_list]
    elif isinstance(gold_list, (list, tuple)):
        gt_list = [str(x) for x in gold_list] or [""]
    else:
        gt_list = [str(gold_list)]
    pred_tokens = normalize_answer(str(pred)).split()
    if not pred_tokens:
        return 0.0
    max_f1 = 0.0
    for gt in gt_list:
        gold_tokens = normalize_answer(str(gt)).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


def exact_match(pred, gold_list):
    if not isinstance(gold_list, list):
        gold_list = [gold_list]
    return any(normalize_answer(str(pred)) == normalize_answer(str(g)) for g in gold_list if g)


def substring_match(pred, gold_list):
    if not isinstance(gold_list, list):
        gold_list = [gold_list]
    p = normalize_answer(str(pred))
    return any(normalize_answer(str(g)) in p or p in normalize_answer(str(g)) for g in gold_list if g)


class SemanticSimilarity:
    def __init__(self):
        self.backend = "none"
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
            self.backend = "sbert"
            project_root = Path(__file__).resolve().parents[2]
            model_dir = project_root / "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(str(model_dir))
            self.st_util = st_util
            print(f"[INFO] Using sentence-transformers ({model_dir})")
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                self.backend = "tfidf"
                self.vectorizer = TfidfVectorizer()
                self.cosine_similarity = cosine_similarity
                print("[INFO] Using TF-IDF for semantic similarity")
            except Exception:
                self.backend = "none"
                print("[INFO] Using lexical-overlap based similarity")

    def pair(self, a: str, b: str) -> float:
        a, b = a.strip(), b.strip()
        if not a or not b:
            return 1.0

        if self.backend == "sbert":
            ea = self.model.encode([a], convert_to_tensor=True, normalize_embeddings=True)
            eb = self.model.encode([b], convert_to_tensor=True, normalize_embeddings=True)
            return float(self.st_util.cos_sim(ea, eb).item())
        elif self.backend == "tfidf":
            try:
                X = self.vectorizer.fit_transform([a, b])
                if X.shape[1] == 0:
                    return 1.0
                return float(self.cosine_similarity(X[0], X[1]).item())
            except ValueError:
                return 1.0
        else:
            sa = set(normalize_answer(a).split())
            sb = set(normalize_answer(b).split())
            if not sa or not sb:
                return 1.0
            return len(sa & sb) / len(sa | sb)


def evaluate_batch(results: List[Dict[str, Any]], compute_semantic: bool = True) -> Dict[str, Any]:
    sim_calculator = SemanticSimilarity() if compute_semantic else None

    em_sum, sm_sum = 0, 0
    f1_list, sem_list = [], []
    n_interactions_list = []
    llm_input_tokens_list = []
    llm_output_tokens_list = []
    agent_generated_tokens_list = []

    for item in tqdm(results, desc="Evaluating", leave=False):
        pred = item.get("predicted_answer") or ""
        golds = item.get("ground_truth", [])
        if not isinstance(golds, list):
            golds = [str(golds)]

        em_sum += 1 if exact_match(pred, golds) else 0
        sm_sum += 1 if substring_match(pred, golds) else 0
        f1_list.append(f1_score(pred, golds))
        if compute_semantic:
            max_semantic = 0.0
            if sim_calculator:
                for gold in golds:
                    max_semantic = max(max_semantic, sim_calculator.pair(pred, str(gold)))
            sem_list.append(max_semantic)

        n_interactions_list.append(item.get("n_interactions", 0))
        llm_input_tokens_list.append(item.get("llm_input_tokens", 0))
        llm_output_tokens_list.append(item.get("llm_output_tokens", 0))
        agent_generated_tokens_list.append(item.get("agent_output_tokens_sum", 0))

    n = len(results)
    safe_mean = lambda lst: sum(lst) / n if n > 0 else 0.0

    metrics = {
        "count": n,
        "exact_match": {"mean": em_sum / n if n > 0 else 0.0, "sum": em_sum},
        "substring_match": {"mean": sm_sum / n if n > 0 else 0.0, "sum": sm_sum},
        "f1": {"mean": safe_mean(f1_list)},
        "n_interactions": {"mean": safe_mean(n_interactions_list), "total": sum(n_interactions_list)},
        "llm_input_tokens": {"mean": safe_mean(llm_input_tokens_list), "total": sum(llm_input_tokens_list)},
        "llm_output_tokens": {"mean": safe_mean(llm_output_tokens_list), "total": sum(llm_output_tokens_list)},
        "agent_generated_tokens": {"mean": safe_mean(agent_generated_tokens_list), "total": sum(agent_generated_tokens_list)},
    }

    if compute_semantic:
        metrics["semantic_similarity"] = {"mean": safe_mean(sem_list)}

    return metrics


def process_single_dataset(res_json_path: Path, compute_semantic: bool) -> Dict[str, Any]:
    try:
        with open(res_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {res_json_path}: {e}")
        return None

    if not isinstance(data, list):
        print(f"[ERROR] JSON file should contain a list: {res_json_path}")
        return None

    metrics = evaluate_batch(data, compute_semantic=compute_semantic)

    eval_json_path = res_json_path.parent / "eval.json"
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    tqdm.write(f"  -> eval.json saved: {eval_json_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation metrics -> CSV")
    parser.add_argument("--res-dir", type=str, default=DEFAULT_RES_DIR,
                        help=f"Results directory (default: {DEFAULT_RES_DIR})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename (default: auto-generated)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip semantic similarity computation")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for CSV LLM column")
    args = parser.parse_args()

    res_dir = Path(args.res_dir)
    compute_semantic = not args.no_semantic

    if not res_dir.exists():
        print(f"[ERROR] Directory not found: {res_dir}")
        return

    dataset_folders = sorted([d for d in res_dir.iterdir() if d.is_dir() and (d / "res.json").exists()])

    if not dataset_folders:
        print(f"[ERROR] No dataset folders with res.json found under {res_dir}")
        return

    print(f"[INFO] Found {len(dataset_folders)} dataset folders")
    print("=" * 80)

    all_csv_rows = []

    for dataset_folder in tqdm(dataset_folders, desc="Processing datasets"):
        res_json = dataset_folder / "res.json"
        dataset_name = dataset_folder.name

        metrics = process_single_dataset(res_json, compute_semantic)
        if metrics is None:
            continue

        model_name = args.model_name or res_dir.name

        row = {
            "LLM": model_name,
            "dataset": dataset_name,
            "count": metrics["count"],
            "exact_match": f"{metrics['exact_match']['mean']:.4f}",
            "substring_match": f"{metrics['substring_match']['mean']:.4f}",
            "f1": f"{metrics['f1']['mean']:.4f}",
            "avg_interactions": f"{metrics['n_interactions']['mean']:.2f}",
            "avg_llm_input_tokens": f"{metrics['llm_input_tokens']['mean']:.1f}",
            "avg_llm_output_tokens": f"{metrics['llm_output_tokens']['mean']:.1f}",
            "avg_agent_generated_tokens": f"{metrics['agent_generated_tokens']['mean']:.1f}",
        }

        if compute_semantic:
            row["semantic_similarity"] = f"{metrics['semantic_similarity']['mean']:.4f}"

        all_csv_rows.append(row)

        tqdm.write(
            f"  OK {dataset_name}: EM={metrics['exact_match']['mean']:.4f}  "
            f"F1={metrics['f1']['mean']:.4f}  "
            f"Interactions={metrics['n_interactions']['mean']:.2f}"
        )

    print("=" * 80)

    if not all_csv_rows:
        print("[ERROR] No datasets successfully processed")
        return

    csv_filename = args.output or f"{args.model_name or res_dir.name}_metrics.csv"
    csv_path = res_dir / csv_filename

    fieldnames = [
        "LLM", "dataset", "count",
        "exact_match", "substring_match", "f1",
        "avg_interactions", "avg_llm_input_tokens", "avg_llm_output_tokens", "avg_agent_generated_tokens",
    ]
    if compute_semantic:
        fieldnames.append("semantic_similarity")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\n[SUCCESS] Results saved to: {csv_path}")
    print(f"[INFO] Processed {len(all_csv_rows)} datasets")

    print("\n" + "=" * 80)
    print("Summary:")
    n_ds = len(all_csv_rows)
    print(f"  Avg EM:              {sum(float(r['exact_match']) for r in all_csv_rows) / n_ds:.4f}")
    print(f"  Avg F1:              {sum(float(r['f1']) for r in all_csv_rows) / n_ds:.4f}")
    print(f"  Avg SubStr Match:    {sum(float(r['substring_match']) for r in all_csv_rows) / n_ds:.4f}")
    print(f"  Avg Interactions:    {sum(float(r['avg_interactions']) for r in all_csv_rows) / n_ds:.2f}")
    print(f"  Avg LLM Input Tok:   {sum(float(r['avg_llm_input_tokens']) for r in all_csv_rows) / n_ds:.1f}")
    print(f"  Avg LLM Output Tok:  {sum(float(r['avg_llm_output_tokens']) for r in all_csv_rows) / n_ds:.1f}")
    print(f"  Avg Agent Gen Tok:   {sum(float(r['avg_agent_generated_tokens']) for r in all_csv_rows) / n_ds:.1f}")
    if compute_semantic:
        print(f"  Avg Semantic Sim:    {sum(float(r['semantic_similarity']) for r in all_csv_rows) / n_ds:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
