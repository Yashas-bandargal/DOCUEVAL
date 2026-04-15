"""
eval_retrieval.py
-----------------
Evaluates the retrieval component of the RAG pipeline.

Metrics computed:
  - Recall@K    : Did we retrieve the relevant document in top K?
  - Precision@K : Of K retrieved, how many are relevant?
  - MRR         : Mean Reciprocal Rank — where does the relevant chunk appear?
  - Hit Rate@K  : Binary — did at least 1 relevant chunk appear in top K?

Saves results to: results/retrieval_scores.json

Run:
    python src/eval_retrieval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retrieve import retrieve

# ── Config ────────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "results/retrieval_scores.json"
K_VALUES = [3, 5]   # evaluate at K=3 and K=5
# ─────────────────────────────────────────────────────────────────────────────


def is_relevant(chunk, expected_source):
    """
    A chunk is considered relevant if it comes from the expected source document.
    This is our ground truth relevance signal.
    """
    return chunk["source"] == expected_source


def recall_at_k(retrieved_chunks, expected_source, k):
    """
    Recall@K = number of relevant docs retrieved in top K / total relevant docs.
    Since we have exactly 1 relevant source per query, this is 1 or 0.
    """
    top_k = retrieved_chunks[:k]
    relevant_found = sum(1 for c in top_k if is_relevant(c, expected_source))
    # total relevant = all chunks from the source (we treat it as 1 for simplicity)
    return min(relevant_found, 1)


def precision_at_k(retrieved_chunks, expected_source, k):
    """
    Precision@K = number of relevant docs in top K / K
    """
    top_k = retrieved_chunks[:k]
    relevant_found = sum(1 for c in top_k if is_relevant(c, expected_source))
    return relevant_found / k


def reciprocal_rank(retrieved_chunks, expected_source):
    """
    Reciprocal Rank = 1 / rank of first relevant result.
    If no relevant result found, returns 0.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_relevant(chunk, expected_source):
            return 1 / rank
    return 0.0


def hit_rate_at_k(retrieved_chunks, expected_source, k):
    """
    Hit Rate@K = 1 if at least one relevant chunk in top K, else 0.
    """
    top_k = retrieved_chunks[:k]
    return 1 if any(is_relevant(c, expected_source) for c in top_k) else 0


def evaluate_retrieval():
    print("\n=== Retrieval Evaluation ===\n")
    os.makedirs("results", exist_ok=True)

    # Load ground truth (skip out-of-scope queries)
    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    in_scope_queries = [q for q in ground_truth if not q.get("out_of_scope", False)]
    print(f"Evaluating {len(in_scope_queries)} in-scope queries at K = {K_VALUES}\n")

    per_query_results = []

    # Metrics accumulators
    metrics = {k: {"recall": [], "precision": [], "rr": [], "hit_rate": []} for k in K_VALUES}

    for q in in_scope_queries:
        query_id = q["id"]
        query_text = q["query"]
        expected_source = q["relevant_source"]

        # Retrieve with max K
        max_k = max(K_VALUES)
        retrieved = retrieve(query_text, k=max_k)

        query_result = {
            "query_id": query_id,
            "query": query_text,
            "expected_source": expected_source,
            "retrieved_sources": [c["source"] for c in retrieved],
            "scores": {}
        }

        for k in K_VALUES:
            rec = recall_at_k(retrieved, expected_source, k)
            prec = precision_at_k(retrieved, expected_source, k)
            rr = reciprocal_rank(retrieved[:k], expected_source)
            hr = hit_rate_at_k(retrieved, expected_source, k)

            query_result["scores"][f"k={k}"] = {
                "recall": rec,
                "precision": round(prec, 4),
                "reciprocal_rank": round(rr, 4),
                "hit_rate": hr
            }

            metrics[k]["recall"].append(rec)
            metrics[k]["precision"].append(prec)
            metrics[k]["rr"].append(rr)
            metrics[k]["hit_rate"].append(hr)

        per_query_results.append(query_result)
        print(f"  [{query_id}] Hit@5={query_result['scores']['k=5']['hit_rate']} | "
              f"MRR@5={query_result['scores']['k=5']['reciprocal_rank']:.2f} | "
              f"Prec@5={query_result['scores']['k=5']['precision']:.2f}")

    # Compute averages
    summary = {}
    print("\n── Summary ──────────────────────────────────────")
    for k in K_VALUES:
        n = len(metrics[k]["recall"])
        avg_recall = sum(metrics[k]["recall"]) / n
        avg_precision = sum(metrics[k]["precision"]) / n
        avg_mrr = sum(metrics[k]["rr"]) / n
        avg_hit_rate = sum(metrics[k]["hit_rate"]) / n

        summary[f"k={k}"] = {
            "avg_recall": round(avg_recall, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_mrr": round(avg_mrr, 4),
            "avg_hit_rate": round(avg_hit_rate, 4),
            "num_queries": n
        }

        print(f"\n  K={k}:")
        print(f"    Recall@{k}    : {avg_recall:.4f}")
        print(f"    Precision@{k} : {avg_precision:.4f}")
        print(f"    MRR@{k}       : {avg_mrr:.4f}")
        print(f"    Hit Rate@{k}  : {avg_hit_rate:.4f}")

    print("\n─────────────────────────────────────────────────\n")

    # Save results
    output = {
        "summary": summary,
        "per_query": per_query_results
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}\n")
    print("=== Retrieval Evaluation Complete ===\n")


if __name__ == "__main__":
    evaluate_retrieval()
