"""
eval_oos.py
-----------
Tests the RAG pipeline with out-of-scope queries — questions whose answers
are NOT present in the knowledge base.

A well-built RAG system should say "I don't have information about this"
instead of hallucinating an answer.

Saves results to: results/oos_results.json

Run:
    python src/eval_oos.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retrieve import retrieve
from generate import generate

# ── Config ────────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "results/oos_results.json"
# ─────────────────────────────────────────────────────────────────────────────

# The exact phrase our prompt instructs the LLM to use when it can't answer
OOS_REFUSAL_PHRASE = "I don't have information about this"


def is_refusal(answer):
    """Returns True if the LLM correctly declined to answer."""
    return OOS_REFUSAL_PHRASE.lower() in answer.lower()


def evaluate_oos():
    print("\n=== Out-of-Scope Query Evaluation ===\n")
    os.makedirs("results", exist_ok=True)

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    oos_queries = [q for q in ground_truth if q.get("out_of_scope", False)]
    print(f"Testing {len(oos_queries)} out-of-scope queries\n")
    print("Expected behavior: LLM should refuse to answer (not hallucinate)\n")

    results = []
    correct_refusals = 0

    for q in oos_queries:
        query_id = q["id"]
        query_text = q["query"]

        print(f"  [{query_id}] {query_text}")

        # Retrieve and generate (same pipeline as normal)
        chunks = retrieve(query_text, k=5)
        gen_result = generate(query_text, chunks)
        answer = gen_result["answer"]

        refused = is_refusal(answer)
        if refused:
            correct_refusals += 1
            status = "✓ CORRECT REFUSAL"
        else:
            status = "✗ HALLUCINATION DETECTED"

        print(f"    Status : {status}")
        print(f"    Answer : {answer[:120]}...\n" if len(answer) > 120 else f"    Answer : {answer}\n")

        results.append({
            "query_id": query_id,
            "query": query_text,
            "generated_answer": answer,
            "correctly_refused": refused,
            "status": status
        })

    refusal_rate = correct_refusals / len(oos_queries) if oos_queries else 0

    summary = {
        "total_oos_queries": len(oos_queries),
        "correct_refusals": correct_refusals,
        "hallucinations": len(oos_queries) - correct_refusals,
        "refusal_rate": round(refusal_rate, 4)
    }

    print("── Summary ──────────────────────────────────────")
    print(f"  Total OOS queries   : {summary['total_oos_queries']}")
    print(f"  Correct refusals    : {summary['correct_refusals']}")
    print(f"  Hallucinations      : {summary['hallucinations']}")
    print(f"  Refusal rate        : {refusal_rate*100:.1f}%")
    print("─────────────────────────────────────────────────\n")

    output = {
        "summary": summary,
        "per_query": results
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}\n")
    print("=== OOS Evaluation Complete ===\n")


if __name__ == "__main__":
    evaluate_oos()
