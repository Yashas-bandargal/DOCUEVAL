"""
generate_report.py
------------------
Reads all evaluation result files and prints a clean, formatted summary report.
Also saves a combined report to results/final_report.json.

Run AFTER all eval scripts have been executed:
    python src/generate_report.py
"""

import json
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
RETRIEVAL_RESULTS = "results/retrieval_scores.json"
GENERATION_RESULTS = "results/generation_scores.json"
OOS_RESULTS = "results/oos_results.json"
FINAL_REPORT = "results/final_report.json"
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds for pass/fail (you can tune these)
THRESHOLDS = {
    "recall_at_5": 0.70,
    "mrr_at_5": 0.60,
    "hit_rate_at_5": 0.75,
    "faithfulness": 0.80,
    "answer_relevance": 3.5,
    "rouge_l": 0.25,
    "refusal_rate": 0.60
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def pass_fail(value, threshold, higher_is_better=True):
    if higher_is_better:
        return "PASS ✓" if value >= threshold else "FAIL ✗"
    else:
        return "PASS ✓" if value <= threshold else "FAIL ✗"


def print_report():
    print("\n" + "="*60)
    print("       DocuEval — Final Evaluation Report")
    print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    retrieval = load_json(RETRIEVAL_RESULTS)
    generation = load_json(GENERATION_RESULTS)
    oos = load_json(OOS_RESULTS)

    combined = {"generated_at": datetime.now().isoformat()}

    # ── Retrieval Section ──────────────────────────────────────
    print("\n📊 RETRIEVAL METRICS")
    print("-" * 40)
    if retrieval:
        r5 = retrieval["summary"].get("k=5", {})
        r3 = retrieval["summary"].get("k=3", {})

        recall_5 = r5.get("avg_recall", 0)
        prec_5 = r5.get("avg_precision", 0)
        mrr_5 = r5.get("avg_mrr", 0)
        hr_5 = r5.get("avg_hit_rate", 0)

        print(f"  Recall@3       : {r3.get('avg_recall', 0):.4f}")
        print(f"  Recall@5       : {recall_5:.4f}   [{pass_fail(recall_5, THRESHOLDS['recall_at_5'])}]")
        print(f"  Precision@3    : {r3.get('avg_precision', 0):.4f}")
        print(f"  Precision@5    : {prec_5:.4f}")
        print(f"  MRR@5          : {mrr_5:.4f}   [{pass_fail(mrr_5, THRESHOLDS['mrr_at_5'])}]")
        print(f"  Hit Rate@5     : {hr_5:.4f}   [{pass_fail(hr_5, THRESHOLDS['hit_rate_at_5'])}]")
        print(f"  Queries tested : {r5.get('num_queries', 0)}")

        combined["retrieval"] = retrieval["summary"]
    else:
        print("  [No retrieval results found. Run eval_retrieval.py first.]")

    # ── Generation Section ─────────────────────────────────────
    print("\n🤖 GENERATION METRICS")
    print("-" * 40)
    if generation:
        g = generation["summary"]
        faith = g.get("avg_faithfulness", 0)
        relev = g.get("avg_answer_relevance", 0)
        rouge = g.get("avg_rouge_l", 0)

        print(f"  Faithfulness        : {faith:.4f}   [{pass_fail(faith, THRESHOLDS['faithfulness'])}]")
        print(f"  Answer Relevance    : {relev:.4f}/5 [{pass_fail(relev, THRESHOLDS['answer_relevance'])}]")
        print(f"  ROUGE-L             : {rouge:.4f}   [{pass_fail(rouge, THRESHOLDS['rouge_l'])}]")
        print(f"  Faithfulness Pass % : {faith*100:.1f}%")
        print(f"  Queries tested      : {g.get('num_queries', 0)}")

        combined["generation"] = generation["summary"]
    else:
        print("  [No generation results found. Run eval_generation.py first.]")

    # ── OOS Section ────────────────────────────────────────────
    print("\n🚫 OUT-OF-SCOPE (HALLUCINATION) TEST")
    print("-" * 40)
    if oos:
        o = oos["summary"]
        refusal_rate = o.get("refusal_rate", 0)
        print(f"  Total OOS queries : {o.get('total_oos_queries', 0)}")
        print(f"  Correct refusals  : {o.get('correct_refusals', 0)}")
        print(f"  Hallucinations    : {o.get('hallucinations', 0)}")
        print(f"  Refusal rate      : {refusal_rate*100:.1f}%   [{pass_fail(refusal_rate, THRESHOLDS['refusal_rate'])}]")

        combined["oos"] = oos["summary"]
    else:
        print("  [No OOS results found. Run eval_oos.py first.]")

    # ── Overall Summary ────────────────────────────────────────
    print("\n🏁 OVERALL ASSESSMENT")
    print("-" * 40)
    checks = []
    if retrieval:
        checks.append(r5.get("avg_recall", 0) >= THRESHOLDS["recall_at_5"])
        checks.append(r5.get("avg_mrr", 0) >= THRESHOLDS["mrr_at_5"])
        checks.append(r5.get("avg_hit_rate", 0) >= THRESHOLDS["hit_rate_at_5"])
    if generation:
        checks.append(g.get("avg_faithfulness", 0) >= THRESHOLDS["faithfulness"])
        checks.append(g.get("avg_answer_relevance", 0) >= THRESHOLDS["answer_relevance"])
    if oos:
        checks.append(o.get("refusal_rate", 0) >= THRESHOLDS["refusal_rate"])

    passed = sum(checks)
    total = len(checks)
    print(f"  Metrics passed    : {passed}/{total}")
    print(f"  Overall status    : {'SYSTEM READY ✓' if passed >= total * 0.7 else 'NEEDS IMPROVEMENT ✗'}")

    print("\n" + "="*60)
    print("  Full per-query results available in results/*.json")
    print("="*60 + "\n")

    # Save combined report
    combined["thresholds_used"] = THRESHOLDS
    combined["overall"] = {"passed": passed, "total": total}
    with open(FINAL_REPORT, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Final report saved to {FINAL_REPORT}\n")


if __name__ == "__main__":
    print_report()
