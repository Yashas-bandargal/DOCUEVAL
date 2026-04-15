"""
eval_generation.py - fixed version
Evaluates generation quality: Faithfulness, Answer Relevance, ROUGE-L
"""

import json
import os
import sys
import time

from google import genai
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(__file__))
from retrieve import retrieve
from generate import generate

# ── Config ────────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "results/generation_scores.json"
GEMINI_MODEL = "gemini-2.0-flash"
API_SLEEP = 11
# ─────────────────────────────────────────────────────────────────────────────

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def call_with_retry(prompt):
    """Call Gemini with automatic retry on rate limit."""
    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise e
    return None


def score_faithfulness(answer, context_chunks):
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"""You are an expert evaluator checking if an AI answer is grounded in context.

CONTEXT:
{context_text}

AI ANSWER:
{answer}

Is every factual claim in the answer supported by the context?
Reply with ONLY 1 (yes, fully grounded) or 0 (no, contains hallucination). No explanation."""

    result = call_with_retry(prompt)
    return 1 if result and result.strip() == "1" else 0


def score_answer_relevance(query, answer):
    prompt = f"""You are an expert evaluator scoring answer relevance.

QUESTION: {query}

AI ANSWER: {answer}

Score relevance 1-5:
5 = Completely answers the question
4 = Mostly answers with minor gaps
3 = Partially answers
2 = Barely addresses
1 = Does not address at all

Reply with ONLY a single digit 1-5. No explanation."""

    result = call_with_retry(prompt)
    try:
        score = int(result.strip())
        return score if 1 <= score <= 5 else 1
    except (ValueError, TypeError, AttributeError):
        return 1


def score_rouge_l(generated_answer, expected_answer):
    scores = scorer.score(expected_answer, generated_answer)
    return round(scores["rougeL"].fmeasure, 4)


def evaluate_generation():
    print("\n=== Generation Evaluation ===\n")
    os.makedirs("results", exist_ok=True)

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    in_scope_queries = [q for q in ground_truth if not q.get("out_of_scope", False)]
    print(f"Evaluating {len(in_scope_queries)} in-scope queries")
    print(f"Using model : {GEMINI_MODEL}")
    print(f"Sleep between queries: {API_SLEEP}s")
    print(f"Estimated time: ~{len(in_scope_queries) * API_SLEEP * 2 // 60} minutes\n")

    per_query_results = []
    faithfulness_scores = []
    relevance_scores = []
    rouge_scores_list = []

    for idx, q in enumerate(in_scope_queries):
        query_id = q["id"]
        query_text = q["query"]
        expected_answer = q["expected_answer"]

        print(f"  [{idx+1}/{len(in_scope_queries)}] Processing [{query_id}]...")

        # Retrieve
        chunks = retrieve(query_text, k=5)

        # Generate answer
        time.sleep(API_SLEEP)
        gen_result = generate(query_text, chunks)
        generated_answer = gen_result["answer"]

        # Score faithfulness
        time.sleep(API_SLEEP)
        faithfulness = score_faithfulness(generated_answer, chunks[:4])

        # Score answer relevance
        time.sleep(API_SLEEP)
        relevance = score_answer_relevance(query_text, generated_answer)

        # ROUGE-L (local, no API)
        rouge_l = score_rouge_l(generated_answer, expected_answer)

        faithfulness_scores.append(faithfulness)
        relevance_scores.append(relevance)
        rouge_scores_list.append(rouge_l)

        per_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "scores": {
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "rouge_l": rouge_l
            }
        })

        print(f"    Faithfulness={faithfulness} | Relevance={relevance}/5 | ROUGE-L={rouge_l:.4f}")

    n = len(faithfulness_scores)
    summary = {
        "avg_faithfulness": round(sum(faithfulness_scores) / n, 4),
        "avg_answer_relevance": round(sum(relevance_scores) / n, 4),
        "avg_rouge_l": round(sum(rouge_scores_list) / n, 4),
        "faithfulness_pass_rate": round(sum(faithfulness_scores) / n, 4),
        "num_queries": n
    }

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Avg Faithfulness     : {summary['avg_faithfulness']}")
    print(f"  Avg Answer Relevance : {summary['avg_answer_relevance']} / 5")
    print(f"  Avg ROUGE-L          : {summary['avg_rouge_l']}")
    print(f"  Faithfulness Pass %  : {summary['faithfulness_pass_rate']*100:.1f}%")
    print("─────────────────────────────────────────────────\n")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "per_query": per_query_results}, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}\n")
    print("=== Generation Evaluation Complete ===\n")


if __name__ == "__main__":
    evaluate_generation()
