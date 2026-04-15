"""
test_regression.py
------------------
Regression test suite using pytest.
Asserts that evaluation scores stay above defined thresholds.

If you change chunking, embedding, or prompts — run this to verify
that quality has NOT degraded. This is a Phase 6 concept.

Run:
    pytest src/test_regression.py -v

Requirements:
    pip install pytest

NOTE: Retrieval and generation eval must have been run first.
      This test reads from results/*.json, it does NOT re-run the pipeline.
"""

import json
import os
import pytest

# ── Thresholds — tune these based on your actual scores ──────────────────────
MIN_RECALL_AT_5 = 0.65
MIN_MRR_AT_5 = 0.55
MIN_HIT_RATE_AT_5 = 0.70
MIN_FAITHFULNESS = 0.75
MIN_ANSWER_RELEVANCE = 3.0
MIN_ROUGE_L = 0.20
MIN_REFUSAL_RATE = 0.60
# ─────────────────────────────────────────────────────────────────────────────


def load_retrieval_results():
    path = "results/retrieval_scores.json"
    assert os.path.exists(path), f"Run eval_retrieval.py first. File not found: {path}"
    with open(path) as f:
        return json.load(f)


def load_generation_results():
    path = "results/generation_scores.json"
    assert os.path.exists(path), f"Run eval_generation.py first. File not found: {path}"
    with open(path) as f:
        return json.load(f)


def load_oos_results():
    path = "results/oos_results.json"
    assert os.path.exists(path), f"Run eval_oos.py first. File not found: {path}"
    with open(path) as f:
        return json.load(f)


# ── Retrieval Tests ───────────────────────────────────────────────────────────

class TestRetrieval:

    def test_recall_at_5_above_threshold(self):
        """Recall@5 should be above minimum threshold."""
        data = load_retrieval_results()
        score = data["summary"]["k=5"]["avg_recall"]
        assert score >= MIN_RECALL_AT_5, (
            f"Recall@5 dropped to {score:.4f}, below threshold {MIN_RECALL_AT_5}. "
            f"Check your chunking or embedding configuration."
        )

    def test_mrr_at_5_above_threshold(self):
        """MRR@5 should be above minimum threshold."""
        data = load_retrieval_results()
        score = data["summary"]["k=5"]["avg_mrr"]
        assert score >= MIN_MRR_AT_5, (
            f"MRR@5 dropped to {score:.4f}, below threshold {MIN_MRR_AT_5}."
        )

    def test_hit_rate_at_5_above_threshold(self):
        """Hit Rate@5 should be above minimum threshold."""
        data = load_retrieval_results()
        score = data["summary"]["k=5"]["avg_hit_rate"]
        assert score >= MIN_HIT_RATE_AT_5, (
            f"Hit Rate@5 dropped to {score:.4f}, below threshold {MIN_HIT_RATE_AT_5}."
        )

    def test_no_zero_recall_queries(self):
        """No single query should have 0 recall at K=5 more than 30% of the time."""
        data = load_retrieval_results()
        per_query = data["per_query"]
        zero_recall = [q for q in per_query if q["scores"]["k=5"]["recall"] == 0]
        ratio = len(zero_recall) / len(per_query)
        assert ratio <= 0.30, (
            f"{len(zero_recall)} queries had 0 recall at K=5 ({ratio*100:.1f}%). "
            f"Queries: {[q['query_id'] for q in zero_recall]}"
        )


# ── Generation Tests ──────────────────────────────────────────────────────────

class TestGeneration:

    def test_faithfulness_above_threshold(self):
        """Avg faithfulness score should be above threshold (no widespread hallucination)."""
        data = load_generation_results()
        score = data["summary"]["avg_faithfulness"]
        assert score >= MIN_FAITHFULNESS, (
            f"Faithfulness dropped to {score:.4f}, below {MIN_FAITHFULNESS}. "
            f"LLM may be hallucinating. Check your prompt template."
        )

    def test_answer_relevance_above_threshold(self):
        """Avg answer relevance should be above threshold."""
        data = load_generation_results()
        score = data["summary"]["avg_answer_relevance"]
        assert score >= MIN_ANSWER_RELEVANCE, (
            f"Answer relevance dropped to {score:.4f}/5, below {MIN_ANSWER_RELEVANCE}."
        )

    def test_rouge_l_above_threshold(self):
        """Avg ROUGE-L should be above minimum threshold."""
        data = load_generation_results()
        score = data["summary"]["avg_rouge_l"]
        assert score >= MIN_ROUGE_L, (
            f"ROUGE-L dropped to {score:.4f}, below {MIN_ROUGE_L}."
        )

    def test_no_fully_unfaithful_answers(self):
        """Less than 20% of answers should be flagged as unfaithful."""
        data = load_generation_results()
        per_query = data["per_query"]
        unfaithful = [q for q in per_query if q["scores"]["faithfulness"] == 0]
        ratio = len(unfaithful) / len(per_query)
        assert ratio <= 0.20, (
            f"{len(unfaithful)} answers were unfaithful ({ratio*100:.1f}%). "
            f"Queries: {[q['query_id'] for q in unfaithful]}"
        )


# ── OOS Tests ─────────────────────────────────────────────────────────────────

class TestOutOfScope:

    def test_refusal_rate_above_threshold(self):
        """System should refuse to answer out-of-scope queries at a high rate."""
        data = load_oos_results()
        rate = data["summary"]["refusal_rate"]
        assert rate >= MIN_REFUSAL_RATE, (
            f"Refusal rate is {rate*100:.1f}%, below {MIN_REFUSAL_RATE*100:.1f}%. "
            f"System is hallucinating on out-of-scope queries. Check your prompt."
        )

    def test_oos_queries_were_tested(self):
        """At least 5 OOS queries should have been tested."""
        data = load_oos_results()
        total = data["summary"]["total_oos_queries"]
        assert total >= 5, f"Only {total} OOS queries were tested. Add more."
