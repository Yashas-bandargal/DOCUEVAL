# DocuEval - RAG Pipeline with Evaluation Suite

A domain-specific question answering system built on company policy documents, with an evaluation layer that measures both retrieval quality and answer quality.

> This project is focused on RAG evaluation, not just answer generation. It is designed to show how to build, test, compare, benchmark, and regression-check a retrieval-augmented QA pipeline.

## What This Project Does

- Builds a RAG pipeline over HR and company policy documents
- Evaluates retrieval quality using Recall@K, Precision@K, MRR, and Hit Rate
- Evaluates generation quality using Faithfulness, Answer Relevance, and ROUGE-L
- Tests hallucination behavior on out-of-scope queries
- Compares two pipeline configurations through A/B testing
- Includes a pytest regression suite to catch quality degradation
- Includes a Streamlit dashboard for exploring the pipeline and evaluation outputs
- Includes a GitHub Actions workflow for repeatable evaluation runs

## Project Structure

```text
docueval/
|-- app.py
|-- pages_ui/
|   |-- pg_overview.py
|   |-- pg_query.py
|   |-- pg_retrieval.py
|   |-- pg_generation.py
|   |-- pg_oos.py
|   |-- pg_abtest.py
|   `-- pg_regression.py
|-- data/
|   |-- raw/
|   |   |-- ai_policy.txt
|   |   |-- remote_work_policy.txt
|   |   |-- leave_policy.txt
|   |   `-- data_security_policy.txt
|   |-- ground_truth.json
|   `-- chunk_map.json
|-- src/
|   |-- ingest.py
|   |-- retrieve.py
|   |-- generate.py
|   |-- eval_retrieval.py
|   |-- eval_generation.py
|   |-- eval_oos.py
|   |-- ab_test.py
|   |-- generate_report.py
|   `-- test_regression.py
|-- results/
|   |-- retrieval_scores.json
|   |-- ab_test_results.json
|   `-- final_report.json
|-- chroma_store/
|-- .github/
|   `-- workflows/
|       `-- eval.yml
|-- requirements.txt
|-- requirements_ui.txt
|-- LICENSE
|-- .gitignore
`-- README.md
```

## Setup

### Backend evaluation environment

```bash
pip install -r requirements.txt
```

### Streamlit UI environment

```bash
pip install -r requirements_ui.txt
```

### Set your Gemini API key

Linux/macOS:

```bash
export GOOGLE_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your_key_here"
```

You can create a Gemini API key at: https://aistudio.google.com/

## Running the Project

Run the scripts from the project root in this order:

### 1. Ingest documents into the vector store

```bash
python src/ingest.py
```

Loads documents from `data/raw/`, chunks them, embeds them using `all-MiniLM-L6-v2`, and stores them in ChromaDB.

### 2. Retrieval sanity check

```bash
python src/retrieve.py
```

### 3. Generation sanity check

```bash
python src/generate.py
```

### 4. Evaluate retrieval quality

```bash
python src/eval_retrieval.py
```

Computes Recall@3, Recall@5, Precision@3, Precision@5, MRR@5, and Hit Rate@5 across the in-scope golden queries.

### 5. Evaluate generation quality

```bash
python src/eval_generation.py
```

Uses Gemini as an LLM judge to score faithfulness and answer relevance, and computes ROUGE-L.

### 6. Test out-of-scope behavior

```bash
python src/eval_oos.py
```

Checks whether the system abstains from unsupported queries instead of hallucinating.

### 7. Run the A/B test

```bash
python src/ab_test.py
```

Compares chunking configurations and reports which setup performs better.

### 8. Run regression tests

```bash
python -m pytest src/test_regression.py -v
```

### 9. Generate the final report

```bash
python src/generate_report.py
```

### 10. Launch the UI

```bash
streamlit run app.py
```

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| LLM | Google Gemini 2.0 Flash |
| Evaluation | Custom metrics + LLM-as-judge scoring |
| UI | Streamlit |
| Text overlap scoring | `rouge-score` |
| Testing | `pytest` |
| CI/CD | GitHub Actions |

## Evaluation Metrics

### Retrieval metrics

| Metric | What it measures |
|---|---|
| Recall@K | Whether the relevant source appears in the top K results |
| Precision@K | How many of the returned results are actually relevant |
| MRR | How high the first relevant result appears in the ranking |
| Hit Rate@K | Whether at least one relevant result appears in the top K |

### Generation metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Whether the answer is grounded in retrieved context |
| Answer Relevance | Whether the answer directly addresses the question |
| ROUGE-L | Overlap between generated and reference answers |

### Out-of-scope testing

Measures whether the assistant refuses unsupported questions instead of confidently fabricating answers.

## Design Decisions and Tradeoffs

**Why sentence-transformers instead of an API embedding model?**  
It runs locally, avoids extra API cost during ingestion, and is fast enough for a compact document corpus.

**Why ChromaDB instead of FAISS?**  
ChromaDB provides built-in persistence and a simpler local workflow for this project.

**Why use an LLM judge for faithfulness?**  
A judge model captures paraphrased but grounded answers better than simple exact-match heuristics.

**Why keep ROUGE-L as well?**  
ROUGE-L is deterministic and useful for regression checks, while LLM scoring captures semantic quality.

## Current Results

The repository currently includes retrieval and A/B test outputs generated on April 8, 2026.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Recall@5 | 1.00 | 0.70 | Pass |
| Precision@5 | 0.88 | n/a | Reported |
| MRR@5 | 1.00 | 0.60 | Pass |
| Hit Rate@5 | 1.00 | 0.75 | Pass |
| Faithfulness | Not committed | 0.80 | Pending |
| Answer Relevance | Not committed | 3.5/5 | Pending |
| ROUGE-L | Not committed | 0.25 | Pending |
| OOS Refusal Rate | Not committed | 0.60 | Pending |

### A/B test summary

- Config A: Small chunks (`chunk_size=300`, `k=3`) achieved `avg_recall=1.00`, `avg_mrr=1.00`, `avg_hit_rate=1.00`
- Config B: Large chunks (`chunk_size=500`, `k=5`) achieved `avg_recall=1.00`, `avg_mrr=1.00`, `avg_hit_rate=1.00`

Both tested configurations performed perfectly on the current 20-query retrieval benchmark, so the next useful differentiator would be a larger or more adversarial evaluation set.

## What I Would Improve Next

- Add RAGAS or a second evaluation framework for comparison
- Track evaluation history over time with an experiment log
- Add adversarial and paraphrased query testing
- Expand the document corpus beyond the current policy set
- Measure latency across ingestion, retrieval, and generation stages

## Resume-Friendly Summary

**DocuEval - RAG Pipeline with Evaluation Suite**  
Python, ChromaDB, Gemini API, sentence-transformers, Streamlit, pytest, GitHub Actions

- Built a modular RAG pipeline covering ingestion, chunking, embedding, retrieval, generation, evaluation, and dashboarding
- Created a golden dataset with in-scope and out-of-scope queries for systematic benchmarking
- Implemented retrieval evaluation with Recall@K, Precision@K, MRR, and Hit Rate
- Added generation evaluation with LLM-as-judge scoring and ROUGE-L
- Built regression tests to catch quality drops before code changes are merged
- Added a Streamlit dashboard and GitHub Actions automation for repeatable project checks
