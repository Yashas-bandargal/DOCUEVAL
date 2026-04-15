# DocuEval

DocuEval is a beginner-level RAG project built on company policy documents. The main goal of this project was to learn how retrieval-augmented generation works and how to evaluate it using simple metrics instead of only checking whether the model gives an answer.

## Features

- Document ingestion and chunking
- Embedding-based retrieval using ChromaDB
- Answer generation using Gemini
- Retrieval evaluation with Recall, Precision, MRR, and Hit Rate
- Basic generation evaluation
- Out-of-scope query testing
- A/B testing for chunk settings
- Streamlit UI to explore the workflow
- Simple regression checks with `pytest`

## Project Structure

```text
docueval/
|-- app.py
|-- pages_ui/
|-- data/
|-- src/
|-- results/
|-- chroma_store/
|-- .github/
|-- requirements.txt
|-- requirements_ui.txt
|-- .gitignore
`-- README.md
```

## Tech Stack

- Python
- ChromaDB
- sentence-transformers
- Google Gemini
- Streamlit
- pytest
- GitHub Actions

## Setup

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install UI dependencies:

```bash
pip install -r requirements_ui.txt
```

Set the Gemini API key:

```powershell
$env:GOOGLE_API_KEY="your_key_here"
```

## How To Run

### 1. Ingest the documents

```bash
python src/ingest.py
```

### 2. Test retrieval

```bash
python src/retrieve.py
```

### 3. Test generation

```bash
python src/generate.py
```

### 4. Run retrieval evaluation

```bash
python src/eval_retrieval.py
```

### 5. Run generation evaluation

```bash
python src/eval_generation.py
```

### 6. Run out-of-scope testing

```bash
python src/eval_oos.py
```

### 7. Run A/B test

```bash
python src/ab_test.py
```

### 8. Run regression tests

```bash
python -m pytest src/test_regression.py -v
```

### 9. Generate final report

```bash
python src/generate_report.py
```

### 10. Launch the Streamlit app

```bash
streamlit run app.py
```

## Current Results

Some retrieval and A/B test results are already included in the repository.

| Metric | Score |
|---|---|
| Recall@5 | 1.00 |
| Precision@5 | 0.88 |
| MRR@5 | 1.00 |
| Hit Rate@5 | 1.00 |

## What I Learned

- How to build a basic RAG pipeline
- How chunking affects retrieval quality
- How to store embeddings in ChromaDB
- How to test a project with simple evaluation scripts
- How to present results through a small Streamlit dashboard

## Future Improvements

- Add more documents for testing
- Improve answer quality evaluation
- Add better handling for out-of-scope questions
- Improve the UI further
- Compare more chunking strategies
