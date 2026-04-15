"""
ab_test.py
----------
A/B test comparing two pipeline configurations:
  Config A: chunk_size=300, k=3
  Config B: chunk_size=500, k=5

For each config, re-ingests documents, runs retrieval eval, and compares scores.
This is a Phase 6 concept — comparing pipeline versions objectively.

Saves results to: results/ab_test_results.json

Run:
    python src/ab_test.py

NOTE: This re-ingests the vector store for each config. Takes a few minutes.
"""

import json
import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "results/ab_test_results.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "chroma_store"
COLLECTION_NAME = "docueval_ab"

CONFIGS = {
    "config_A": {"chunk_size": 300, "overlap": 50, "k": 3, "label": "Small chunks (300), K=3"},
    "config_B": {"chunk_size": 500, "overlap": 80, "k": 5, "label": "Large chunks (500), K=5"},
}
# ─────────────────────────────────────────────────────────────────────────────


def load_documents():
    docs = []
    for fname in os.listdir(RAW_DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(RAW_DATA_DIR, fname), "r") as f:
                docs.append((fname, f.read().strip()))
    return docs


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period != -1 and last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 30]


def build_temp_collection(chunk_size, overlap, collection_suffix):
    """Ingest with given chunk size into a temporary collection."""
    model = SentenceTransformer(EMBED_MODEL)
    documents = load_documents()

    all_chunks = []
    for filename, text in documents:
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename.replace('.txt','')}_c{i}"
            all_chunks.append((chunk_id, chunk, filename))

    texts = [c for _, c, _ in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll_name = f"{COLLECTION_NAME}_{collection_suffix}"

    existing = [c.name for c in client.list_collections()]
    if coll_name in existing:
        client.delete_collection(coll_name)

    collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c[0] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c[2]} for c in all_chunks]
    )
    return collection, model


def retrieve_from_collection(collection, model, query, k):
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=k,
                                include=["documents", "metadatas", "distances"])
    return [
        {"chunk_id": results["ids"][0][i],
         "source": results["metadatas"][0][i]["source"]}
        for i in range(len(results["ids"][0]))
    ]


def eval_retrieval_for_config(collection, model, ground_truth, k):
    in_scope = [q for q in ground_truth if not q.get("out_of_scope", False)]
    recalls, mrrs, hit_rates = [], [], []

    for q in in_scope:
        retrieved = retrieve_from_collection(collection, model, q["query"], k)
        expected = q["relevant_source"]

        # Recall@K
        recall = 1 if any(c["source"] == expected for c in retrieved[:k]) else 0
        recalls.append(recall)

        # MRR
        rr = 0.0
        for rank, c in enumerate(retrieved, 1):
            if c["source"] == expected:
                rr = 1 / rank
                break
        mrrs.append(rr)

        # Hit Rate
        hit_rates.append(recall)

    n = len(recalls)
    return {
        "avg_recall": round(sum(recalls) / n, 4),
        "avg_mrr": round(sum(mrrs) / n, 4),
        "avg_hit_rate": round(sum(hit_rates) / n, 4),
        "num_queries": n
    }


def run_ab_test():
    print("\n=== A/B Test: Chunk Size Comparison ===\n")
    os.makedirs("results", exist_ok=True)

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    results = {}

    for config_name, config in CONFIGS.items():
        print(f"Running {config_name}: {config['label']}...")
        collection, model = build_temp_collection(
            config["chunk_size"], config["overlap"], config_name
        )
        scores = eval_retrieval_for_config(collection, model, ground_truth, config["k"])
        results[config_name] = {
            "label": config["label"],
            "chunk_size": config["chunk_size"],
            "k": config["k"],
            "scores": scores
        }
        print(f"  Recall@{config['k']}={scores['avg_recall']} | MRR={scores['avg_mrr']} | Hit Rate={scores['avg_hit_rate']}\n")

    # Compare
    print("── Comparison ───────────────────────────────────")
    headers = ["Metric", "Config A (chunk=300, k=3)", "Config B (chunk=500, k=5)", "Winner"]
    metrics_to_compare = ["avg_recall", "avg_mrr", "avg_hit_rate"]

    for metric in metrics_to_compare:
        a_val = results["config_A"]["scores"][metric]
        b_val = results["config_B"]["scores"][metric]
        winner = "Config A" if a_val >= b_val else "Config B"
        print(f"  {metric:<20} A={a_val:.4f}   B={b_val:.4f}   → {winner}")

    print("─────────────────────────────────────────────────\n")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"A/B test results saved to {RESULTS_PATH}\n")
    print("=== A/B Test Complete ===\n")


if __name__ == "__main__":
    run_ab_test()
