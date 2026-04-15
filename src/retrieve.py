"""
retrieve.py
-----------
Retrieves top-K relevant chunks from ChromaDB for a given query.
Uses sentence-transformers to embed the query, then cosine similarity search.

Can be used standalone:
    python src/retrieve.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "chroma_store"
COLLECTION_NAME = "docueval"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 5
# ─────────────────────────────────────────────────────────────────────────────

# Load model and collection once (module-level, so other scripts can import)
_model = None
_collection = None


def _load_resources():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)


def retrieve(query, k=DEFAULT_K):
    """
    Retrieve top-K chunks for a given query.

    Returns a list of dicts:
    [
        {
            "chunk_id": "leave_policy_chunk_0",
            "text": "...",
            "source": "leave_policy.txt",
            "distance": 0.12   # lower = more similar (cosine distance)
        },
        ...
    ]
    """
    _load_resources()

    query_embedding = _model.encode([query]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    return chunks


if __name__ == "__main__":
    # Quick test
    test_query = "How many sick leave days do employees get?"
    print(f"\nQuery: {test_query}\n")
    results = retrieve(test_query, k=3)
    for i, r in enumerate(results):
        print(f"Rank {i+1} | {r['chunk_id']} | distance: {r['distance']:.4f}")
        print(f"  Source: {r['source']}")
        print(f"  Text: {r['text'][:120]}...\n")
