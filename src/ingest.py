"""
ingest.py
---------
Loads documents from data/raw/, chunks them, embeds using sentence-transformers,
and stores in a local ChromaDB vector store.

Run this first before anything else:
    python src/ingest.py
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
CHROMA_DB_PATH = "chroma_store"
COLLECTION_NAME = "docueval"
CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────


def load_documents(folder_path):
    """Load all .txt files from a folder. Returns list of (filename, text)."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            documents.append((filename, text))
            print(f"  Loaded: {filename} ({len(text)} chars)")
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple character-level chunking with overlap.
    Splits on sentence boundaries where possible.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary (period + space)
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period != -1 and last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap  # overlap so context isn't lost

    # Remove empty chunks
    chunks = [c for c in chunks if len(c) > 30]
    return chunks


def build_chunk_id(filename, chunk_index):
    """Creates a unique ID for each chunk."""
    base = filename.replace(".txt", "").replace(" ", "_")
    return f"{base}_chunk_{chunk_index}"


def ingest():
    print("\n=== DocuEval Ingestion Pipeline ===\n")

    # Step 1: Load documents
    print("Step 1: Loading documents...")
    documents = load_documents(RAW_DATA_DIR)
    print(f"  Total documents loaded: {len(documents)}\n")

    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    all_chunks = []       # (chunk_id, chunk_text, source_filename)
    chunk_map = {}        # chunk_id → source filename (saved for eval use)

    for filename, text in documents:
        chunks = chunk_text(text)
        print(f"  {filename}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunk_id = build_chunk_id(filename, i)
            all_chunks.append((chunk_id, chunk, filename))
            chunk_map[chunk_id] = filename

    print(f"  Total chunks created: {len(all_chunks)}\n")

    # Save chunk map so eval scripts can use it
    with open("data/chunk_map.json", "w") as f:
        json.dump(chunk_map, f, indent=2)
    print("  Saved chunk_map.json\n")

    # Step 3: Embed chunks
    print("Step 3: Embedding chunks using sentence-transformers...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [chunk for _, chunk, _ in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}\n")

    # Step 4: Store in ChromaDB
    print("Step 4: Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if it exists (clean re-ingest)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection for clean re-ingest.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [chunk_id for chunk_id, _, _ in all_chunks]
    metadatas = [{"source": fname} for _, _, fname in all_chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )

    print(f"  Stored {len(ids)} chunks in ChromaDB at '{CHROMA_DB_PATH}'\n")
    print("=== Ingestion Complete ===\n")


if __name__ == "__main__":
    ingest()
