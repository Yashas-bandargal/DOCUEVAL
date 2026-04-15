"""
generate.py - fixed version
"""
import os
from google import genai

GEMINI_MODEL = "gemini-2.0-flash"
MAX_CONTEXT_CHUNKS = 4

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def build_prompt(query, chunks):
    context_text = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not present in the context, respond with exactly:
"I don't have information about this in the provided documents."

Do not make up information. Do not use outside knowledge.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""
    return prompt

def generate(query, chunks):
    context_chunks = chunks[:MAX_CONTEXT_CHUNKS]
    prompt = build_prompt(query, context_chunks)
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    answer = response.text.strip()
    return {
        "answer": answer,
        "prompt": prompt,
        "chunks_used": len(context_chunks)
    }

if __name__ == "__main__":
    print("generate.py loaded. GEMINI_MODEL =", GEMINI_MODEL)
    print("Model name is correct: gemini-2.0-flash")
