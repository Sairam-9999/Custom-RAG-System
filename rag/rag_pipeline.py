from .chunker import chunk_text
from .embedder import get_embeddings
from .vector_store import VectorStore
from .retriever import retrieve


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def build_rag(file_path):
    text = load_file(file_path)

    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    store = VectorStore(embeddings)

    return store, chunks


def ask_rag(query, store, chunks):
    context_chunks = retrieve(query, store, chunks, top_k=3)

    raw_context = "\n\n".join(context_chunks[:3])

    evidence_prompt = f"""
Convert the story evidence into simple notes.

Rules:
- Do not copy phrases from the story.
- Use modern English.
- Explain only why Gisburn stopped painting.
- Output exactly 3 short bullet points.

Context:
{raw_context}

Simple notes:
"""

    mistral_prompt = f"""
<s>[INST]
You are a helpful assistant.

Answer the question using the context.

Question:
{query}

Context:
{raw_context}

Rules:
- Explain in your own words
- Do not copy sentences
- Focus only on the reason
- Answer in 3 clear sentences
[/INST]
"""

    return evidence_prompt, mistral_prompt, context_chunks
