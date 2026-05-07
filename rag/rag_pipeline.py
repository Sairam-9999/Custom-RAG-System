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
    context_chunks = retrieve(query, store, chunks, top_k=4)

    raw_context = "\n\n".join(context_chunks)

    mistral_prompt = f"""
<s>[INST]
You are a grounded RAG assistant.

Answer ONLY using the provided context.

Question:
{query}

Context:
{raw_context}

Rules:
- Give short factual answers when possible.
- For reasoning questions, explain in 2-3 sentences.
- Do not invent facts.
- If answer is missing, say:
"I don't know from the provided context."
[/INST]
"""

    return mistral_prompt, context_chunks
