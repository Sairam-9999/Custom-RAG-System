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


def is_fact_question(query):
    q = query.lower()

    starters = (
        "who",
        "what",
        "when",
        "where",
        "which",
        "how much",
        "how many",
    )

    return q.startswith(starters)


def build_prompt(query, context):
    if is_fact_question(query):
        return f"""
<s>[INST]
Answer using ONLY the context.

Give a SHORT direct answer.

If answer is missing say:
"I don't know from the provided context."

Question:
{query}

Context:
{context}
[/INST]
"""
    else:
        return f"""
<s>[INST]
Answer using ONLY the context.

Explain briefly in 2-3 sentences.

If answer is missing say:
"I don't know from the provided context."

Question:
{query}

Context:
{context}
[/INST]
"""


def ask_rag(query, store, chunks):
    context_chunks = retrieve(query, store, chunks, top_k=10)

    raw_context = "\n\n".join(context_chunks[:3])

    mistral_prompt = build_prompt(query, raw_context)

    return mistral_prompt, context_chunks
