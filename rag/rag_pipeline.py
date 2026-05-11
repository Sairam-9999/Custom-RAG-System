import time

from .chunker import chunk_text
from .embedder import get_embeddings
from .vector_store import VectorStore
from .retriever import retrieve
from .retrieval_types import CompressedContext


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


def ask_rag(
    query,
    store,
    chunks,
    reranker=None,
    context_selector=None,
    retrieval_top_k=10,
    rerank_top_n=5,
    context_top_n=3,
):
    timings = {}

    t0 = time.perf_counter()
    retrieved_results = retrieve(query, store, chunks, top_k=retrieval_top_k)
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    if reranker is not None:
        t0 = time.perf_counter()
        results = reranker.rerank(query=query, results=retrieved_results, top_n=rerank_top_n)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    else:
        results = retrieved_results
        timings["rerank_ms"] = 0.0

    if context_selector is not None:
        t0 = time.perf_counter()
        compressed = context_selector.select(query, results)
        timings["compression_ms"] = (time.perf_counter() - t0) * 1000
        mistral_prompt = build_prompt(query, compressed.context)
    else:
        compressed = None
        timings["compression_ms"] = 0.0
        raw_context = "\n\n".join(r.text for r in results[:context_top_n])
        mistral_prompt = build_prompt(query, raw_context)

    return mistral_prompt, results, compressed, timings
