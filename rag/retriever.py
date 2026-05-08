import re
import math
from collections import Counter
from .embedder import embed_query


STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "did", "who", "what", "why", "how",
    "when", "where", "does", "do", "from", "after", "before", "about"
}


def tokenize(text):
    return [
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def bm25_scores(query, chunks, k1=1.5, b=0.75):
    query_terms = tokenize(query)
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    if not query_terms:
        return [0.0] * len(chunks)

    doc_freq = Counter()
    for terms in tokenized_chunks:
        for term in set(terms):
            doc_freq[term] += 1

    avg_len = sum(len(t) for t in tokenized_chunks) / max(len(tokenized_chunks), 1)
    scores = []

    for terms in tokenized_chunks:
        term_counts = Counter(terms)
        doc_len = len(terms)
        score = 0.0

        for term in query_terms:
            if term not in term_counts:
                continue

            df = doc_freq.get(term, 0)
            idf = math.log((len(chunks) - df + 0.5) / (df + 0.5) + 1)
            tf = term_counts[term]

            denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            score += idf * ((tf * (k1 + 1)) / denom)

        scores.append(score)

    return scores


def normalize(values):
    if not values:
        return []

    min_v = min(values)
    max_v = max(values)

    if max_v == min_v:
        return [0.0 for _ in values]

    return [(v - min_v) / (max_v - min_v) for v in values]


def retrieve(query, vector_store, chunks, top_k=5):
    query_embedding = embed_query(query)

    semantic_scores, semantic_indices = vector_store.search(
        query_embedding,
        top_k=min(120, len(chunks)),
    )

    bm25_raw = bm25_scores(query, chunks)
    bm25_norm = normalize(bm25_raw)

    semantic_map = {}
    semantic_norm = normalize([float(s) for s in semantic_scores])

    for idx, score in zip(semantic_indices, semantic_norm):
        if 0 <= idx < len(chunks):
            semantic_map[idx] = score

    candidate_indices = set(semantic_map.keys())

    # Add strongest BM25 candidates globally, even if FAISS missed them.
    bm25_ranked = sorted(
        range(len(chunks)),
        key=lambda i: bm25_norm[i],
        reverse=True,
    )

    candidate_indices.update(bm25_ranked[: min(80, len(chunks))])

    scored = []

    for idx in candidate_indices:
        semantic = semantic_map.get(idx, 0.0)
        lexical = bm25_norm[idx]

        # Generic hybrid score. Lexical is slightly stronger for factual QA.
        final_score = (0.45 * semantic) + (0.55 * lexical)

        scored.append((final_score, idx, chunks[idx]))

    scored.sort(reverse=True, key=lambda x: x[0])

    results = []
    seen = set()

    for _, idx, chunk in scored:
        if chunk not in seen:
            results.append(chunk)
            seen.add(chunk)

        if len(results) >= top_k:
            break

    return results
