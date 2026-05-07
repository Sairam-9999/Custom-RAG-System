import re
from .embedder import embed_query


STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "did", "who", "what", "why", "how",
    "when", "where", "does", "do", "from", "after", "before"
}


def tokenize(text):
    return [
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def keyword_score(query, chunk):
    query_terms = tokenize(query)
    chunk_terms = set(tokenize(chunk))

    if not query_terms:
        return 0.0

    matches = sum(1 for term in query_terms if term in chunk_terms)
    return matches / len(query_terms)


def retrieve(query, vector_store, chunks, top_k=5):
    query_embedding = embed_query(query)

    semantic_scores, indices = vector_store.search(
        query_embedding,
        top_k=min(80, len(chunks)),
    )

    semantic_candidates = set()

    for idx in indices:
        if 0 <= idx < len(chunks):
            semantic_candidates.add(idx)

    lexical_candidates = set()

    for idx, chunk in enumerate(chunks):
        if keyword_score(query, chunk) > 0:
            lexical_candidates.add(idx)

    candidate_indices = semantic_candidates.union(lexical_candidates)

    scored = []

    for idx in candidate_indices:
        chunk = chunks[idx]

        lexical = keyword_score(query, chunk)

        semantic = 0.0

        if idx in indices:
            semantic_position = list(indices).index(idx)
            semantic = float(semantic_scores[semantic_position])

        final_score = semantic + (0.7 * lexical)

        scored.append((final_score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [chunk for _, chunk in scored[:top_k]]
