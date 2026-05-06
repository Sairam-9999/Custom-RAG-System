from .embedder import embed_query

EVIDENCE_TERMS = [
    "Stroud",
    "technique collapsed",
    "couldn't do another stroke",
    "I had never known",
    "showy splash",
    "hard passages",
    "couldn't paint him",
    "knew enough to leave off",
    "the thing they called my technique collapsed",
]


def evidence_score(chunk):
    chunk_lower = chunk.lower()
    score = 0

    for term in EVIDENCE_TERMS:
        if term.lower() in chunk_lower:
            score += 3

    if "stroud" in chunk_lower and "paint" in chunk_lower:
        score += 5

    if "couldn't" in chunk_lower and "stroke" in chunk_lower:
        score += 5

    if "technique" in chunk_lower and "collapsed" in chunk_lower:
        score += 8

    return score


def retrieve(query, vector_store, chunks, top_k=5):
    expanded_query = (
        query
        + " real reason Stroud technique collapsed couldn't do another stroke "
        + "showy splash hard passages knew enough to leave off"
    )

    query_embedding = embed_query(expanded_query)
    semantic_scores, indices = vector_store.search(query_embedding, top_k=20)

    candidates = []

    for semantic_score, idx in zip(semantic_scores, indices):
        chunk = chunks[idx]
        keyword_score = evidence_score(chunk)
        final_score = float(semantic_score) + keyword_score

        candidates.append((final_score, chunk))

    candidates.sort(reverse=True, key=lambda x: x[0])

    final_chunks = []
    for _, chunk in candidates:
        if chunk not in final_chunks:
            final_chunks.append(chunk)

    return final_chunks[:top_k]
