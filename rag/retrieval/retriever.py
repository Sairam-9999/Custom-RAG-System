import re
import math
from collections import Counter
from ..indexing.embedder import embed_query
from ..core.types import RetrievalResult
from ..core.config import RETRIEVAL_CONFIG


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


def retrieve(query, vector_store, chunks, top_k=5, comparison_targets=None):
    # Entity-balanced retrieval for comparative queries
    if comparison_targets and len(comparison_targets) >= 2:
        return retrieve_entity_balanced(
            query, vector_store, chunks, comparison_targets, top_k
        )
    
    query_embedding = embed_query(query)

    semantic_scores, semantic_indices = vector_store.search(
        query_embedding,
        top_k=min(top_k * RETRIEVAL_CONFIG.semantic_candidate_multiplier, len(chunks)),
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

    candidate_indices.update(bm25_ranked[: min(top_k * RETRIEVAL_CONFIG.lexical_candidate_multiplier, len(chunks))])

    scored = []

    for idx in candidate_indices:
        semantic = semantic_map.get(idx, 0.0)
        lexical = bm25_norm[idx]

        # Generic hybrid score. Lexical is slightly stronger for factual QA.
        final_score = (RETRIEVAL_CONFIG.semantic_weight * semantic) + (RETRIEVAL_CONFIG.lexical_weight * lexical)

        scored.append((final_score, idx, chunks[idx]))

    scored.sort(reverse=True, key=lambda x: x[0])

    results = []
    seen = set()

    for final_score, idx, chunk in scored:
        if chunk not in seen:
            semantic = semantic_map.get(idx, 0.0)
            lexical = bm25_norm[idx]

            results.append(
                RetrievalResult(
                    chunk_id=idx,
                    text=chunk,
                    semantic_score=semantic,
                    bm25_score=lexical,
                    hybrid_score=final_score,
                )
            )
            seen.add(chunk)

        if len(results) >= top_k:
            break

    return results


def retrieve_entity_balanced(query, vector_store, chunks, comparison_targets, top_k=5):
    """
    Perform entity-balanced retrieval for comparative queries.
    
    Instead of retrieving with the full query (which may favor one entity),
    retrieve separately for each comparison target and merge results with balancing.
    
    Args:
        query: Original query (for context)
        vector_store: Vector store for semantic search
        chunks: List of document chunks
        comparison_targets: List of entities being compared
        top_k: Total number of results to return
        
    Returns:
        List of RetrievalResult objects with balanced entity representation
    """
    
    # Calculate per-entity budget (ensure both entities get fair representation)
    per_entity_k = max(RETRIEVAL_CONFIG.min_entity_results, top_k // max(len(comparison_targets), 1))
    
    # Retrieve for each entity separately
    entity_results = {}
    for target in comparison_targets:
        # Build entity-focused query
        entity_query = f"{target}"
        
        query_embedding = embed_query(entity_query)
        semantic_scores, semantic_indices = vector_store.search(
            query_embedding,
            top_k=min(per_entity_k * RETRIEVAL_CONFIG.semantic_candidate_multiplier, len(chunks))
        )
        
        bm25_raw = bm25_scores(entity_query, chunks)
        bm25_norm = normalize(bm25_raw)
        
        semantic_map = {}
        semantic_norm = normalize([float(s) for s in semantic_scores])
        
        for idx, score in zip(semantic_indices, semantic_norm):
            if 0 <= idx < len(chunks):
                semantic_map[idx] = score
        
        candidate_indices = set(semantic_map.keys())
        
        # Add strongest BM25 candidates
        bm25_ranked = sorted(
            range(len(chunks)),
            key=lambda i: bm25_norm[i],
            reverse=True,
        )
        candidate_indices.update(bm25_ranked[: min(per_entity_k * RETRIEVAL_CONFIG.lexical_candidate_multiplier, len(chunks))])
        
        scored = []
        for idx in candidate_indices:
            semantic = semantic_map.get(idx, 0.0)
            lexical = bm25_norm[idx]
            final_score = (RETRIEVAL_CONFIG.semantic_weight * semantic) + (RETRIEVAL_CONFIG.lexical_weight * lexical)
            scored.append((final_score, idx, chunks[idx]))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Select top per_entity_k results for this entity
        entity_specific_results = []
        seen = set()
        for final_score, idx, chunk in scored:
            if chunk not in seen:
                semantic = semantic_map.get(idx, 0.0)
                lexical = bm25_norm[idx]
                
                result = RetrievalResult(
                    chunk_id=idx,
                    text=chunk,
                    semantic_score=semantic,
                    bm25_score=lexical,
                    hybrid_score=final_score,
                )
                entity_specific_results.append(result)
                seen.add(chunk)
            
            if len(entity_specific_results) >= per_entity_k:
                break
        
        entity_results[target] = entity_specific_results
    
    # Merge results with balancing
    merged_results = []
    seen_chunks = set()
    
    # Interleave results from each entity to ensure balance
    max_results_per_entity = max(len(results) for results in entity_results.values())
    
    for i in range(max_results_per_entity):
        for target in comparison_targets:
            if i < len(entity_results[target]):
                result = entity_results[target][i]
                if result.text not in seen_chunks:
                    merged_results.append(result)
                    seen_chunks.add(result.text)
    
    # Trim to top_k if needed
    merged_results = merged_results[:top_k]
    
    if RETRIEVAL_CONFIG.debug:
        print(f"\n=== ENTITY BALANCED RETRIEVAL ===")
        print(f"Comparison targets: {comparison_targets}")
        print(f"Total merged chunks: {len(merged_results)}")
        for i, r in enumerate(merged_results):
            print(f"  [{i+1}] Score: {r.hybrid_score:.3f} | {r.text[:100]}...")
        print("=== END ENTITY BALANCED RETRIEVAL ===\n")
    
    return merged_results
