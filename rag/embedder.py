from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = SentenceTransformer(MODEL_NAME)


def get_embeddings(chunks, batch_size=32):
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    if not clean_chunks:
        raise ValueError("All chunks are empty after cleaning.")

    return _model.encode(
        clean_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query = "Represent this question for retrieving relevant passages: " + query.strip()

    return _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
