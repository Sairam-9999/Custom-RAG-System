import faiss
import numpy as np


class VectorStore:
    def __init__(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty.")

        self.embeddings = embeddings
        self.dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=10):
        query_embedding = np.asarray(query_embedding, dtype="float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} "
                f"does not match index dimension {self.dim}."
            )

        top_k = min(top_k, len(self.embeddings))

        scores, indices = self.index.search(query_embedding, top_k)

        return scores[0], indices[0]
