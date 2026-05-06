import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = np.array(embeddings).astype("float32")
        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=10):
        query_embedding = np.array(query_embedding).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]
