from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embeddings(chunks):
    return model.encode(chunks, normalize_embeddings=True)

def embed_query(query):
    query = "Represent this question for retrieval: " + query
    return model.encode([query], normalize_embeddings=True)
