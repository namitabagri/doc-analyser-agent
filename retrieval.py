import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(query_embedding, chunk_embeddings):
    similarities = []

    for emb in chunk_embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append(sim)

    # Get index of highest similarity
    best_index = np.argmax(similarities)

    return best_index, similarities