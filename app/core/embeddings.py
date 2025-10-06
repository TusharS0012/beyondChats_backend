import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List

load_dotenv()

HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")
TOP_K_SNIPPETS = int(os.getenv("TOP_K_SNIPPETS", 3))

embedder = SentenceTransformer(HF_EMBEDDING_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, show_progress_bar=False).tolist()

def search_embedding(query: str, embeddings: List[List[float]], texts: List[str], top_k: int = None) -> List[dict]: # type: ignore
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm

    if top_k is None:
        top_k = TOP_K_SNIPPETS

    query_vec = embedder.encode([query])[0]
    scores = [dot(query_vec, vec)/(norm(query_vec)*norm(vec)) for vec in embeddings]
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [{"text": texts[i], "score": scores[i]} for i in top_idx]
