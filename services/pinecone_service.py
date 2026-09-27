import os
import json
import math
import urllib.request
from typing import List, Tuple, Dict, Any
from utils.text_utils import normalize_query
from dotenv import load_dotenv
from fastapi import HTTPException
from pinecone import Pinecone

# Explicitly load .env from project root directory
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

PINECONE_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pinecone_cache_cloud.json"
)

if os.path.exists(PINECONE_CACHE_FILE):
    try:
        with open(PINECONE_CACHE_FILE, "r", encoding="utf-8") as f:
            pinecone_cache = json.load(f)
    except Exception:
        pinecone_cache = []
else:
    pinecone_cache = []


def save_pinecone_cache():
    try:
        temp_file = PINECONE_CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(pinecone_cache, f, indent=2)
        os.replace(temp_file, PINECONE_CACHE_FILE)
    except Exception as e:
        print(f"[PINECONE NOTICE] Failed to save pinecone cache: {e}")


def embed_query_with_pinecone(text: str) -> List[float]:
    """
    Generate 768-dimensional query embedding vector via Pinecone Hosted Inference (llama-text-embed-v2).
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=pinecone_api_key)
    response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={
            "input_type": "query",
            "truncate": "END",
            "dimension": 768
        }
    )
    return response.data[0].values


def get_vectorstore():
    """
    Returns configured Pinecone index client for langchainvector-cloud.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "langchainvector-cloud")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def simple_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())

    union = set_a | set_b
    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


def similarity_search(vectorstore, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Run similarity search against Pinecone vector store using Pinecone Hosted Inference (llama-text-embed-v2, 768 dim).
    Returns list of (content, metadata) tuples.
    """
    if vectorstore is None:
        try:
            vectorstore = get_vectorstore()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Vector store is not initialized: {str(e)}"
            )

    query_clean = normalize_query(query)

    # 1. Exact Cache Check
    for item in pinecone_cache:
        if item.get("query") == query_clean:
            print("[PINECONE EXACT CACHE HIT] (Cloud)")
            return item["result"]

    # 2. String Similarity Cache Check
    for item in pinecone_cache:
        score = simple_similarity(query_clean, item.get("query", ""))
        if score > 0.7:
            print("[PINECONE STRING CACHE HIT] (Cloud)")
            return item["result"]

    # 3. Generate Hosted Query Embedding (llama-text-embed-v2)
    query_embedding = None
    try:
        query_embedding = embed_query_with_pinecone(query)
    except Exception as e:
        print("[PINECONE NOTICE] Hosted embedding failed:", repr(e))
        query_embedding = None

    # 4. Semantic Similarity Cache Check
    if query_embedding is not None:
        for item in pinecone_cache:
            cached_embedding = item.get("embedding")
            if cached_embedding and len(cached_embedding) == len(query_embedding):
                similarity = cosine_similarity(query_embedding, cached_embedding)
                if similarity > 0.85:
                    print(f"[PINECONE SEMANTIC CACHE HIT] (score: {similarity:.2f})")
                    return item["result"]

    if query_embedding is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to generate query embedding via Pinecone Hosted Inference (llama-text-embed-v2)."
        )

    # 5. Query Pinecone
    print("[PINECONE API CALL] Executing similarity search on langchainvector-cloud...")
    try:
        response = vectorstore.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        formatted = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            text_content = metadata.get("text", "")
            formatted.append((text_content, metadata))

        # Update Cache
        if not any(item.get("query") == query_clean for item in pinecone_cache):
            pinecone_cache.append({
                "query": query_clean,
                "embedding": [round(x, 4) for x in query_embedding],
                "result": formatted
            })

        if len(pinecone_cache) > 500:
            pinecone_cache.pop(0)

        save_pinecone_cache()

        return formatted
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pinecone search error: {str(e)}"
        )
