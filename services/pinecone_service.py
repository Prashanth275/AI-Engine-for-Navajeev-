import os
import json
import hashlib
import math
from utils.text_utils import normalize_query
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Explicitly load .env from project root directory
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            _embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=openai_key
            )
        else:
            _embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
    return _embedding_model


PINECONE_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinecone_cache.json")

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
        with open(PINECONE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pinecone_cache, f, indent=2)
    except Exception as e:
        print(f"[PINECONE NOTICE] Failed to save pinecone cache: {e}")


def get_vectorstore():
    """
    Returns configured Pinecone vectorstore.
    Uses the same index name and embedding model as original backend.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    embedding = get_embedding_model()

    return PineconeVectorStore(
        index_name="langchainvector",
        embedding=embedding,
        pinecone_api_key=pinecone_api_key
    )


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


def simple_similarity(a, b):
    set_a = set(a.split())
    set_b = set(b.split())

    union = set_a | set_b
    if not union:
        return 0

    return len(set_a & set_b) / len(union)


def similarity_search(vectorstore, query: str, k: int = 5):
    """
    Run similarity search against Pinecone vector store with multi-tier caching.
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
    for item in pinecone_cache:
        if item.get("query") == query_clean:
            print("[PINECONE EXACT CACHE HIT]")
            return item["result"]

    for item in pinecone_cache:
        score = simple_similarity(query_clean, item.get("query", ""))
        if score > 0.7:
            print("[PINECONE STRING CACHE HIT]")
            return item["result"]

    query_embedding = None
    try:
        embedder = get_embedding_model()
        query_embedding = embedder.embed_query(query)
    except Exception as e:
        print("[PINECONE NOTICE] Embedding failed:", repr(e))
        print("[PINECONE NOTICE] Falling back to direct Pinecone search")
        query_embedding = None

    # Semantic cache check
    if query_embedding is not None:
        for item in pinecone_cache:
            cached_embedding = item.get("embedding")
            if cached_embedding:
                similarity = cosine_similarity(query_embedding, cached_embedding)
                if similarity > 0.85:
                    print(f"[PINECONE SEMANTIC CACHE HIT] (score: {similarity:.2f})")
                    return item["result"]

    print("[PINECONE API CALL] Executing similarity search...")
    try:
        results = vectorstore.similarity_search(query, k=k)
        formatted = [(doc.page_content, doc.metadata) for doc in results]

        if query_embedding is not None:
            if not any(item.get("query") == query_clean for item in pinecone_cache):
                pinecone_cache.append({
                    "query": query_clean,
                    "embedding": [round(x, 2) for x in query_embedding],
                    "result": formatted
                })

        if len(pinecone_cache) > 500:
            pinecone_cache.pop(0)

        if query_embedding is not None:
            save_pinecone_cache()

        return formatted
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pinecone search error: {str(e)}"
        )
