import os
import json
import urllib.request
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
index_name = os.getenv("PINECONE_INDEX_NAME", "langchainvector-ollama")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)


def embed_query(text: str) -> list[float]:
    req_data = json.dumps({
        "model": ollama_embed_model,
        "prompt": text
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_base_url}/api/embeddings",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("embedding", [])


def query_pdf(user_query: str, k: int = 3):
    query_vec = embed_query(user_query)
    response = index.query(vector=query_vec, top_k=k, include_metadata=True)
    return [(match["metadata"].get("text", ""), match["metadata"]) for match in response.get("matches", [])]


def generate_answer(context: str, question: str) -> str:
    prompt = f"""You are a helpful assistant who assists women from their 0th day of pregnancy until the child becomes 2 years old.
Use the following context from a PDF to answer the user's question. Be concise and only use information from the context. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""
    req_data = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_base_url}/api/generate",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


if __name__ == "__main__":
    print("Enter your question:")
    user_query = input().strip()
    if not user_query:
        user_query = "What are the feeding guidelines for a newborn?"
    results = query_pdf(user_query, k=3)
    context = "\n---\n".join([content for content, _ in results])
    answer = generate_answer(context, user_query)
    print("\nAnswer:\n", answer)
