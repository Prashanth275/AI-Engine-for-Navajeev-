import os
import json
import urllib.request
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
index_name = os.getenv("PINECONE_INDEX_NAME", "langchainvector-ollama")

if not pinecone_api_key:
    raise ValueError("Missing PINECONE_API_KEY in .env")


class OllamaLocalEmbeddings(Embeddings):
    """
    Self-contained Ollama Embeddings class compatible with LangChain and Pinecone.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        req_data = json.dumps({
            "model": self.model,
            "prompt": text
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("embedding", [])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings


def load_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(docs)


def main():
    print("--------------------------------------------------")
    print("[INGESTION] Ingesting documents into Pinecone (Ollama Edition)...")
    print(f"[CONFIG] Embedding Model: {ollama_embed_model} at {ollama_base_url}")
    print(f"[CONFIG] Pinecone Target Index: {index_name}")

    embeddings = OllamaLocalEmbeddings(base_url=ollama_base_url, model=ollama_embed_model)

    # 1. Verify embedding dimension
    test_vector = embeddings.embed_query("Maternal and Infant Health Test")
    dim = len(test_vector)
    print(f"[VERIFIED] Embedding dimension: {dim} (Expected: 768)")
    if dim != 768:
        raise ValueError(f"Unexpected embedding dimension: {dim}. Expected 768.")

    # 2. Load and split PDF
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JOURNEY OF THE FIRST 100 DAYS.pdf")
    print(f"[LOADER] Loading PDF from: {pdf_path}")
    documents = load_and_split(pdf_path)
    total_docs = len(documents)
    print(f"[LOADER] Total chunks created: {total_docs}")

    # 3. Connect to existing Pinecone index
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    print(f"[PINECONE] Connected to Pinecone index: {index_name}")

    # 4. Embed and Upsert in batches with deterministic IDs to prevent duplicate vectors
    batch_size = 25
    print(f"[PROGRESS] Embedding and uploading {total_docs} vectors in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        batch_embeddings = embeddings.embed_documents(texts)
        
        vectors_to_upsert = []
        for j, (doc, vector) in enumerate(zip(batch, batch_embeddings)):
            vector_id = f"doc_chunk_{i + j}"
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}
            metadata["text"] = doc.page_content
            vectors_to_upsert.append({
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            })
            
        index.upsert(vectors=vectors_to_upsert)
        print(f"  -> Uploaded {min(i + batch_size, total_docs)}/{total_docs} vectors")

    # 5. Confirm index stats
    stats = index.describe_index_stats()
    print("--------------------------------------------------")
    print(f"[COMPLETE] Ingestion Complete! Index stats: {stats}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
