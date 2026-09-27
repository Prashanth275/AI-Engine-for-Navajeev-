import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TARGET_INDEX_NAME = os.getenv("PINECONE_CLOUD_INDEX_NAME", "langchainvector-cloud")
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 768

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")


def load_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(docs)


def embed_passages_with_pinecone(pc: Pinecone, texts: list[str]) -> list[list[float]]:
    """
    Generate 768-dimensional embeddings using Pinecone Hosted Inference API (llama-text-embed-v2).
    """
    response = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={
            "input_type": "passage",
            "truncate": "END",
            "dimension": EMBEDDING_DIMENSION
        }
    )
    return [item.values for item in response.data]


def main():
    print("==================================================")
    print("PINECONE HOSTED EMBEDDING INGESTION (CLOUD)")
    print("==================================================")
    print(f"[CONFIG] Embedding Model: {EMBEDDING_MODEL} (Hosted Inference)")
    print(f"[CONFIG] Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"[CONFIG] Target Pinecone Index: {TARGET_INDEX_NAME}")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1. Verify embedding dimension and API connectivity
    print("\n[STEP 1] Verifying Pinecone hosted embedding API...")
    test_vectors = embed_passages_with_pinecone(pc, ["Maternal and Infant Health Test Passage"])
    dim = len(test_vectors[0])
    print(f"[VERIFIED] Pinecone Hosted Embedding Dimension: {dim} (Expected: {EMBEDDING_DIMENSION})")
    if dim != EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected embedding dimension: {dim}. Expected {EMBEDDING_DIMENSION}.")

    # 2. Load and split PDF
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JOURNEY OF THE FIRST 100 DAYS.pdf")
    print(f"\n[STEP 2] Loading and splitting PDF from: {pdf_path}")
    documents = load_and_split(pdf_path)
    total_docs = len(documents)
    print(f"[LOADER] Total chunks created: {total_docs} (chunk_size=600, chunk_overlap=80)")

    # 3. Connect to target Pinecone index
    print(f"\n[STEP 3] Connecting to Pinecone index: {TARGET_INDEX_NAME}...")
    index = pc.Index(TARGET_INDEX_NAME)
    print(f"[PINECONE] Successfully connected to: {TARGET_INDEX_NAME}")

    # 4. Embed and Upsert in batches using deterministic IDs
    batch_size = 50
    print(f"\n[STEP 4] Embedding and uploading {total_docs} vectors in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        
        # Call Pinecone Hosted Inference Embeddings
        batch_embeddings = embed_passages_with_pinecone(pc, texts)
        
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
        print(f"  -> Uploaded chunks {i} to {min(i + batch_size, total_docs) - 1} ({min(i + batch_size, total_docs)}/{total_docs})")

    # 5. Confirm index stats
    print("\n[STEP 5] Fetching updated index statistics...")
    stats = index.describe_index_stats()
    print("==================================================")
    print(f"[COMPLETE] Ingestion Complete!")
    print(f"Target Index: {TARGET_INDEX_NAME}")
    print(f"Total Vector Count: {stats.total_vector_count}")
    print(f"Dimension: {stats.dimension}")
    print("==================================================")


if __name__ == "__main__":
    main()
