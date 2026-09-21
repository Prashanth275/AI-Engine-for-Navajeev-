from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# load .env
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore(
    index_name="langchainvector",
    embedding=embedding,
    pinecone_api_key=pinecone_api_key
)

results = vectorstore.similarity_search("pregnancy diet", k=5)

print(results)
