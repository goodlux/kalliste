"""Initialize ChromaDB for Kalliste."""
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Load CHROMADB_PATH from .env
load_dotenv()
CHROMADB_PATH = os.getenv('CHROMADB_PATH')

def init_db():
    """Initialize ChromaDB with persistent storage."""
    client = chromadb.PersistentClient(
        path=CHROMADB_PATH
    )
    
    # Create collections for different types of embeddings
    images = client.get_or_create_collection(
        name="images",
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity for embeddings
    )

    return client, images

if __name__ == "__main__":
    client, images = init_db()
    print("ChromaDB initialized successfully")
    print(f"Collections: {[col.name for col in client.list_collections()]}")
    print(f"Using database at: {CHROMADB_PATH}")
