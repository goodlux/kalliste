"""Initialize ChromaDB for Kalliste."""
import chromadb
from chromadb.config import Settings

def init_db():
    """Initialize ChromaDB with persistent storage."""
    client = chromadb.PersistentClient(
        path="/Volumes/m01/kalliste_data/chromadb"
    )
    
    # Create collections for different types of embeddings
    images = client.create_collection(
        name="images",
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity for embeddings
    )

    return client, images

if __name__ == "__main__":
    client, images = init_db()
    print("ChromaDB initialized successfully")
    print(f"Collections: {[col.name for col in client.list_collections()]}")
