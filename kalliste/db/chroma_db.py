"""Chroma DB interface for Kalliste."""
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
from ..config import CHROMA_DB_DIR

class ChromaDB:
    def __init__(self):
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False
                )
        )
        
        embedding_function = OpenCLIPEmbeddingFunction()
        data_loader = ImageLoader()
        self.collection = client.get_or_create_collection(
            name="kalliste",
            embedding_function=embedding_function,
            data_loader=data_loader
        )

    def add_image_to_chroma(self, id: str, file_path: str):
        self.collection.add(ids=str(id), uris=str(file_path))