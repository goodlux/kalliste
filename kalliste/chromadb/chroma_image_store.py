"""ChromaDB integration for Kalliste images."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import chromadb
from chromadb.config import Settings
import numpy as np
import base64
from loguru import logger
from ..config import CHROMADB_DIR

class ChromaImageStore:
    """Manages ChromaDB storage and retrieval for Kalliste images."""
    
    def __init__(self, persist_dir: Optional[Path] = None):
        """Initialize ChromaDB connection.
        
        Args:
            persist_dir: Optional override for ChromaDB directory. Defaults to config value.
        """
        self.persist_dir = persist_dir or CHROMADB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            persist_directory=str(self.persist_dir),
            anonymized_telemetry=False,
            allow_reset=True  # Useful for development
        )
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), 
            settings=settings
        )
        
        # Create our main collection for images
        self.images = self.client.get_or_create_collection(
            name="kalliste_images",
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity for embeddings
                "description": "Main Kalliste image collection with embeddings and metadata"
            }
        )

    def add_image(self, 
                 image_path: Path,
                 kalliste_tags: Dict[str, Any]
                ) -> bool:
        """Add a single image with its metadata to ChromaDB.
        
        Args:
            image_path: Path to the image file
            kalliste_tags: Dictionary of Kalliste metadata tags
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the CLIP embedding
            clip_embedding_tag = kalliste_tags.get("KallisteClipEmbedding")
            if not clip_embedding_tag:
                logger.error("No CLIP embedding found in kalliste_tags")
                return False
                
            # Deserialize the CLIP embedding
            embedding = self._deserialize_clip_embedding(clip_embedding_tag)
            if embedding is None:
                return False
                
            # Generate a unique ID from the image path
            image_id = str(image_path)
            
            # Convert the metadata into a ChromaDB-friendly structure
            metadata = {} #self._prepare_metadata(kalliste_tags)
            
            # Add to ChromaDB
            self.images.add(
                ids=[image_id],
                embeddings=[embedding.tolist()],
                #metadatas=[metadata],
                documents=[kalliste_tags.get("KallisteCaption", "").to_chroma() 
                          if "KallisteCaption" in kalliste_tags else ""]
            )
            
            logger.info(f"Successfully added image to ChromaDB: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add image to ChromaDB: {e}")
            return False
    
    def _deserialize_clip_embedding(self, clip_tag: Any) -> Optional[np.ndarray]:
        """Deserialize CLIP embedding from base64 string."""
        try:
            # Get the base64 string from the tag
            base64_str = clip_tag.to_chroma()
            
            # Decode base64 to bytes
            embedding_bytes = base64.b64decode(base64_str)
            
            # Convert to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to deserialize CLIP embedding: {e}")
            return None

    def _prepare_metadata(self, kalliste_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Kalliste tags into a ChromaDB-friendly structure.
        
        Args:
            kalliste_tags: Raw Kalliste metadata tags
            
        Returns:
            Dict containing metadata optimized for ChromaDB
        """
        # Define the metadata fields we want to keep
        metadata_fields = {
            # Core descriptive metadata
            "person_name": "KallistePersonName",
            "photoshoot_name": "KallistePhotoshootName",
            "source_type": "KallisteSourceType",
            "region_type": "KallisteRegionType",
            "training_target": "KallisteTrainingTarget",
            
            # Assessment scores
            "nima_score_aesthetic": "KallisteNimaScoreAesthetic",
            "nima_score_technical": "KallisteNimaScoreTechnical",
            "nima_calc_average": "KallisteNimaCalcAverage",
            "nima_calc_distribution": "KallisteNimaCalcDistribution",
            
            # Collections/arrays
            "kalliste_tags": "KallisteTags",
            "wd14_attributes": "KallisteWd14Attributes",
            "wd14_content": "KallisteWd14Content",
            "wd14_tags": "KallisteWd14Tags",
            
            # Other metadata
            "orientation_tag": "KallisteOrientationTag",
            "orientation_data": "KallisteOrientationDataRaw",
            "nima_assessment_aesthetic": "KallisteNimaAssessmentAesthetic",
            "nima_assessment_technical": "KallisteNimaAssessmentTechnical",
            "nima_overall_assessment": "KallisteNimaOverallAssessment",
            "original_path": "KallisteOriginalPath",
            "assessment": "KallisteAssessment"
        }
        
        # Convert tags to ChromaDB format
        metadata = {}
        for chroma_key, tag_key in metadata_fields.items():
            if tag_key in kalliste_tags:
                value = kalliste_tags[tag_key].to_chroma()
                if value is not None:  # Skip None values
                    metadata[chroma_key] = value
        
        return metadata
        
    def search(self, 
              query_embedding: np.ndarray,
              n_results: int = 10,
              metadata_filter: Optional[Dict[str, Any]] = None
             ) -> List[Dict[str, Any]]:
        """Basic similarity search for images.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional filter conditions for metadata fields
            
        Returns:
            List of results containing image paths and metadata
        """
        # TODO: Implement basic similarity search
        pass
        
    def search_diverse(self,
                      query_embedding: np.ndarray,
                      n_results: int = 10,
                      metadata_filter: Optional[Dict[str, Any]] = None,
                      diversity_threshold: float = 0.8
                     ) -> List[Dict[str, Any]]:
        """Search for diverse images using MMR (Maximal Marginal Relevance).
        
        This implementation will:
        1. Get a larger pool of similar images
        2. Use MMR to select a diverse subset
        3. Consider both embedding similarity and metadata diversity
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional filter conditions for metadata fields
            diversity_threshold: Balance between relevance and diversity (0-1)
            
        Returns:
            List of diverse results containing image paths and metadata
        """
        # TODO: Implement MMR-based diverse search
        pass
        
    def delete_image(self, image_path: Path) -> bool:
        """Delete an image from ChromaDB.
        
        Args:
            image_path: Path to the image to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_id = str(image_path)
            self.images.delete(ids=[image_id])
            logger.info(f"Successfully deleted image from ChromaDB: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image from ChromaDB: {e}")
            return False

    def reset(self) -> bool:
        """Reset the ChromaDB collection (mainly for development)."""
        try:
            self.client.reset()
            # Recreate our collection
            self.images = self.client.get_or_create_collection(
                name="kalliste_images",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            return False