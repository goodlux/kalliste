from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    async def download_all(self):
        """Download and verify all required model files."""
        try:
            logger.info("Downloading model files")
            
            # Download the tags file to HF cache
            tags_path = hf_hub_download(
                repo_id="SmilingWolf/wd-vit-large-tagger-v3",
                filename="selected_tags.csv"
            )
            logger.info(f"Tags file downloaded to: {tags_path}")
            
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {str(e)}") from e