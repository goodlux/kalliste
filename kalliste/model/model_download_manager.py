from huggingface_hub import hf_hub_download
import logging
import csv

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    async def load_tags(self):
        """Load the tags from selected_tags.csv."""
        try:
            logger.info("Loading tags")
            tags_path = hf_hub_download(
                repo_id="SmilingWolf/wd-vit-large-tagger-v3",
                filename="selected_tags.csv"
            )
            
            with open(tags_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                tags = [row[0] for row in reader]
                
            logger.info(f"Loaded {len(tags)} tags")
            return tags
            
        except Exception as e:
            logger.error(f"Failed to load tags: {str(e)}")
            raise