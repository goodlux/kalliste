
import torch
from PIL import Image
from typing import List, Optional, Dict, Any
from pathlib import Path
from ..model.model_registry import ModelRegistry


class EmbeddingGenerator:
    """Generates embeddings using registered models."""
    
    @staticmethod
    def generate_dinov2_embedding(image_path: str) -> List[float]:
        """Generate DINOv2 embedding directly from image file."""
        model_info = ModelRegistry.get_model("dinov2")
        model = model_info["model"]
        processor = model_info["processor"]
        
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Get CLS token embedding
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            return embedding.tolist()

    @staticmethod
    def generate_openclip_embedding(
        image_path: str, 
        text: Optional[str] = None
    ) -> List[float]:
        """Generate OpenCLIP embedding directly from image file."""
        model_info = ModelRegistry.get_model("openclip")
        model = model_info["model"]
        processor = model_info["processor"]
        device = next(model.parameters()).device  # Get device from model parameters
        
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            image_input = processor(image).unsqueeze(0).to(device)
            
            if text:
                text_tokens = model.tokenizer([text]).to(device)
                image_features, text_features = model(image_input, text_tokens)
                combined = (image_features + text_features) / 2
                return combined.cpu().numpy()[0].tolist()
            else:
                image_features = model.encode_image(image_input)
                return image_features.cpu().numpy()[0].tolist()