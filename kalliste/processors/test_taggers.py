"""Test suite for the image tagging system."""

import pytest
import asyncio
from pathlib import Path
from .taggers import ImageTagger, TagResult

@pytest.mark.asyncio
async def test_orientation_tagger():
    """Test the orientation tagger with a test image"""
    tagger = ImageTagger(device='cpu')  # Use CPU for testing
    
    # Update this path to point to one of your test images
    test_image = Path("../../test_images/test_person.jpg")  # Adjust path as needed
    
    if not test_image.exists():
        pytest.skip(f"Test image not found at {test_image}")
    
    results = await tagger.tag_image(test_image)
    
    # Basic validation
    assert 'orientation' in results
    assert len(results['orientation']) > 0
    
    # Check first result structure
    first_result = results['orientation'][0]
    assert isinstance(first_result, TagResult)
    assert first_result.label in ['front', 'back', 'side']
    assert 0 <= first_result.confidence <= 1
    assert first_result.category == 'orientation'

@pytest.mark.asyncio
async def test_invalid_image():
    """Test handling of invalid image paths"""
    tagger = ImageTagger(device='cpu')
    results = await tagger.tag_image("nonexistent.jpg")
    assert results.get('orientation', []) == []

if __name__ == "__main__":
    asyncio.run(test_orientation_tagger())
