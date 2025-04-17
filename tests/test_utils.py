import pytest
import numpy as np
from PIL import Image
import io
import os

from src.utils import resize_image, normalize_image, create_response, load_image_from_bytes

@pytest.fixture
def sample_image():
    """Fixture to provide a sample image for testing"""
    return Image.new('RGB', (600, 400), color=(100, 150, 200))

def test_resize_image(sample_image):
    """Test image resizing functionality"""
    # Test with target size
    target_size = (1024, 1024)
    resized = resize_image(sample_image, target_size)
    
    assert resized.size == target_size
    assert isinstance(resized, Image.Image)
    
    # Test with no target size (should return original)
    resized = resize_image(sample_image)
    assert resized.size == sample_image.size
    assert resized is not sample_image  # Should be a copy, not the same object

def test_normalize_image(sample_image):
    """Test image normalization functionality"""
    # Convert PIL image to numpy array
    img_array = np.array(sample_image)
    
    # Normalize the image
    normalized = normalize_image(img_array)
    
    # Check if the result is correct
    assert normalized.dtype == np.float32
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    assert normalized.shape == (*img_array.shape[0:2], 3)  # Preserve spatial dimensions and RGB

def test_create_response():
    """Test response creation functionality"""
    # Create a mock segmentation mask
    mask = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)
    
    # Test with different output formats
    png_response = create_response(mask, format='png')
    assert png_response.startswith(b'\x89PNG')
    
    jpeg_response = create_response(mask, format='jpeg')
    assert jpeg_response.startswith(b'\xff\xd8\xff')
    
    # Test base64 encoding
    b64_response = create_response(mask, format='png', encode_base64=True)
    assert isinstance(b64_response, str)
    assert b64_response.startswith('iVBOR')  # Base64 encoded PNG signature

def test_load_image_from_bytes():
    """Test loading image from bytes"""
    # Create a sample image and get its bytes
    img = Image.new('RGB', (1024, 1024), color=(100, 150, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Load the image from bytes
    loaded_img = load_image_from_bytes(img_bytes)
    
    # Check if the loaded image is correct
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (1024, 1024)
    assert loaded_img.mode == 'RGB'

def test_load_image_from_invalid_bytes():
    """Test loading image from invalid bytes"""
    # Try to load image from invalid bytes
    with pytest.raises(Exception):
        load_image_from_bytes(b'invalid image data')