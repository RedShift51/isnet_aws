import os
import pytest
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add src directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

# Import functions from inference.py
from src.inference import preprocess_image, postprocess_output, run_inference, predict, normalize, create_response

# Define paths to test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture
def sample_image():
    """Fixture to provide a sample image for testing"""
    # Create a simple test image if it doesn't exist
    img_path = os.path.join(TEST_DATA_DIR, "sample.jpg")
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
    
    if not os.path.exists(img_path):
        # Create a simple image for testing
        img = Image.new('RGB', (1024, 768), color=(73, 109, 137))
        img.save(img_path)
    
    return Image.open(img_path)

@pytest.fixture
def sample_image_bytes(sample_image):
    """Fixture to provide sample image bytes for testing"""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_normalize():
    """Test normalize function"""
    # Create a test array
    test_array = np.array([1.0, 2.0, 3.0])
    
    # Test with default parameters
    normalized = normalize(test_array)
    expected = np.array([0.5, 1.5, 2.5])
    assert np.allclose(normalized, expected)
    
    # Test with custom mean and std
    normalized = normalize(test_array, mean=1.0, std=2.0)
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(normalized, expected)

def test_preprocess_image(sample_image):
    """Test image preprocessing function"""
    processed = preprocess_image(sample_image)
    
    # Check dimensions and type
    assert processed.shape[0] == 1, "Batch dimension should be 1"
    assert processed.shape[1] == 3, "Should have 3 channels (RGB)"
    assert processed.shape[2] == 1024, "Height should be 1024"
    assert processed.shape[3] == 1024, "Width should be 1024"
    assert isinstance(processed, np.ndarray), "Output should be numpy array"
    assert processed.dtype == np.float32, "Array should be float32"
    
    # Check value range (0-1)
    assert np.max(processed) <= 1.0, "Max value should be <= 1.0"
    assert np.min(processed) >= 0.0, "Min value should be >= 0.0"

def test_preprocess_image_from_bytes(sample_image_bytes):
    """Test preprocessing from image bytes"""
    processed = preprocess_image(sample_image_bytes)
    
    # Check dimensions and type
    assert processed.shape[0] == 1, "Batch dimension should be 1"
    assert processed.shape[1] == 3, "Should have 3 channels (RGB)"
    assert processed.shape[2] == 1024, "Height should be 1024"
    assert processed.shape[3] == 1024, "Width should be 1024"
    assert isinstance(processed, np.ndarray), "Output should be numpy array"
    assert processed.dtype == np.float32, "Array should be float32"

def test_postprocess_output():
    """Test mask post-processing"""
    # Create a mock output tensor
    mock_output = np.random.rand(1, 1, 512, 512).astype(np.float32)
    
    result = postprocess_output(mock_output)
    
    # Check dimensions and value range
    assert result.shape == (512, 512), "Output should be 2D image"
    assert result.min() >= 0 and result.max() <= 255, "Values should be in 0-255 range"
    assert result.dtype == np.uint8, "Result should be uint8"

def test_create_response():
    """Test response creation"""
    # Create a mock mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Add a white square
    
    # Test PNG response
    png_response = create_response(mask, format='png')
    assert isinstance(png_response, bytes)
    assert png_response.startswith(b'\x89PNG')
    
    # Test JPEG response
    jpeg_response = create_response(mask, format='jpeg')
    assert isinstance(jpeg_response, bytes)
    assert jpeg_response.startswith(b'\xff\xd8')
    
    # Test base64 encoding
    base64_response = create_response(mask, format='png', encode_base64=True)
    assert isinstance(base64_response, str)
    assert len(base64_response) > 0

@pytest.mark.integration
def test_predict_integration(sample_image_bytes, monkeypatch):
    """Test the full prediction pipeline with mocked model inference"""
    
    # Mock the run_inference function to avoid actual model loading
    def mock_run_inference(input_data):
        # Return fake segmentation mask
        return np.random.rand(1, 1, input_data.shape[2], input_data.shape[3]).astype(np.float32)
    
    # Apply monkeypatch to skip actual model inference
    monkeypatch.setattr("src.inference.run_inference", mock_run_inference)
    
    # Run prediction
    result = predict(sample_image_bytes)
    
    # Check the result
    assert isinstance(result, bytes), "Result should be bytes"
    assert result.startswith(b'\x89PNG') or result.startswith(b'\xff\xd8'), "Should be a valid image format"