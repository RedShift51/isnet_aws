import os
import pytest
import numpy as np
from PIL import Image
import io
from src.inference import preprocess_image, postprocess_output, predict

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
        img = Image.new('RGB', (512, 512), color = (73, 109, 137))
        img.save(img_path)
    
    return Image.open(img_path)

@pytest.fixture
def sample_image_bytes(sample_image):
    """Fixture to provide sample image bytes for testing"""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_preprocess_image(sample_image):
    """Test image preprocessing function"""
    processed = preprocess_image(sample_image)
    
    # Check dimensions and type
    assert processed.shape[0] == 1, "Batch dimension should be 1"
    assert processed.shape[1] == 3, "Should have 3 channels (RGB)"
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

def test_predict_integration(sample_image, monkeypatch):
    """Test the full prediction pipeline with mocked model inference"""
    
    # Mock the ONNX inference to avoid actual model loading
    def mock_inference(input_data):
        # Return fake segmentation mask
        return np.random.rand(1, 1, input_data.shape[2], input_data.shape[3]).astype(np.float32)
    
    # Apply monkeypatch to skip actual model inference
    import src.inference
    monkeypatch.setattr(src.inference, "run_inference", mock_inference)
    
    # Run prediction
    result = predict(sample_image)
    
    # Check the result
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.shape == (sample_image.height, sample_image.width), "Output dimensions should match input"
    assert result.dtype == np.uint8, "Result should be uint8"

def test_predict_with_bytes(sample_image_bytes, monkeypatch):
    """Test prediction with image bytes input"""
    
    # Mock the ONNX inference
    def mock_inference(input_data):
        return np.random.rand(1, 1, input_data.shape[2], input_data.shape[3]).astype(np.float32)
    
    # Apply monkeypatch
    import src.inference
    monkeypatch.setattr(src.inference, "run_inference", mock_inference)
    
    # Run prediction with bytes
    result = predict(sample_image_bytes)
    
    # Check the result
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.dtype == np.uint8, "Result should be uint8"