import os
import pytest
import numpy as np
import onnxruntime as ort
from unittest.mock import patch, MagicMock

from src.model import ISNetModel, load_model

class TestISNetModel:
    
    @pytest.fixture
    def mock_onnx_session(self):
        """Fixture to create a mock ONNX session"""
        mock_session = MagicMock()
        # Configure the mock session to return random data when run
        mock_session.run.return_value = [np.random.rand(1, 1, 1024, 1024).astype(np.float32)]
        return mock_session
    
    @patch('onnxruntime.InferenceSession')
    def test_model_initialization(self, mock_ort_session, mock_onnx_session):
        """Test model initialization"""
        mock_ort_session.return_value = mock_onnx_session
        
        # Initialize the model
        model = ISNetModel(model_path="dummy_path")
        
        # Check if model was initialized correctly
        assert model.session is not None
        assert model.input_name == 'input'
        assert model.output_name == 'output'
    
    @patch('onnxruntime.InferenceSession')
    def test_model_predict(self, mock_ort_session, mock_onnx_session):
        """Test model prediction"""
        mock_ort_session.return_value = mock_onnx_session
        
        # Initialize the model
        model = ISNetModel(model_path="dummy_path")
        
        # Create a dummy input tensor
        input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
        
        # Run prediction
        output = model.predict(input_tensor)
        
        # Check if output has the right shape
        assert output.shape == (1, 1, 1024, 1024)
        assert output.dtype == np.float32
        
        # Verify that run was called with correct arguments
        mock_onnx_session.run.assert_called_once()
        args, kwargs = mock_onnx_session.run.call_args
        assert args[0] == [model.output_name]
        assert kwargs['input_feed'] == {model.input_name: input_tensor}
    
    @patch('os.path.exists')
    @patch('onnxruntime.InferenceSession')
    def test_load_model_function(self, mock_ort_session, mock_path_exists, mock_onnx_session):
        """Test the load_model function"""
        # Configure mocks
        mock_path_exists.return_value = True
        mock_ort_session.return_value = mock_onnx_session
        
        # Call the function
        model = load_model("dummy_path")
        
        # Check if the function returns an ISNetModel instance
        assert isinstance(model, ISNetModel)
    
    @patch('os.path.exists')
    def test_load_model_file_not_found(self, mock_path_exists):
        """Test load_model when file doesn't exist"""
        mock_path_exists.return_value = False
        
        # Check if the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_path")

@pytest.mark.integration
def test_model_inference_shape():
    """Test the model inference produces correct output shape"""
    # Skip if model file doesn't exist (for CI environments without model file)
    model_path = os.environ.get('TEST_MODEL_PATH', '/opt/ml/model/model.onnx')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")
    
    # Load the model
    model = ISNetModel(model_path=model_path)
    
    # Create a dummy input tensor
    input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    
    # Run prediction
    output = model.predict(input_tensor)
    
    # Check output shape matches input spatial dimensions
    assert output.shape == (1, 1, 1024, 1024)