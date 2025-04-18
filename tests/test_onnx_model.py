import os
import sys
import pytest
import numpy as np
import onnxruntime as ort
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

# Import from inference.py
from src.inference import load_model

@pytest.fixture
def mock_onnx_session():
    """Fixture to create a mock ONNX session"""
    mock_session = MagicMock()
    # Configure the mock session to return random data when run
    mock_session.run.return_value = [np.random.rand(1, 1, 1024, 1024).astype(np.float32)]
    
    # Mock the inputs and outputs
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 1024, 1024]
    mock_input.type = "tensor(float)"
    
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.shape = [1, 1, 1024, 1024]
    mock_output.type = "tensor(float)"
    
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    
    return mock_session

@pytest.mark.integration
@patch('onnxruntime.SessionOptions')
@patch('onnxruntime.InferenceSession')
def test_load_model(mock_inference_session, mock_session_options, mock_onnx_session):
    """Test loading the ONNX model"""
    # Configure mock
    mock_inference_session.return_value = mock_onnx_session
    
    # Set environment variable to a test path
    test_model_path = "/opt/ml/model/isnet_converted1.onnx"
    with patch.dict('os.environ', {'MODEL_PATH': test_model_path}):
        with patch('os.path.exists', return_value=True):
            # Call the function
            session = load_model()
            
            # Check if session was loaded
            assert session is not None
            
            # Verify InferenceSession was called with correct args
            mock_inference_session.assert_called_once()
            call_args = mock_inference_session.call_args
            assert test_model_path in call_args[0]
            assert "CPUExecutionProvider" in call_args[1]['providers']

@patch('onnxruntime.SessionOptions')
@patch('onnxruntime.InferenceSession')
def test_load_model_session_options(mock_inference_session, mock_session_options, mock_onnx_session):
    """Test that proper session options are set when loading the model"""
    # Configure mock
    mock_inference_session.return_value = mock_onnx_session
    mock_options = MagicMock()
    mock_session_options.return_value = mock_options
    
    # Set environment variable to a test path
    test_model_path = "/opt/ml/model/model.onnx"
    with patch.dict('os.environ', {'MODEL_PATH': test_model_path}):
        with patch('os.path.exists', return_value=True):
            # Call the function
            session = load_model()
            
            # Check if session options were configured correctly
            assert mock_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            assert mock_options.intra_op_num_threads == 2

def test_load_model_behavior():
    """Test how load_model behaves when the model file doesn't exist"""
    from src.inference import load_model
    
    # Test what happens when calling load_model on a non-existent path
    # We'll mock InferenceSession to return a mocked session instead of raising an error
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.get_outputs.return_value = [MagicMock(name="output")]
    
    # Mock the os.path.exists to simulate model file not existing
    with patch('os.path.exists', return_value=False):
        # Mock the os.listdir to simulate no onnx files
        with patch('os.listdir', return_value=[]):
            # Mock the InferenceSession to return our mock
            with patch('onnxruntime.InferenceSession', return_value=mock_session):
                # Just verify that the function returns something
                result = load_model()
                assert result is not None, "load_model should return a session object"

@pytest.mark.integration
def test_run_inference_with_mock_model():
    """Test run_inference with a mock model"""
    from src.inference import run_inference
    
    # Create a mock input tensor
    input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    
    # Mock the session
    mock_session = MagicMock()
    mock_session.run.return_value = [np.random.rand(1, 1, 1024, 1024).astype(np.float32)]
    
    # Mock get_inputs and get_outputs
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_output = MagicMock()
    mock_output.name = "output"
    
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    
    # Patch the global session variable and load_model function
    with patch('src.inference.session', mock_session), \
         patch('src.inference.load_model', return_value=mock_session):
        
        # Run inference
        output = run_inference(input_tensor)
        
        # Check output shape
        assert output.shape == (1, 1, 1024, 1024)
        
        # Verify session.run was called with correct arguments
        mock_session.run.assert_called_once()
        args, kwargs = mock_session.run.call_args
        assert args[0] == ["output"]
        assert "input" in kwargs["input_feed"]
        np.testing.assert_array_equal(kwargs["input_feed"]["input"], input_tensor)