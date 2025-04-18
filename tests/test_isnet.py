import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ISNet models from isnet.py
from src.isnet import ISNetDIS, ISNetGTEncoder, RSU7, REBNCONV, muti_loss_fusion

@pytest.fixture
def sample_input():
    """Fixture to provide a sample input tensor for testing"""
    return torch.randn(1, 3, 512, 512)

@pytest.fixture
def sample_target():
    """Fixture to provide a sample target tensor for testing"""
    return torch.randn(1, 1, 512, 512).sigmoid()

class TestISNetComponents:
    
    def test_rebnconv(self):
        """Test REBNCONV module"""
        # Create a REBNCONV module
        rebnconv = REBNCONV(in_ch=3, out_ch=16, dirate=1)
        
        # Create a sample input
        x = torch.randn(1, 3, 64, 64)
        
        # Forward pass
        output = rebnconv(x)
        
        # Check output shape
        assert output.shape == (1, 16, 64, 64)
        
        # Check if output contains only finite values
        assert torch.isfinite(output).all()
    
    def test_rsu7(self):
        """Test RSU7 module"""
        # Create an RSU7 module
        rsu7 = RSU7(in_ch=3, mid_ch=12, out_ch=16)
        
        # Create a sample input
        x = torch.randn(1, 3, 64, 64)
        
        # Forward pass
        output = rsu7(x)
        
        # Check output shape
        assert output.shape == (1, 16, 64, 64)
        
        # Check if output contains only finite values
        assert torch.isfinite(output).all()
    
    def test_muti_loss_fusion(self):
        """Test muti_loss_fusion function"""
        # Create sample predictions and target
        preds = [torch.rand(1, 1, 512, 512).sigmoid() for _ in range(3)]
        target = torch.rand(1, 1, 512, 512).sigmoid()
        
        # Compute loss
        loss0, loss = muti_loss_fusion(preds, target)
        
        # Check if losses are scalars
        assert loss0.dim() == 0
        assert loss.dim() == 0
        
        # Check if losses are positive
        assert loss0.item() > 0
        assert loss.item() > 0
        
        # Check if total loss is greater than or equal to the first loss
        assert loss.item() >= loss0.item()

class TestISNetDIS:
    
    def test_model_initialization(self):
        """Test ISNetDIS initialization"""
        # Create the model
        model = ISNetDIS(in_ch=3, out_ch=1)
        
        # Check that the model was created successfully
        assert isinstance(model, ISNetDIS)
        
        # Check some key components
        assert hasattr(model, 'conv_in')
        assert hasattr(model, 'stage1')
        assert hasattr(model, 'side1')
    
    def test_model_forward(self, sample_input):
        """Test ISNetDIS forward pass"""
        # Skip if on CI environment without enough memory
        if os.environ.get('CI') == 'true':
            pytest.skip("Skipping memory-intensive test on CI")
        
        # Create the model
        model = ISNetDIS(in_ch=3, out_ch=1)
        
        # Forward pass
        with torch.no_grad():
            outputs, features = model(sample_input)
        
        # Check number of outputs (6 side outputs)
        assert len(outputs) == 6
        
        # Check shape of outputs
        for output in outputs:
            assert output.shape == (1, 1, 512, 512)
        
        # Check number of feature maps
        assert len(features) == 6
    
    def test_compute_loss(self, sample_input, sample_target):
        """Test ISNetDIS loss computation"""
        # Skip if on CI environment without enough memory
        if os.environ.get('CI') == 'true':
            pytest.skip("Skipping memory-intensive test on CI")
        
        # Create the model
        model = ISNetDIS(in_ch=3, out_ch=1)
        
        # Forward pass
        with torch.no_grad():
            outputs, _ = model(sample_input)
        
        # Compute loss
        loss0, loss = model.compute_loss(outputs, sample_target)
        
        # Check if losses are scalars
        assert loss0.dim() == 0
        assert loss.dim() == 0
        
        # Check if losses are positive
        assert loss0.item() > 0
        assert loss.item() > 0

class TestISNetGTEncoder:
    
    def test_model_initialization(self):
        """Test ISNetGTEncoder initialization"""
        # Create the model
        model = ISNetGTEncoder(in_ch=1, out_ch=1)
        
        # Check that the model was created successfully
        assert isinstance(model, ISNetGTEncoder)
        
        # Check some key components
        assert hasattr(model, 'conv_in')
        assert hasattr(model, 'stage1')
        assert hasattr(model, 'side1')
    
    @pytest.mark.integration
    def test_model_forward(self):
        """Test ISNetGTEncoder forward pass"""
        # Create a small input tensor (1 channel for GT)
        x = torch.rand(1, 1, 64, 64)
        
        # Create the model with smaller size for testing
        model = ISNetGTEncoder(in_ch=1, out_ch=1)
        
        # Forward pass
        with torch.no_grad():
            outputs, features = model(x)
        
        # Check number of outputs (6 side outputs)
        assert len(outputs) == 6
        
        # Check shape of outputs (should match input spatial dimensions)
        for output in outputs:
            assert output.shape[0] == 1  # batch size
            assert output.shape[1] == 1  # channels
            assert output.shape[2:] == (64, 64)  # spatial dimensions
        
        # Check number of feature maps
        assert len(features) == 6