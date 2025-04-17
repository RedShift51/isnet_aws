import pytest
import os
import shutil
from PIL import Image
import numpy as np

@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return the path to the test data directory."""
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    yield test_dir
    # Uncomment to clean up test data after tests
    # shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def sample_images(test_data_dir):
    """Create and return paths to sample test images of different sizes."""
    image_paths = {}
    sizes = [(512, 512), (256, 256), (800, 600)]
    
    for width, height in sizes:
        filename = f"sample_{width}x{height}.jpg"
        path = os.path.join(test_data_dir, filename)
        
        if not os.path.exists(path):
            # Create a simple gradient image for testing
            img = Image.new('RGB', (width, height), color=(0, 0, 0))
            pixels = img.load()
            
            for i in range(width):
                for j in range(height):
                    r = int(255 * i / width)
                    g = int(255 * j / height)
                    b = int(255 * (i + j) / (width + height))
                    pixels[i, j] = (r, g, b)
            
            img.save(path)
        
        image_paths[(width, height)] = path
    
    return image_paths

@pytest.fixture
def mock_model_output():
    """Generate a mock model output for testing."""
    # Create a simple circular mask
    size = 1024
    mask = np.zeros((1, 1, size, size), dtype=np.float32)
    center = size // 2
    radius = size // 4
    
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                mask[0, 0, i, j] = 1.0
    
    return mask

def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require model file"
    )

def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is specified."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)