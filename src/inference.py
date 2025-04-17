import os
import json
import numpy as np
from PIL import Image
import io
import base64
import logging
import onnxruntime as ort
from typing import Dict, List, Union, BinaryIO, Any

# Configure logging
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))

# Set model path from environment variable or use default
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/isnet_converted1.onnx")
ISNET_DIR = os.environ.get("ISNET_DIR", "/opt/ml/model/isnet_converted")

# Initialize onnx runtime session - will be lazy loaded
session = None

def load_model():
    """Load the ONNX model"""
    global session
    if session is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            # Configure session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # Adjust based on instance type
            
            # Use CUDA provider if available, otherwise use CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(MODEL_PATH, sess_options, providers=providers)
            
            # Get input and output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            logger.info(f"Model loaded successfully. Input name: {input_name}, Output name: {output_name}")
            
            return session
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def normalize(img, mean=0.5, std=1):
    return (img - mean) / std

def preprocess_image(img):
    """Preprocess the input image for the model"""
    # Resize the image to the model's input size
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    
    # Resize to model input dimensions
    img = img.convert("RGB")
    img = img.resize((1024, 1024), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_np = np.array(img).astype(np.float32)
    
    # Normalize to 0-1
    img_np = img_np / 255.0
    
    # Transpose from HWC to CHW format for PyTorch/ONNX
    img_np = img_np.transpose(2, 0, 1)
    
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np

def run_inference(input_data):
    """Run inference with the ONNX model"""
    global session
    if session is None:
        session = load_model()
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    result = session.run([output_name], {input_name: input_data})
    return result[0]

def postprocess_output(output):
    """Process the model output to a displayable mask"""
    # Squeeze batch and channel dimensions
    mask = np.squeeze(output)
    
    # Scale to 0-255 range
    mask = mask * 255.0
    
    # Clip values to valid range
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    return mask

def create_response(mask, format='png', encode_base64=False):
    """Create a response with the segmentation mask"""
    # Convert mask to image
    mask_img = Image.fromarray(mask)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    mask_img.save(img_byte_arr, format=format)
    img_bytes = img_byte_arr.getvalue()
    
    # Encode to base64 if requested
    if encode_base64:
        return base64.b64encode(img_bytes).decode('utf-8')
    else:
        return img_bytes

def predict(data, content_type="application/x-image", accept="application/x-image"):
    """Main prediction function that handles different input types"""
    try:
        # Process input based on content type
        if content_type == "application/x-image" or content_type.startswith("image/"):
            # Input is a raw image
            if isinstance(data, bytes):
                img_bytes = data
            else:
                img_bytes = data.read()
            
            # Preprocess the image
            preprocessed = preprocess_image(img_bytes)
            
        elif content_type == "application/json":
            # Handle JSON input with base64 encoded image
            json_data = json.loads(data.decode() if isinstance(data, bytes) else data)
            if "image" in json_data:
                img_bytes = base64.b64decode(json_data["image"])
                preprocessed = preprocess_image(img_bytes)
            else:
                raise ValueError("JSON input must contain 'image' field with base64 encoded image")
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Run inference
        logger.info("Running inference")
        output = run_inference(preprocessed)
        
        # Post-process the output
        mask = postprocess_output(output)
        
        # Return response based on accept type
        if accept == "application/json":
            return {
                "mask": base64.b64encode(create_response(mask)).decode('utf-8'),
                "mask_size": mask.shape
            }
        else:  # Default to returning the image
            return create_response(mask)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Entry point for SageMaker
def model_fn(model_dir):
    """Load the model for SageMaker"""
    global MODEL_PATH, ISNET_DIR
    
    # Update model path based on model_dir
    if os.path.exists(os.path.join(model_dir, "model.onnx")):
        MODEL_PATH = os.path.join(model_dir, "model.onnx")
    elif os.path.exists(os.path.join(model_dir, "isnet_converted", "model.onnx")):
        MODEL_PATH = os.path.join(model_dir, "isnet_converted", "model.onnx")
    else:
        files = os.listdir(model_dir)
        onnx_files = [f for f in files if f.endswith('.onnx')]
        if onnx_files:
            MODEL_PATH = os.path.join(model_dir, onnx_files[0])
        else:
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")
    
    # Update ISNET_DIR if it exists
    if os.path.exists(os.path.join(model_dir, "isnet_converted")):
        ISNET_DIR = os.path.join(model_dir, "isnet_converted")
    
    logger.info(f"Model path set to: {MODEL_PATH}")
    logger.info(f"ISNet directory set to: {ISNET_DIR}")
    
    # Load the model
    return load_model()

def input_fn(request_body, request_content_type):
    """Handle input data for SageMaker"""
    return request_body, request_content_type

def predict_fn(input_data, model):
    """Prediction function for SageMaker"""
    data, content_type = input_data
    return predict(data, content_type)

def output_fn(prediction, accept):
    """Format output for SageMaker"""
    if accept == "application/json":
        if isinstance(prediction, dict):
            return json.dumps(prediction), "application/json"
        else:
            return json.dumps({"mask": base64.b64encode(prediction).decode('utf-8')}), "application/json"
    elif accept.startswith("image/"):
        return prediction, accept
    else:
        # Default to PNG image
        return prediction, "image/png"

# For local testing
if __name__ == "__main__":
    # Test with a sample image
    test_img_path = "tests/test_data/sample.jpg"
    if os.path.exists(test_img_path):
        with open(test_img_path, "rb") as f:
            test_img = f.read()
        
        result = predict(test_img)
        
        # Save the result
        with open("output.png", "wb") as f:
            f.write(result)
        
        print("Inference completed and saved to output.png")
    else:
        print(f"Test image not found at {test_img_path}")