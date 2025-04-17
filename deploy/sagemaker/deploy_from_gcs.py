import argparse
import boto3
import sagemaker
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy ISNet model from Google Cloud Storage to SageMaker')
    parser.add_argument('--gcs-uri', type=str, 
                        default='https://storage.cloud.google.com/abuz_public/model.tar.gz',
                        help='Public Google Cloud Storage URL for model.tar.gz')
    parser.add_argument('--role', type=str,
                        help='SageMaker execution role ARN (if not provided, gets default role)')
    parser.add_argument('--instance-type', type=str, default='ml.c5.large',
                        help='SageMaker instance type for the endpoint (use ml.g4dn.xlarge for GPU)')
    parser.add_argument('--endpoint-name', type=str,
                        default=f'isnet-endpoint-{datetime.now().strftime("%Y%m%d%H%M%S")}',
                        help='Name for the SageMaker endpoint')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region')
    parser.add_argument('--update-if-exists', action='store_true',
                        help='Update endpoint if it already exists')
    return parser.parse_args()

def deploy_model_from_gcs():
    args = parse_args()
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Get role ARN if not provided
    role = args.role if args.role else sagemaker.get_execution_role()
    
    # Determine the correct image URI for the instance type
    is_gpu_instance = 'g' in args.instance_type or 'p' in args.instance_type
    framework_version = '1.12.1'
    py_version = 'py38'
    processor = 'gpu' if is_gpu_instance else 'cpu'
    
    image_uri = f'763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-inference:{framework_version}-{processor}-{py_version}'
    print(f"Using image URI: {image_uri}")
    
    # Environment variables to pass to the model container
    environment = {
        'SAGEMAKER_PROGRAM': 'inference.py',
        'MODEL_PATH': '/opt/ml/model/isnet_converted/isnet_converted1.onnx',
        'LOG_LEVEL': 'INFO'
    }
    
    # Create SageMaker model
    print(f"Creating model from GCS URL: {args.gcs_uri}")
    model = sagemaker.model.Model(
        model_data=args.gcs_uri,  # Direct use of GCS URL
        role=role,
        image_uri=image_uri,
        env=environment
    )
    
    # Deploy model to endpoint
    endpoint_name = args.endpoint_name
    client = boto3.client('sagemaker', region_name=args.region)
    
    update_endpoint = False
    if args.update_if_exists:
        try:
            client.describe_endpoint(EndpointName=endpoint_name)
            print(f"Endpoint {endpoint_name} exists, updating...")
            update_endpoint = True
        except:
            print(f"Creating new endpoint {endpoint_name}...")
    
    # Deploy the model
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
        update_endpoint=update_endpoint
    )
    
    print(f"Deployment initiated for endpoint: {endpoint_name}")
    print(f"Once ready, you can invoke with: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} --content-type 'application/x-image' --body fileb://image.jpg output.png")

if __name__ == "__main__":
    deploy_model_from_gcs()