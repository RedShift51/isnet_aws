from setuptools import setup, find_packages

setup(
    name="isnet-production",
    version="0.1.0",
    description="Production-ready ISNet segmentation model",
    author="Kittl AI Engineer",
    author_email="alexey.buzovkin@gmail.com",
    python_requires=">=3.8, <4",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pillow>=9.0.0,<10.0.0",
        "onnxruntime>=1.11.0,<2.0.0",
        "fastapi>=0.87.0,<1.0.0",
        "uvicorn>=0.17.0,<1.0.0",
        "python-multipart>=0.0.5,<0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
        ],
        "deploy": [
            "boto3>=1.24.0",
            "sagemaker>=2.100.0",
            "azure-identity>=1.12.0",
            "azure-ai-ml>=1.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)