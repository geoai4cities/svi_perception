#!/bin/bash

# DINOv3 Feature Extraction Environment Setup
# For setup instructions, see: https://github.com/facebookresearch/dinov3

echo "========================================="
echo "DINOv3 Feature Extraction Environment"
echo "========================================="

# Environment name
ENV_NAME="dinov3_env"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    exit 1
fi

# Clean removal of old environment
echo "Cleaning up old environment..."
conda deactivate 2>/dev/null || true
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

# Create fresh environment with Python 3.10
echo ""
echo "Creating Python 3.10 environment..."
conda create -n ${ENV_NAME} python=3.10 -y

# Initialize and activate
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Verify we're in the right environment
echo ""
echo "Environment check:"
which python
python --version

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
echo ""
echo "Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Core packages
echo ""
echo "Installing core packages..."
pip install numpy==1.26.4
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0
pip install scipy>=1.10.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# ML packages
echo "Installing ML packages..."
pip install scikit-learn>=1.3.0
pip install xgboost>=2.0.0

# DL utilities
echo "Installing deep learning utilities..."
pip install timm>=0.9.0
pip install transformers>=4.35.0
pip install accelerate>=0.24.0

# Other utilities
echo "Installing utilities..."
pip install tqdm>=4.65.0
pip install h5py>=3.9.0
pip install openpyxl>=3.1.0
pip install omegaconf>=2.3.0

# Segmentation
echo "Installing segmentation packages..."
pip install pycocotools>=2.0.7
pip install albumentations>=1.3.0

# Verify installation
echo ""
echo "========================================="
echo "Verification:"
echo "========================================="
python -c "
import sys
import torch
import numpy as np
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test imports
try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except:
    print('OpenCV import failed')

try:
    import timm
    print(f'timm: {timm.__version__}')
except:
    print('timm import failed')
"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "For DINOv3 setup, follow:"
echo "  https://github.com/facebookresearch/dinov3"
echo "========================================="
