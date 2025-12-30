#!/usr/bin/env python3
"""
Configuration file for segmentation analysis
Modify this file to adapt to different machines/environments
"""

import os
from pathlib import Path
import torch

# =============================================================================
# MACHINE-SPECIFIC CONFIGURATION
# =============================================================================

# Base directory configuration - SET THIS FOR YOUR ENVIRONMENT
# Option 1: Set environment variable FEATURE_EXTRACTOR_BASE
# Option 2: Modify DEFAULT_BASE_DIR below
DEFAULT_BASE_DIR = os.environ.get('FEATURE_EXTRACTOR_BASE',
                                   str(Path(__file__).parent.parent))

# Allow override via environment variable
BASE_DIR = Path(DEFAULT_BASE_DIR)

# =============================================================================
# DEVICE CONFIGURATION  
# =============================================================================

# Device selection - CHANGE THIS FOR GPU/CPU PREFERENCE
# Options: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', etc.
DEVICE_PREFERENCE = os.environ.get('DEVICE_PREFERENCE', 'cuda')

def get_device():
    """Get the best available device based on preference and availability"""
    if DEVICE_PREFERENCE == 'cpu':
        return 'cpu'
    elif DEVICE_PREFERENCE == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif DEVICE_PREFERENCE.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"⚠️  CUDA not available, falling back to CPU")
            return 'cpu'
        
        # Handle specific GPU index (e.g., 'cuda:2')
        if ':' in DEVICE_PREFERENCE:
            device_parts = DEVICE_PREFERENCE.split(':')
            if len(device_parts) == 2:
                try:
                    gpu_id = int(device_parts[1])
                    if gpu_id >= torch.cuda.device_count():
                        print(f"⚠️  GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found), using cuda:0")
                        return 'cuda:0'
                    else:
                        print(f"✅ Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                        return DEVICE_PREFERENCE
                except ValueError:
                    print(f"⚠️  Invalid GPU ID in '{DEVICE_PREFERENCE}', using cuda:0")
                    return 'cuda:0'
        else:
            # Just 'cuda' - use default GPU
            return 'cuda'
    else:
        print(f"⚠️  Unknown device preference '{DEVICE_PREFERENCE}', falling back to auto")
        return 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# All paths relative to BASE_DIR
PATHS = {
    # DINOv3 repository
    'dinov3_repo': BASE_DIR / "dinov3_idd_finetuning/dinov3_github_repo/dinov3",
    
    # Model weights
    'segmentor_weights': BASE_DIR / "weights/dinov3/adapters/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    'backbone_weights': BASE_DIR / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth",
    'finetuned_checkpoint': BASE_DIR / "dinov3_idd_finetuning/clean_finetuning_setup/checkpoints/train_30epoch_full_20250824_143906/best_model.pth",
    
    # Data files
    'pp_scores': BASE_DIR / "data/pp_scores.txt",
    'local_scores': BASE_DIR / "data/local_scores.txt",
    'pp_images_manifest': BASE_DIR / "data/manifests/pp_images.txt",
    'local_images_manifest': BASE_DIR / "data/manifests/local_images.txt",
    
    # Configuration files
    'idd_config': BASE_DIR / "dinov3_idd_finetuning/clean_finetuning_setup",
    'hybrid_config': BASE_DIR / "dinov3_idd_finetuning/finetuned_image_testing",
    'segmentation_analysis': BASE_DIR / "segmentation_analysis",
}

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Image processing settings
IMAGE_SIZE = 512
BATCH_SIZE = 1  # Keep at 1 for memory efficiency with hybrid approach

# Progress saving interval (for large datasets)
PROGRESS_SAVE_INTERVAL = 100

# Test mode settings
TEST_MODE_MAX_IMAGES = 5

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Hybrid segmentation mapping (19 ADE20K + 7 IDD classes)
ADE20K_TO_IDD_MAPPING = {
    7: 0,    # road -> road
    12: 2,   # sidewalk -> sidewalk  
    92: 3,   # dirt;track -> rail track
    13: 4,   # person -> person
    117: 6,  # minibike;motorbike -> motorcycle
    128: 7,  # bicycle -> bicycle
    21: 9,   # car -> car
    84: 10,  # truck -> truck
    81: 11,  # bus -> bus
    1: 14,   # wall -> wall
    33: 15,  # fence -> fence
    39: 16,  # railing;rail -> guard rail
    101: 17, # poster;placard -> billboard
    44: 17,  # signboard;sign -> billboard (UPDATED)
    137: 19, # traffic light -> traffic light
    94: 20,  # pole -> pole
    2: 22,   # building -> building
    62: 23,  # bridge -> bridge
    5: 24,   # tree -> vegetation
    3: 25,   # sky -> sky
}

# IDD-only classes (now includes traffic sign)
IDD_ONLY_CLASSES = {1, 5, 8, 12, 13, 18, 21}

# Perception score columns
PERCEPTION_COLUMNS = ["beautiful", "lively", "boring", "safe"]

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that all required paths exist"""
    print("🔍 Validating configuration...")
    
    missing_paths = []
    for name, path in PATHS.items():
        if not path.exists():
            missing_paths.append((name, path))
    
    if missing_paths:
        print("❌ Missing required paths:")
        for name, path in missing_paths:
            print(f"   {name}: {path}")
        return False
    
    device = get_device()
    print(f"✅ Configuration valid")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Selected device: {device}")
    
    if torch.cuda.is_available():
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            is_selected = (device == f"cuda:{i}") or (device == "cuda" and i == 0)
            marker = " <- SELECTED" if is_selected else ""
            print(f"     GPU {i}: {gpu_name} ({gpu_memory:.1f} GB){marker}")
    else:
        print(f"   No GPUs available, using CPU")
    
    return True

# =============================================================================
# USAGE EXAMPLES FOR DIFFERENT MACHINES
# =============================================================================

"""
# Example 1: Different base directory
export FEATURE_EXTRACTOR_BASE="/home/user/projects/perception/Feature_Extractor"

# Example 2: Force CPU usage  
export DEVICE_PREFERENCE="cpu"

# Example 3: Use default GPU (cuda:0)
export DEVICE_PREFERENCE="cuda"

# Example 4: Use specific GPU (GPU 2)
export DEVICE_PREFERENCE="cuda:2"

# Example 5: Use in Python script
from config import PATHS, get_device, BASE_DIR
device = get_device()
model_path = PATHS['segmentor_weights']

# Example 6: Set specific GPU in code
import os
os.environ['DEVICE_PREFERENCE'] = 'cuda:2'
"""
