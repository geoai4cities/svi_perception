#!/usr/bin/env python3
"""
Configuration file for 80-class object detection feature extraction
This file contains all configurable parameters for comprehensive COCO detection
"""

import os
from pathlib import Path
import torch

# =====================================================
# BASE PATHS - SET THESE FOR YOUR ENVIRONMENT
# =====================================================
# Option 1: Set environment variable FEATURE_EXTRACTOR_BASE
# Option 2: Modify the default path below
BASE_DIR = Path(os.environ.get('FEATURE_EXTRACTOR_BASE',
                               str(Path(__file__).parent.parent)))
OD_DIR = Path(__file__).parent

# =====================================================
# MODEL PATHS
# =====================================================
PATHS = {
    # DINOv3 repository
    'dinov3_repo': BASE_DIR / "dinov3_idd_finetuning/dinov3_github_repo/dinov3",
    
    # Model weights
    'detector_weights': BASE_DIR / "weights/dinov3/adapters/detection/dinov3_vit7b16_coco_detr_head-b0235ff7.pth",
    'backbone_weights': BASE_DIR / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth",
    
    # Data paths
    'pp_images_manifest': BASE_DIR / "data/manifests/pp_images.txt",
    'local_images_manifest': BASE_DIR / "data/manifests/local_images.txt",
    'pp_scores': BASE_DIR / "data/pp_scores.txt",
    'local_scores': BASE_DIR / "data/local_scores.txt",
    
    # Output directories
    'output_dir': OD_DIR / "output",
    'visualization_dir': OD_DIR / "visualization_output",
}

# =====================================================
# DETECTION CONFIGURATION - ALL 80 COCO CLASSES
# =====================================================

# All 80 COCO object classes (class_id: class_name)
# Excludes 'N/A' entries from the original 91-class list
TARGET_CLASSES = {
    1: "person",
    2: "bicycle", 
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic_light",
    11: "fire_hydrant",
    13: "stop_sign",
    14: "parking_meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports_ball",
    38: "kite",
    39: "baseball_bat",
    40: "baseball_glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis_racket",
    44: "bottle",
    46: "wine_glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot_dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted_plant",
    65: "bed",
    67: "dining_table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell_phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy_bear",
    89: "hair_drier",
    90: "toothbrush"
}

# Full COCO class names (91 classes including N/A)
COCO_CLASS_NAMES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# =====================================================
# PROCESSING PARAMETERS
# =====================================================

# Image processing - optimized for 140GB memory
IMAGE_SIZE = 768  # DINOv3 detection model input size (reduced for better batch processing)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections

# Batch processing - optimized for 140GB memory
TEST_MODE_MAX_IMAGES = 5  # Number of images to process in test mode
PROGRESS_SAVE_INTERVAL = 100  # Save progress every N images in full mode
BATCH_SIZE = 6  # Batch size for inference (increased due to smaller image size)

# =====================================================
# PERCEPTION SCORE CONFIGURATION
# =====================================================
# PERCEPTION_COLUMNS removed - only extracting detection features

# =====================================================
# DEVICE CONFIGURATION
# =====================================================
# Device selection: 'auto', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', etc.
DEVICE = 'auto'  # Using GPU 2 for training tasks

def get_device():
    """
    Get the configured device for computation
    
    Options:
    - 'auto': Automatically select best available device
    - 'cpu': Force CPU usage
    - 'cuda:0', 'cuda:1', 'cuda:2': Use specific GPU
    - 'cuda': Use default GPU (cuda:0)
    """
    device_setting = DEVICE.lower()
    
    if device_setting == 'auto':
        # Automatic selection - prefer GPU if available
        if torch.cuda.is_available():
            device = 'cuda:0'
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 AUTO: Using GPU {device} - {device_name} with {memory:.1f} GB memory")
        else:
            device = 'cpu'
            print("💻 AUTO: Using CPU (GPU not available)")
    
    elif device_setting == 'cpu':
        # Force CPU usage
        device = 'cpu'
        print("💻 Using CPU (forced)")
    
    elif device_setting.startswith('cuda'):
        # Use specific GPU
        if torch.cuda.is_available():
            # Parse device number if specified
            if ':' in device_setting:
                device_num = int(device_setting.split(':')[1])
            else:
                device_num = 0
            
            # Check if requested device exists
            if device_num < torch.cuda.device_count():
                device = f'cuda:{device_num}'
                device_name = torch.cuda.get_device_name(device_num)
                memory = torch.cuda.get_device_properties(device_num).total_memory / 1e9
                print(f"🚀 Using GPU {device} - {device_name} with {memory:.1f} GB memory")
            else:
                print(f"⚠️ Requested {device_setting} not available. Found {torch.cuda.device_count()} GPU(s)")
                print("💻 Falling back to CPU")
                device = 'cpu'
        else:
            print(f"⚠️ CUDA requested but not available")
            print("💻 Falling back to CPU")
            device = 'cpu'
    
    else:
        # Invalid setting - default to auto
        print(f"⚠️ Invalid device setting '{device_setting}'. Using auto mode.")
        return get_device()  # Recursive call with auto mode
    
    return device

# =====================================================
# VALIDATION
# =====================================================
def validate_config():
    """Validate that all required paths and files exist"""
    print("🔍 Validating 80-class detection configuration...")
    
    errors = []
    
    # Check critical paths
    critical_paths = [
        ('DINOv3 repository', PATHS['dinov3_repo']),
        ('Detector weights', PATHS['detector_weights']),
        ('Backbone weights', PATHS['backbone_weights']),
        ('PP images manifest', PATHS['pp_images_manifest']),
        ('Local images manifest', PATHS['local_images_manifest']),
        ('PP scores', PATHS['pp_scores']),
        ('Local scores', PATHS['local_scores']),
    ]
    
    for name, path in critical_paths:
        if not path.exists():
            errors.append(f"❌ {name} not found: {path}")
        else:
            print(f"   ✓ {name} found")
    
    # Create output directories if they don't exist
    PATHS['output_dir'].mkdir(parents=True, exist_ok=True)
    PATHS['visualization_dir'].mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directories created/verified")
    
    if errors:
        print("\n⚠️  Configuration errors found:")
        for error in errors:
            print(f"   {error}")
        return False
    
    print("✅ 80-class configuration validation passed!")
    return True

# =====================================================
# FEATURE NAMES
# =====================================================
def get_detection_feature_names():
    """Get list of detection feature column names for all 80 classes"""
    return [f'det_vit_{class_name}' for class_name in TARGET_CLASSES.values()]

# =====================================================
# DISPLAY CONFIGURATION
# =====================================================
def print_config():
    """Print current 80-class configuration"""
    print("\n" + "="*60)
    print("80-CLASS OBJECT DETECTION CONFIGURATION")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {PATHS['output_dir']}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Target classes: {len(TARGET_CLASSES)} classes (ALL COCO classes)")
    print(f"Test mode images: {TEST_MODE_MAX_IMAGES}")
    print(f"Progress save interval: {PROGRESS_SAVE_INTERVAL}")
    print("="*60)
    
    # Show first 10 and last 10 classes for brevity
    print("\nFirst 10 classes:")
    for i, (class_id, class_name) in enumerate(list(TARGET_CLASSES.items())[:10]):
        print(f"   {class_id:2d}: {class_name}")
    
    print("\n... (60 more classes) ...")
    
    print("\nLast 10 classes:")
    for i, (class_id, class_name) in enumerate(list(TARGET_CLASSES.items())[-10:]):
        print(f"   {class_id:2d}: {class_name}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test configuration when run directly
    print_config()
    validate_config()
