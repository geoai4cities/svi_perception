#!/usr/bin/env python3
"""
Purpose: Extract all 150 ADE20K classes segmentation area ratios as features
Arguments: None (processes images from manifests)
Returns: CSV file with Image_ID and 151 segmentation area ratios (150 classes + background)
Note: Uses only ADE20K pretrained model, extracts all 150 classes + background without mapping

IMPORTANT: This script uses the same +1 correction as the 26-class version:
- Model outputs 0-149, but ADE20K classes are 1-150 (0 is background)
- See extract_segmentation_features_ade20k_only.py for the 26-class mapping reference

Environment Setup:
    source dinov3/bin/activate
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from functools import partial

# Import configuration
from config import (
    PATHS, get_device, BASE_DIR, IMAGE_SIZE, 
    PROGRESS_SAVE_INTERVAL, TEST_MODE_MAX_IMAGES, validate_config
)

# Add paths for ADE20K segmentation (using config)
sys.path.append(str(PATHS['dinov3_repo']))

from dinov3.eval.segmentation.inference import make_inference

# Import ADE20K class definitions
sys.path.append(str(PATHS['segmentation_analysis'] / "full_ade20k_150_features_extraction"))
from ADE20K_colormaps import ADE20K_CLASS_NAMES

# Get paths from config
SEGMENTOR_WEIGHTS = PATHS['segmentor_weights']
BACKBONE_WEIGHTS = PATHS['backbone_weights']
DINOV3_REPO = str(PATHS['dinov3_repo'])

# No perception scores needed for 150-class extraction

def load_ade20k_model(device='cpu'):
    """
    Purpose: Load the ADE20K pretrained model
    Arguments: device (str) - device to load model on ('cpu' or 'cuda')
    Returns: Loaded ADE20K segmentation model
    """
    print(f"🔍 Loading ADE20K pretrained model on {device.upper()}...")
    
    try:
        # Load model on CPU first, then move to device if needed
        segmentor = torch.hub.load(
            DINOV3_REPO, 
            'dinov3_vit7b16_ms', 
            source="local",
            weights=str(SEGMENTOR_WEIGHTS),
            backbone_weights=str(BACKBONE_WEIGHTS)
        )
        
        # Move to specified device. Force float32 to avoid dtype mismatches.
        if device != 'cpu':
            try:
                segmentor = segmentor.to(device=device, dtype=torch.float32)
                print(f"   ✓ Model moved to {device} (float32)")
            except torch.cuda.OutOfMemoryError as e:
                print(f"   ⚠️  CUDA out of memory on {device}, falling back to CPU")
                device = 'cpu'
                segmentor = segmentor.to(device=device, dtype=torch.float32)
                print(f"   ✓ Model moved to {device} (float32)")
        
        segmentor.eval()
        return segmentor
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        if device != 'cpu':
            print(f"   🔄 Retrying with CPU...")
            return load_ade20k_model('cpu')
        else:
            raise e

def make_transform(resize_size: int = 512):
    """
    Purpose: Create image transformation pipeline for DINOv3 input
    Arguments: resize_size (int) - size to resize images to
    Returns: torchvision transform pipeline
    """
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def process_ade20k_output(segmentor, batch_img, img_size, target_size):
    """
    Purpose: Process ADE20K model output to get segmentation map
    Arguments: 
        segmentor - ADE20K model
        batch_img - input image tensor
        img_size - input image size
        target_size - target output size
    Returns: Segmentation map as numpy array
    """
    segmentation_map = make_inference(
        batch_img,
        segmentor,
        inference_mode="slide",
        decoder_head_type="m2f",
        rescale_to=target_size,
        n_output_channels=150,
        crop_size=(img_size, img_size),
        stride=(img_size, img_size),
        output_activation=partial(torch.nn.functional.softmax, dim=1),
    ).argmax(dim=1, keepdim=True)
    
    # Extract segmentation map without +1 correction
    # The +1 correction will be applied in calculate_ade20k_area_ratios when getting class names
    # This matches the official testing approach
    seg_map = segmentation_map[0, 0].cpu().numpy()
    
    return seg_map

def calculate_ade20k_area_ratios(segmentation_map, num_classes=151):
    """
    Purpose: Calculate area percentage for each ADE20K class (0-100 scale)
    Arguments: 
        segmentation_map - ADE20K segmentation map
        num_classes - number of classes to process (151 for ADE20K: 0 background + 150 classes)
    Returns: Dictionary of area ratios for each class with 'ade20k_' prefix
    """
    total_pixels = segmentation_map.size
    area_ratios = {}
    
    # Process all possible classes (0-150) to ensure we get all 151 features
    for class_id in range(num_classes):
        # Apply +1 correction for ADE20K class mapping (matches official testing approach)
        # Model outputs 0-149, but ADE20K classes are 1-150 (0 is background)
        # Model class 0 -> ADE20K class 1, Model class 1 -> ADE20K class 2, etc.
        corrected_id = class_id + 1
        
        # Apply +1 correction to get correct ADE20K class name
        if corrected_id < len(ADE20K_CLASS_NAMES):
            class_name = ADE20K_CLASS_NAMES[corrected_id]
        else:
            class_name = f"unknown_class_{class_id}"
        
        class_pixels = (segmentation_map == class_id).sum()
        area_percentage = float(class_pixels / total_pixels * 100)  # Convert to percentage
        area_ratios[f'ade20k_{class_name}'] = area_percentage
    
    return area_ratios

def load_and_transform_image(image_path, device='cpu'):
    """
    Purpose: Load and transform image for segmentation
    Arguments: 
        image_path - path to image file
        device - device to load tensor on
    Returns: Transformed image tensor, original size, and image size
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Transform for DINOv3 (reduce size to lower memory pressure)
        img_size = min(IMAGE_SIZE, 384)
        transform = make_transform(img_size)
        
        # Apply transform and move to device; force float32 to match model dtype
        batch_img = transform(image)[None]
        if device != 'cpu':
            batch_img = batch_img.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            batch_img = batch_img.to(dtype=torch.float32)
        
        # Clean up PIL image to free memory
        original_size = image.size[::-1]  # (height, width) for processing
        del image
        gc.collect()
        
        return batch_img, original_size, img_size
        
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None, None, None

def analyze_single_image(ade20k_model, image_path, device='cpu'):
    """
    Purpose: Analyze a single image and return ADE20K segmentation area ratios
    Arguments: 
        ade20k_model - ADE20K segmentation model
        image_path - path to image file
        device - device to run inference on
    Returns: Dictionary with image_id and ADE20K segmentation area ratios
    """
    try:
        # Load and transform image
        batch_img, original_size, img_size = load_and_transform_image(image_path, device)
        if batch_img is None:
            return None
        
        # Run ADE20K model with per-image fallback to CPU on OOM/dtype errors
        try:
            # Enforce float32 and disable autocast to avoid dtype mismatches
            if device != 'cpu':
                batch_img = batch_img.to(dtype=torch.float32)
            with torch.inference_mode():
                if isinstance(batch_img, torch.Tensor) and batch_img.is_cuda:
                    with torch.autocast(device_type='cuda', enabled=False):
                        ade20k_seg = process_ade20k_output(ade20k_model, batch_img, img_size, original_size)
                else:
                    ade20k_seg = process_ade20k_output(ade20k_model, batch_img, img_size, original_size)
        except Exception as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cudabfloat16' in msg or 'expected weight to have type' in msg:
                try:
                    print(f"   ⚠️  GPU error for {Path(image_path).name}: {e}. Retrying on CPU...")
                    # Move to CPU float32 and retry
                    ade20k_model_cpu = ade20k_model.to(device='cpu', dtype=torch.float32)
                    batch_cpu = batch_img.to(device='cpu', dtype=torch.float32)
                    with torch.inference_mode():
                        ade20k_seg = process_ade20k_output(ade20k_model_cpu, batch_cpu, img_size, original_size)
                except Exception as e2:
                    print(f"❌ CPU fallback failed for {Path(image_path).name}: {e2}")
                    return None
            else:
                print(f"❌ Error analyzing image {image_path}: {e}")
                return None
        
        # Extract image ID from full path
        image_id = Path(image_path).name
        
        # Calculate area ratios for all 150 ADE20K classes (+ background key)
        area_ratios = calculate_ade20k_area_ratios(ade20k_seg)
        
        # Create result dictionary
        result = {
            'image_id': image_id
        }
        
        # Add segmentation area ratios with keys as in original schema (ade20k_*)
        for key, value in area_ratios.items():
            result[key] = value
        
        # Clean up tensors to free memory
        del batch_img, ade20k_seg
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing image {image_path}: {e}")
        gc.collect()
        return None


def load_image_manifests():
    """
    Purpose: Load image paths from manifests
    Arguments: None
    Returns: Tuple of (place_pulse_images, local_images) lists
    """
    print("📁 Loading image manifests...")
    
    base_path = BASE_DIR
    
    # Load place pulse images
    with open(base_path / "data/manifests/pp_images.txt", 'r') as f:
        pp_images = [line.strip() for line in f if line.strip()]
    
    # Load local images
    with open(base_path / "data/manifests/local_images.txt", 'r') as f:
        local_images = [line.strip() for line in f if line.strip()]
    
    print(f"   Place pulse images: {len(pp_images)}")
    print(f"   Local images: {len(local_images)}")
    print(f"   Total images: {len(pp_images) + len(local_images)}")
    
    return pp_images, local_images


def save_progress(results, output_dir, test_mode, processed_count, total_count):
    """
    Purpose: Save progress to avoid data loss
    Arguments: 
        results - list of processed results
        output_dir - directory to save progress
        test_mode - whether in test mode
        processed_count - number of images processed
        total_count - total number of images
    Returns: None
    """
    progress_file = output_dir / "progress_ade20k_150.json"
    progress_data = {
        'processed_count': processed_count,
        'total_count': total_count,
        'test_mode': test_mode,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results_count': len(results)
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(output_dir):
    """
    Purpose: Load previous progress to resume processing
    Arguments: 
        output_dir - directory containing progress file
    Returns: Tuple of (results, processed_count) or (None, 0) if no progress found
    """
    progress_file = output_dir / "progress_ade20k_150.json"
    
    if not progress_file.exists():
        return None, 0
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        print(f"📂 Found previous progress: {progress_data['processed_count']}/{progress_data['total_count']} images processed")
        print(f"   Timestamp: {progress_data['timestamp']}")
        print(f"   Results saved: {progress_data['results_count']}")
        
        # Load the partial results if they exist
        results_file = output_dir / "partial_results_ade20k_150.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"   Loaded {len(results)} previous results")
            return results, progress_data['processed_count']
        else:
            print("   ⚠️  No partial results file found, starting fresh")
            return None, 0
            
    except Exception as e:
        print(f"❌ Error loading progress: {e}")
        return None, 0

def save_partial_results(results, output_dir):
    """
    Purpose: Save partial results to enable resume
    Arguments: 
        results - list of processed results
        output_dir - directory to save results
    Returns: None
    """
    results_file = output_dir / "partial_results_ade20k_150.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

def clear_progress(output_dir):
    """
    Purpose: Clear previous progress to start fresh
    Arguments: 
        output_dir - directory containing progress files
    Returns: None
    """
    progress_file = output_dir / "progress_ade20k_150.json"
    partial_results_file = output_dir / "partial_results_ade20k_150.json"
    
    if progress_file.exists():
        progress_file.unlink()
        print("🧹 Cleared progress file")
    
    if partial_results_file.exists():
        partial_results_file.unlink()
        print("🧹 Cleared partial results file")

def run_feature_extraction(test_mode=True, max_images=TEST_MODE_MAX_IMAGES):
    """
    Purpose: Run ADE20K 150-class segmentation feature extraction on images
    Arguments: 
        test_mode - whether to run in test mode with limited images
        max_images - maximum number of images to process in test mode
    Returns: pandas DataFrame with extracted features (151 columns including background), or None on failure
    
    Resume Functionality:
        - Automatically saves progress every 100 images (full mode only)
        - Can resume from where it left off if interrupted
        - Progress files: progress_ade20k_150.json, partial_results_ade20k_150.json
        - User can choose to continue or start fresh when resuming
    """
    import sys
    print("🚀 Starting ADE20K 150-Class Segmentation Feature Extraction")
    print("=" * 70)
    
    # Get device from config
    device = get_device()
    
    # Load ADE20K model only
    print("Loading ADE20K segmentation model...")
    ade20k_model = load_ade20k_model(device)
    
    print(f"🔧 Using {device.upper()} for ADE20K segmentation")
    print(f"📝 Extracting all 150 ADE20K classes (including background)")
    
    # Load data
    pp_images, local_images = load_image_manifests()
    
    # Combine all images
    all_images = pp_images + local_images
    
    if test_mode:
        all_images = all_images[:max_images]
        print(f"🧪 TEST MODE: Processing only {len(all_images)} images")
    
    # Create output directory in segmentation_analysis folder
    output_dir = Path(__file__).parent  # This will be the segmentation_analysis directory
    output_dir.mkdir(exist_ok=True)
    
    # Try to load previous progress for resume functionality
    results, start_index = load_progress(output_dir)
    if results is None:
        results = []
        start_index = 0
        print(f"\n🔍 Starting fresh extraction from {len(all_images)} images...")
    else:
        print(f"\n🔄 Resuming extraction from image {start_index + 1}/{len(all_images)}...")
        # Ask user if they want to continue or start fresh (only in interactive mode)
        if not test_mode and sys.stdin.isatty():
            response = input("Continue with previous progress? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                clear_progress(output_dir)
                results = []
                start_index = 0
                print("🔄 Starting fresh extraction...")
        else:
            # Non-interactive mode - automatically continue with previous progress
            print("🔄 Automatically continuing with previous progress...")
    
    # Process images with progress saving
    for i, image_path in enumerate(tqdm(all_images[start_index:], desc="Processing images", initial=start_index, total=len(all_images))):
        try:
            result = analyze_single_image(ade20k_model, image_path, device)
            if result:
                results.append(result)
                
                # Print progress for first few images
                if i < 5:
                    # Calculate total area coverage (should be close to 100%)
                    total_area = sum(result[f'ade20k_{class_name}'] for class_name in ADE20K_CLASS_NAMES[1:])  # Skip background
                    print(f"   Image {i+1}: {result['image_id']} - Total area coverage: {total_area:.2f}%")
            
            # Save progress every 100 images for big data runs
            if not test_mode and (i + 1) % 100 == 0:
                save_progress(results, output_dir, test_mode, i + 1, len(all_images))
                save_partial_results(results, output_dir)
                print(f"💾 Progress saved: {i + 1}/{len(all_images)} images processed")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    if not results:
        print("❌ No results generated")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if test_mode:
        filename = "test_ade20k_150_features.csv"
    else:
        # Check if we're processing 200 images (missing batch)
        if len(results_df) == 200:
            filename = "missing_200_ade20k_150_features.csv"
        else:
            filename = "full_ade20k_150_features.csv"
    
    output_path = output_dir / filename
    results_df.to_csv(output_path, index=False)
    print(f"💾 Features saved to: {output_path}")
    
    # Save final progress and clean up partial results
    if not test_mode:
        save_progress(results, output_dir, test_mode, len(all_images), len(all_images))
        # Clean up partial results file since we have the final CSV
        partial_results_file = output_dir / "partial_results_ade20k_150.json"
        if partial_results_file.exists():
            partial_results_file.unlink()
            print("🧹 Cleaned up partial results file")
    
    # Print summary
    print(f"\n📊 ADE20K 150-Class Segmentation Feature Extraction Summary:")
    print(f"   Total images processed: {len(results_df)}")
    print(f"   Segmentation columns: {len([col for col in results_df.columns if col.startswith('ade20k_')])} (150 classes)")
    print(f"   Classes extracted: All 150 ADE20K classes (1-150) + background (0) = 151 total features")
    
    # Show sample results for first few classes
    print(f"\n📋 Sample Results (first 10 ADE20K features):")
    feature_cols = ['image_id'] + [col for col in results_df.columns if col.startswith('ade20k_')][:10]
    sample_df = results_df[feature_cols].head()
    # Format the display to show percentages with 2 decimal places
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(sample_df.to_string())
    pd.reset_option('display.float_format')
    
    # Clean up model memory before returning (avoid deleting results_df before return)
    del ade20k_model
    gc.collect()
    return results_df

def main():
    """
    Purpose: Main function to run ADE20K 150-class segmentation feature extraction
    Arguments: None
    Returns: None
    """
    print("🚀 Starting ADE20K 150-Class Segmentation Feature Extraction")
    print("=" * 70)
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed!")
        return
    
    # Check if running in non-interactive mode (e.g., from shell script)
    import sys
    is_interactive = sys.stdin.isatty()
    
    if is_interactive:
        # Interactive mode - run test first, then ask user
        print(f"🧪 Running test extraction on {TEST_MODE_MAX_IMAGES} images...")
        test_results = run_feature_extraction(test_mode=True, max_images=TEST_MODE_MAX_IMAGES)
        
        if test_results is not None:
            print("\n🎉 Test extraction completed!")
            
            # Ask user if they want to run full extraction
            print(f"\n" + "="*70)
            response = input("Do you want to run the full extraction on all images? (y/n): ").lower().strip()
            
            if response in ['y', 'yes']:
                print("🚀 Running full extraction...")
                print("⚠️  This will process ~111,000 images and may take several hours.")
                print("💾 Progress will be saved every 100 images to prevent data loss.")
                print("📝 Extracting all 150 ADE20K classes (including background).")
                
                confirm = input("Are you sure? (y/n): ").lower().strip()
                if confirm in ['y', 'yes']:
                    full_results = run_feature_extraction(test_mode=False)
                    print("🎉 Full extraction completed!")
                else:
                    print("✅ Full extraction cancelled.")
            else:
                print("✅ Test extraction completed. Full extraction skipped.")
        else:
            print("❌ Test extraction failed!")
    else:
        # Non-interactive mode - run full extraction directly
        print("🚀 Running full extraction in non-interactive mode...")
        print("⚠️  This will process ~111,000 images and may take several hours.")
        print("💾 Progress will be saved every 100 images to prevent data loss.")
        print("📝 Extracting all 150 ADE20K classes (including background).")
        
        full_results = run_feature_extraction(test_mode=False)
        if full_results is not None:
            print("🎉 Full extraction completed!")
        else:
            print("❌ Full extraction failed!")

if __name__ == "__main__":
    main()
