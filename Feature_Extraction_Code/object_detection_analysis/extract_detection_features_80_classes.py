#!/usr/bin/env python3
"""
Purpose: Extract all 80 COCO object detection counts as features
Arguments: None (processes images from manifests defined in config)
Returns: CSV file with Image_ID and 80 detection counts
Version: 3.0 - 80-class implementation with auto-save and resume capability
"""

import sys
import time
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Import 80-class configuration
from config_80_classes import (
    PATHS, get_device, BASE_DIR, OD_DIR,
    TARGET_CLASSES, COCO_CLASS_NAMES,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD,
    TEST_MODE_MAX_IMAGES, PROGRESS_SAVE_INTERVAL,
    validate_config, print_config, get_detection_feature_names
)

# Add DINOv3 repository to path
sys.path.append(str(PATHS['dinov3_repo']))

# Constants for auto-save and resume
AUTO_SAVE_INTERVAL = 100  # Save dataframe every 100 images
PROGRESS_FILE = "progress_80_classes.json"
FEATURES_FILE = "detection_features_80_classes.csv"
TEMP_FEATURES_FILE = "temp_detection_features_80_classes.csv"

def load_detector(device='cpu'):
    """
    Purpose: Load DINOv3 detection model using config paths
    Arguments: device (str) - device to load model on ('cpu' or 'cuda')
    Returns: Loaded DINOv3 detection model
    """
    print("🔍 Loading DINOv3 detection model...")
    
    try:
        detector = torch.hub.load(
            str(PATHS['dinov3_repo']), 
            'dinov3_vit7b16_de', 
            source="local", 
            weights=str(PATHS['detector_weights']), 
            backbone_weights=str(PATHS['backbone_weights'])
        )
        
        # Move to device
        if device != 'cpu':
            detector = detector.to(device)
            print(f"   ✓ Model moved to {device}")
        
        detector.eval()
        print("   ✓ DINOv3 detector loaded successfully")
        return detector
        
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return None

def make_transform(img_size=IMAGE_SIZE):
    """
    Purpose: Create image transformation pipeline for DINOv3 detection
    Arguments: img_size (int) - target size for image resizing
    Returns: torchvision transform pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_and_transform_image(image_path, device='cpu'):
    """
    Purpose: Load and transform image for DINOv3 detection
    Arguments: 
        image_path - path to image file
        device - device to load tensor on
    Returns: Transformed image tensor
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Transform for DINOv3
        transform = make_transform(IMAGE_SIZE)
        batch_img = transform(image).unsqueeze(0)
        
        # Move to device
        if device != 'cpu':
            batch_img = batch_img.to(device)
        
        # Clean up PIL image to free memory
        del image
        gc.collect()
        
        return batch_img
        
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

def count_objects(detections, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Purpose: Count objects for all 80 target classes with configurable threshold
    Arguments: 
        detections - DINOv3 detection output
        confidence_threshold - minimum confidence for detections
    Returns: Dictionary of detection counts for all 80 classes
    """
    counts = {}
    
    # Initialize counts for all 80 target classes
    for class_id, class_name in TARGET_CLASSES.items():
        counts[f'det_vit_{class_name}'] = 0
    
    # Handle DINOv3 detection output format
    if isinstance(detections, list) and len(detections) > 0:
        detections = detections[0]
    
    # Process detections based on format
    if hasattr(detections, 'pred_logits') and hasattr(detections, 'pred_boxes'):
        # DINOv3 format
        scores = torch.softmax(detections.pred_logits, dim=-1)
        labels = torch.argmax(scores, dim=-1)
        scores = torch.max(scores, dim=-1)[0]
        boxes = detections.pred_boxes
    elif isinstance(detections, dict) and 'scores' in detections:
        # Standard format
        scores = detections['scores']
        labels = detections['labels']
        boxes = detections['boxes']
    else:
        return counts
    
    # Count detections above confidence threshold
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            class_id = label.item()
            if class_id in TARGET_CLASSES:
                class_name = TARGET_CLASSES[class_id]
                counts[f'det_vit_{class_name}'] += 1
    
    return counts

def analyze_single_image(detector, image_path, device='cpu'):
    """
    Purpose: Analyze a single image and return detection counts for all 80 classes
    Arguments: 
        detector - DINOv3 detection model
        image_path - path to image file
        device - device to run inference on
    Returns: Dictionary with image_id and detection counts for all 80 classes
    """
    try:
        # Load and transform image
        batch_img = load_and_transform_image(image_path, device)
        if batch_img is None:
            return None
        
        # Run detection
        with torch.inference_mode():
            detections = detector(batch_img)
        
        # Extract image ID (keep extension)
        image_id = Path(image_path).name
        
        # Count objects for all 80 target classes
        object_counts = count_objects(detections)
        
        # Create result dictionary
        result = {'image_id': image_id}
        result.update(object_counts)
        
        # Clean up tensors to free memory
        del batch_img, detections
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing image {image_path}: {e}")
        gc.collect()
        return None

# Perception scores functionality removed - only extracting detection features

def load_image_manifests():
    """
    Purpose: Load image paths from config-defined manifests
    Arguments: None
    Returns: Tuple of (place_pulse_images, local_images) lists
    """
    print("📁 Loading image manifests...")
    
    # Load place pulse images
    with open(PATHS['pp_images_manifest'], 'r') as f:
        pp_images = [line.strip() for line in f if line.strip()]
    
    # Load local images
    with open(PATHS['local_images_manifest'], 'r') as f:
        local_images = [line.strip() for line in f if line.strip()]
    
    print(f"   Place pulse images: {len(pp_images)}")
    print(f"   Local images: {len(local_images)}")
    print(f"   Total images: {len(pp_images) + len(local_images)}")
    
    return pp_images, local_images

# Perception scores merging functionality removed - only extracting detection features

def save_progress(results, test_mode, processed_count, total_count):
    """
    Purpose: Save progress to avoid data loss
    Arguments: 
        results - list of processed results
        test_mode - whether in test mode
        processed_count - number of images processed
        total_count - total number of images
    Returns: None
    """
    progress_file = PATHS['output_dir'] / PROGRESS_FILE
    progress_data = {
        'processed_count': processed_count,
        'total_count': total_count,
        'test_mode': test_mode,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results_count': len(results),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'target_classes': len(TARGET_CLASSES),
        'version': '3.0',
        'last_saved_count': len(results)
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def save_features_dataframe(results_df, test_mode=False, is_temp=False):
    """
    Purpose: Save features dataframe with detection results only
    Arguments: 
        results_df - DataFrame with detection results
        test_mode - whether in test mode
        is_temp - whether this is a temporary save
    Returns: DataFrame with detection results
    """
    if not results_df.empty:
        # Determine filename
        if is_temp:
            filename = TEMP_FEATURES_FILE
        elif test_mode:
            filename = f"test_{FEATURES_FILE}"
        else:
            filename = FEATURES_FILE
        
        output_path = PATHS['output_dir'] / filename
        results_df.to_csv(output_path, index=False)
        
        if is_temp:
            print(f"💾 Temporary features saved: {output_path}")
        else:
            print(f"💾 Features saved: {output_path}")
        
        return results_df
    return None

def load_existing_progress():
    """
    Purpose: Load existing progress and results if available
    Arguments: None
    Returns: Tuple of (progress_data, existing_df) or (None, None) if no progress found
    """
    progress_file = PATHS['output_dir'] / PROGRESS_FILE
    temp_features_file = PATHS['output_dir'] / TEMP_FEATURES_FILE
    
    if progress_file.exists() and temp_features_file.exists():
        try:
            # Load progress
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Load existing features
            existing_df = pd.read_csv(temp_features_file)
            
            # Validate that the data makes sense
            if len(existing_df) != progress_data['processed_count']:
                print(f"⚠️  Warning: Progress count ({progress_data['processed_count']}) doesn't match results ({len(existing_df)})")
                print(f"   This might indicate corruption. Consider starting fresh.")
                return None, None
            
            print(f"📂 Found existing progress:")
            print(f"   Processed: {progress_data['processed_count']}/{progress_data['total_count']} images")
            print(f"   Results: {len(existing_df)} feature rows")
            print(f"   Last updated: {progress_data['timestamp']}")
            
            return progress_data, existing_df
            
        except Exception as e:
            print(f"⚠️  Error loading existing progress: {e}")
            return None, None
    
    return None, None

def get_remaining_images(all_images, processed_count):
    """
    Purpose: Get list of images that still need to be processed
    Arguments: 
        all_images - list of all image paths
        processed_count - number of images already processed
    Returns: List of remaining image paths
    """
    if processed_count >= len(all_images):
        return []
    return all_images[processed_count:]

def can_resume():
    """
    Purpose: Check if the script can resume from a previous run
    Arguments: None
    Returns: Tuple of (can_resume_flag, progress_data)
    """
    progress_file = PATHS['output_dir'] / PROGRESS_FILE
    temp_features_file = PATHS['output_dir'] / TEMP_FEATURES_FILE
    
    if progress_file.exists() and temp_features_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Check if this is a valid progress file
            if (progress_data.get('version') == '3.0' and 
                'processed_count' in progress_data and 
                'total_count' in progress_data):
                return True, progress_data
        except Exception:
            pass
    
    return False, None

def get_resume_summary(progress_data):
    """
    Purpose: Get a summary of what will be processed when resuming
    Arguments: progress_data - progress data dictionary
    Returns: String summary of resume information
    """
    if progress_data is None:
        return "No resume information available"
    
    processed = progress_data['processed_count']
    total = progress_data['total_count']
    remaining = total - processed
    percent_complete = (processed / total) * 100
    
    return f"{processed:,}/{total:,} images ({percent_complete:.1f}% complete), {remaining:,} remaining"

def run_feature_extraction(test_mode=True, max_images=TEST_MODE_MAX_IMAGES, resume=True):
    """
    Purpose: Run 80-class detection feature extraction using configuration with resume capability
    Arguments: 
        test_mode - whether to run in test mode with limited images
        max_images - maximum number of images to process in test mode
        resume - whether to attempt to resume from previous run
    Returns: DataFrame with extracted features and perception scores
    
    Resume Functionality:
        - Automatically saves progress every 100 images (full mode only)
        - Can resume from where it left off if interrupted
        - Progress files: progress_80_classes.json, temp_detection_features_80_classes.csv
        - User can choose to continue or start fresh when resuming
    """
    print("🚀 Starting 80-Class Object Detection Feature Extraction v3.0")
    print("=" * 70)
    
    # Get device
    device = get_device()
    
    # Load detector
    detector = load_detector(device)
    if detector is None:
        print("❌ Failed to load detector")
        return None
    
    print(f"🔧 Using {device.upper()} for detection")
    print(f"📌 Configuration: {len(TARGET_CLASSES)} target classes, threshold={CONFIDENCE_THRESHOLD}")
    
    # Load data
    pp_images, local_images = load_image_manifests()
    
    # Combine all images
    all_images = pp_images + local_images
    
    if test_mode:
        all_images = all_images[:max_images]
        print(f"🧪 TEST MODE: Processing only {len(all_images)} images")
    
    # Ensure output directory exists
    PATHS['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Check for existing progress
    start_count = 0
    results = []
    
    if resume and not test_mode:
        progress_data, existing_df = load_existing_progress()
        if progress_data and existing_df is not None:
            start_count = progress_data['processed_count']
            results = existing_df.to_dict('records')
            
            # Convert back to list of dictionaries
            results = [dict(row) for row in results]
            
            print(f"🔄 Resuming from image {start_count + 1}")
            print(f"   Already have {len(results)} results")
    
    # Get remaining images to process
    remaining_images = get_remaining_images(all_images, start_count)
    
    if not remaining_images:
        print("✅ All images already processed!")
        if results:
            # Save final results
            results_df = pd.DataFrame(results)
            final_df = save_features_dataframe(results_df, perception_scores, test_mode, is_temp=False)
            return final_df
        return None
    
    print(f"\n🔍 Processing {len(remaining_images)} remaining images...")
    
    # Process remaining images
    for i, image_path in enumerate(tqdm(remaining_images, desc="Processing images")):
        try:
            result = analyze_single_image(detector, image_path, device)
            if result:
                results.append(result)
                
                # Print progress for first few images
                if i < 5:
                    total_detections = sum(result[f'det_vit_{class_name}'] 
                                         for class_name in TARGET_CLASSES.values())
                    print(f"   Image {start_count + i + 1}: {result['image_id']} - Total detections: {total_detections}")
            
            # Save progress periodically
            if not test_mode and (start_count + i + 1) % PROGRESS_SAVE_INTERVAL == 0:
                save_progress(results, test_mode, start_count + i + 1, len(all_images))
                print(f"💾 Progress saved: {start_count + i + 1}/{len(all_images)} images processed")
            
            # Auto-save features every 100 images
            if not test_mode and (start_count + i + 1) % AUTO_SAVE_INTERVAL == 0:
                results_df = pd.DataFrame(results)
                save_features_dataframe(results_df, test_mode, is_temp=True)
                print(f"💾 Auto-save: {start_count + i + 1}/{len(all_images)} images processed")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    if not results:
        print("❌ No results generated")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    print(f"\n🔗 Saving final results...")
    final_df = save_features_dataframe(results_df, test_mode, is_temp=False)
    
    if final_df is None:
        print("❌ Failed to save final results")
        return None
    
    # Save final progress
    if not test_mode:
        save_progress(results, test_mode, len(all_images), len(all_images))
    
    # Print summary
    print(f"\n📊 80-Class Detection Feature Extraction Summary:")
    print(f"   Version: 3.0 (All 80 COCO classes with auto-save and resume)")
    print(f"   Total images processed: {len(final_df)}")
    print(f"   Detection features: {len(get_detection_feature_names())} classes")
    print(f"   Features: 80 detection counts + image_id")
    
    # Show sample results
    print(f"\n📋 Sample Results (first 10 detection features):")
    feature_cols = ['image_id'] + get_detection_feature_names()[:10]
    sample_df = final_df[feature_cols].head()
    print(sample_df.to_string())
    
    # Show detection statistics
    print(f"\n📈 Detection Statistics (total counts for all 80 classes):")
    det_cols = get_detection_feature_names()
    detection_stats = final_df[det_cols].sum().sort_values(ascending=False)
    print(detection_stats.head(20).to_string())  # Show top 20 detected classes
    print(f"... and {len(detection_stats) - 20} more classes")
    
    # Clean up memory
    del results_df, detector
    gc.collect()
    
    # Clean up temporary file if it exists
    if not test_mode:
        temp_file = PATHS['output_dir'] / TEMP_FEATURES_FILE
        if temp_file.exists():
            try:
                temp_file.unlink()
                print(f"🧹 Temporary file cleaned up: {temp_file}")
            except Exception as e:
                print(f"⚠️  Could not clean up temporary file: {e}")
    
    return final_df

def main():
    """
    Purpose: Main function to run 80-class detection feature extraction
    Arguments: None
    Returns: None
    """
    # Print configuration
    print_config()
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed!")
        print("Please check the paths in config_80_classes.py")
        return
    
    # Run test extraction first
    print(f"\n🧪 Running test extraction on {TEST_MODE_MAX_IMAGES} images...")
    test_results = run_feature_extraction(test_mode=True, max_images=TEST_MODE_MAX_IMAGES)
    
    if test_results is not None:
        print("\n🎉 Test extraction completed!")
        
        # Ask user if they want to run full extraction
        print(f"\n" + "="*70)
        response = input("Do you want to run the full extraction on all images? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("🚀 Running full extraction...")
            print(f"⚠️  This will process ~111,000 images and may take several hours.")
            print(f"💾 Progress will be saved every {PROGRESS_SAVE_INTERVAL} images to prevent data loss.")
            print(f"💾 Features will be auto-saved every {AUTO_SAVE_INTERVAL} images")
            
            # Check if we can resume
            can_resume_flag, progress_info = can_resume()
            if can_resume_flag and progress_info is not None:
                print(f"🔄 Resume available: {get_resume_summary(progress_info)}")
                print(f"   Last updated: {progress_info['timestamp']}")
            
            confirm = input("Are you sure? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                if can_resume_flag:
                    print("🔄 Full extraction will resume from where it left off.")
                    resume_choice = input("Do you want to resume or start fresh? (resume/fresh): ").lower().strip()
                    if resume_choice == 'fresh':
                        print("🆕 Starting fresh full extraction.")
                        # Clean up old progress files
                        try:
                            progress_file = PATHS['output_dir'] / PROGRESS_FILE
                            temp_file = PATHS['output_dir'] / TEMP_FEATURES_FILE
                            if progress_file.exists():
                                progress_file.unlink()
                            if temp_file.exists():
                                temp_file.unlink()
                            print("🧹 Old progress files cleaned up.")
                        except Exception as e:
                            print(f"⚠️  Could not clean up old files: {e}")
                        full_results = run_feature_extraction(test_mode=False, resume=False)
                    else:
                        print("🔄 Resuming from previous state.")
                        full_results = run_feature_extraction(test_mode=False, resume=True)
                else:
                    print("🆕 Starting fresh full extraction.")
                    full_results = run_feature_extraction(test_mode=False, resume=False)
                
                if full_results is not None:
                    print("🎉 Full extraction completed!")
                else:
                    print("❌ Full extraction failed!")
            else:
                print("✅ Full extraction cancelled.")
        else:
            print("✅ Test extraction completed. Full extraction skipped.")
    else:
        print("❌ Test extraction failed!")

if __name__ == "__main__":
    main()
