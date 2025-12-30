#!/usr/bin/env python3
"""
Purpose: Visualize object detection results on test images and compare with CSV counts
Arguments: None (uses test results from extract_detection_features.py)
Returns: Saves visualization images and prints comparison
"""

import sys
import os
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Add DINOv3 repository to path - UPDATE THIS or set DINOV3_REPO environment variable
REPO_DIR = os.environ.get('DINOV3_REPO', str(Path(__file__).parent.parent / "DINOv3"))
sys.path.append(REPO_DIR)

# COCO class names (91 classes including background)
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

# Target classes we're interested in (same as in extract_detection_features.py)
TARGET_CLASSES = {
    1: "person",
    2: "bicycle", 
    3: "car",
    4: "motorcycle",
    6: "bus",
    7: "train",
    8: "truck",
    10: "traffic_light",
    18: "dog",
    21: "cow"
}

def get_class_name(class_id):
    """Get COCO class name for given class ID"""
    if 0 <= class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    else:
        return f"unknown_class_{class_id}"

def load_test_results():
    """Load test results from CSV file"""
    csv_path = Path("output/test_detection_features.csv")
    if not csv_path.exists():
        print(f"❌ Test results file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded test results: {len(df)} images")
    return df

def get_image_paths():
    """Get image paths for the 5 test images"""
    base_path = Path(os.environ.get('FEATURE_EXTRACTOR_BASE',
                                    str(Path(__file__).parent.parent)))
    
    # Load image manifests
    with open(base_path / "data/manifests/pp_images.txt", 'r') as f:
        pp_images = [line.strip() for line in f if line.strip()]
    
    # Get first 5 images
    test_images = pp_images[:5]
    print(f"✅ Found {len(test_images)} test images")
    return test_images

def load_detector():
    """Load DINOv3 detection model"""
    print("🔍 Loading DINOv3 detection model...")

    BASE_PATH = Path(os.environ.get('FEATURE_EXTRACTOR_BASE',
                                    str(Path(__file__).parent.parent)))
    DETECTOR_WEIGHTS = BASE_PATH / "weights/dinov3/adapters/detection/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
    BACKBONE_WEIGHTS = BASE_PATH / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth"
    
    try:
        detector = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_de', source="local", 
                                 weights=str(DETECTOR_WEIGHTS), backbone_weights=str(BACKBONE_WEIGHTS))
        print("✅ DINOv3 detector loaded successfully")
        return detector
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return None

def make_transform(resize_size=896):
    """Create image transformation pipeline"""
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def count_objects(detections, confidence_threshold=0.5):
    """Count objects for target classes"""
    counts = {}
    
    # Initialize counts for all target classes
    for class_id, class_name in TARGET_CLASSES.items():
        counts[f'det_vit_{class_name}'] = 0
    
    # Handle DINOv3 detection output format (list with dict inside)
    if isinstance(detections, list) and len(detections) > 0:
        detections = detections[0]  # Get the first (and only) item
    
    # Handle DINOv3 detection output format
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

def draw_detections(image, detections, confidence_threshold=0.5, original_size=None, model_size=896):
    """Draw bounding boxes and labels on image with proper coordinate scaling"""
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Get original image dimensions for coordinate scaling
    if original_size is None:
        original_size = image.size  # (width, height)
    
    orig_width, orig_height = original_size
    scale_x = orig_width / model_size
    scale_y = orig_height / model_size
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    color_idx = 0
    
    # Handle DINOv3 detection output format (list with dict inside)
    if isinstance(detections, list) and len(detections) > 0:
        detections = detections[0]  # Get the first (and only) item
    
    # Handle DINOv3 detection output format
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
        print(f"   ⚠️  Unknown detection format: {type(detections)}")
        return img_draw
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < confidence_threshold:
            continue
            
        # Convert box from [x1, y1, x2, y2] to [left, top, right, bottom]
        x1, y1, x2, y2 = box.tolist()
        
        # Scale coordinates from model output space to original image space
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Ensure coordinates are within image bounds
        x1_scaled = max(0.0, min(x1_scaled, orig_width))
        y1_scaled = max(0.0, min(y1_scaled, orig_height))
        x2_scaled = max(0.0, min(x2_scaled, orig_width))
        y2_scaled = max(0.0, min(y2_scaled, orig_height))
        
        # Draw bounding box
        color = colors[color_idx % len(colors)]
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=3)
        
        # Draw label
        class_name = get_class_name(label.item())
        label_text = f"{class_name}: {score:.2f}"
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text above the box
        text_x = max(0, x1_scaled)
        text_y = max(0, y1_scaled - text_height - 5)
        
        # Draw text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                      fill=color, outline=color)
        
        # Draw text
        draw.text((text_x, text_y), label_text, fill='white', font=font)
        
        color_idx += 1
    
    return img_draw

def visualize_image(detector, image_path, csv_counts, output_dir, device):
    """Visualize detections on a single image and compare with CSV counts"""
    print(f"\n🔍 Processing: {Path(image_path).name}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    transform = make_transform(896)
    
    # Run detection
    with torch.no_grad():
        batch_img = transform(image).unsqueeze(0)
        # Move tensor to device
        batch_img = batch_img.to(device)
        detections = detector(batch_img)
    
    # Count objects from detection
    detection_counts = count_objects(detections)
    
    # Draw detections
    annotated_image = draw_detections(image, detections, original_size=image.size, model_size=896)
    
    # Save annotated image
    image_name = Path(image_path).stem
    output_path = output_dir / f"{image_name}_detections.jpg"
    annotated_image.save(output_path, quality=95)
    
    # Compare counts
    print(f"   📊 Detection Results vs CSV:")
    for class_name in TARGET_CLASSES.values():
        det_key = f'det_vit_{class_name}'
        csv_count = csv_counts.get(det_key, 0)
        det_count = detection_counts.get(det_key, 0)
        
        status = "✅" if csv_count == det_count else "❌"
        print(f"      {status} {class_name}: Detection={det_count}, CSV={csv_count}")
    
    return detection_counts, annotated_image

def main():
    """Main function to visualize test results"""
    print("🎨 Object Detection Visualization & Verification")
    print("=" * 60)
    
    # Load test results
    test_results = load_test_results()
    if test_results is None:
        return
    
    # Get image paths
    image_paths = get_image_paths()
    if not image_paths:
        return
    
    # Load detector
    detector = load_detector()
    if detector is None:
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector.to(device)
    detector.eval()
    print(f"🔧 Using device: {device}")
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Visualization output directory: {output_dir.absolute()}")
    
    # Process each image
    print(f"\n🎯 Processing {len(image_paths)} test images...")
    
    for i, image_path in enumerate(image_paths):
        # Get CSV counts for this image
        image_id = Path(image_path).name
        csv_row = test_results[test_results['image_id'] == image_id]
        
        if len(csv_row) == 0:
            print(f"❌ No CSV data found for {image_id}")
            continue
        
        csv_counts = csv_row.iloc[0].to_dict()
        
        # Visualize image
        detection_counts, annotated_image = visualize_image(detector, image_path, csv_counts, output_dir, device)
        
        print(f"   💾 Saved: {output_dir / f'{Path(image_path).stem}_detections.jpg'}")
    
    print(f"\n🎉 Visualization complete!")
    print(f"📁 Check results in: {output_dir.absolute()}")
    print(f"🔍 Compare detection counts with CSV values above")

if __name__ == "__main__":
    main()
