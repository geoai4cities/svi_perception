#!/usr/bin/env python3
"""
Purpose: Visualize COCO-80 object detection results on test images and compare with CSV counts
Arguments: None (uses manifests and CSVs from config_80_classes)
Returns: Saves visualization images and prints comparison summary
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from tqdm import tqdm

# Import 80-class configuration
from config_80_classes import (
    PATHS, get_device, TARGET_CLASSES, COCO_CLASS_NAMES,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD
)

# Add DINOv3 repository to path
sys.path.append(str(PATHS['dinov3_repo']))


def load_detector(device: str) -> torch.nn.Module:
    """
    Purpose: Load DINOv3 detection model using config paths
    Arguments: device (str) - device to load model on (e.g., 'cpu', 'cuda:0')
    Returns: Loaded DINOv3 detection model in eval mode
    """
    detector = torch.hub.load(
        str(PATHS['dinov3_repo']),
        'dinov3_vit7b16_de',
        source='local',
        weights=str(PATHS['detector_weights']),
        backbone_weights=str(PATHS['backbone_weights'])
    )
    if device != 'cpu':
        detector = detector.to(device)
    detector.eval()
    return detector


def make_transform(img_size: int = IMAGE_SIZE):
    """
    Purpose: Create image transformation pipeline for DINOv3 detection
    Arguments: img_size (int) - target resize for model input
    Returns: torchvision transform pipeline
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_test_results() -> pd.DataFrame | None:
    """
    Purpose: Load detection counts CSV for test images
    Arguments: None
    Returns: pandas DataFrame or None if not found
    """
    candidates = [
        PATHS['output_dir'] / 'test_detection_features_80_classes.csv',
        PATHS['output_dir'] / 'detection_features_80_classes.csv',
        PATHS['output_dir'].parent / 'agrihub_backup' / 'object_detection_analysis' / 'output' / 'test_detection_features_v2.csv',
        PATHS['output_dir'].parent / 'agrihub_backup' / 'object_detection_analysis' / 'output' / 'test_detection_features.csv',
    ]
    for csv_path in candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded detection CSV: {csv_path} ({len(df)} rows)")
            return df
    print("❌ No detection CSV found.")
    return None


def get_test_image_paths(max_images: int = 5) -> list[str]:
    """
    Purpose: Read manifests and return first N PlacePulse images
    Arguments: max_images (int) - number of images to return
    Returns: List of image paths
    """
    with open(PATHS['pp_images_manifest'], 'r') as f:
        pp_images = [line.strip() for line in f if line.strip()]
    print(f"✅ Found {len(pp_images)} PP images in manifest")
    return pp_images[:max_images]


def run_detector(detector: torch.nn.Module, image_path: str, device: str):
    """
    Purpose: Run detector on a single image and return detections
    Arguments:
        detector - loaded detection model
        image_path - path to input image
        device - device string (e.g., 'cpu', 'cuda:0')
    Returns: dict with keys: 'boxes', 'labels', 'scores'
    """
    from torchvision.ops import box_convert
    transform = make_transform(IMAGE_SIZE)
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    batch = transform(image).unsqueeze(0)
    if device != 'cpu':
        batch = batch.to(device)
    with torch.inference_mode():
        outputs = detector(batch)
    # Return raw outputs to draw function to handle scaling consistently
    return {'raw': outputs, 'orig_size': (orig_w, orig_h)}


def count_by_class_from_raw(detections_raw, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, int]:
    """
    Purpose: Count detections per COCO class with score threshold, mirroring extractor logic
    Arguments:
        detections_raw - raw detector outputs (list/module/dict)
        confidence_threshold - minimum score
    Returns: Dict of {feature_name: count}
    """
    counts: dict[str, int] = {f'det_vit_{name}': 0 for name in TARGET_CLASSES.values()}

    det = detections_raw
    if isinstance(det, list) and len(det) > 0:
        det = det[0]

    if hasattr(det, 'pred_logits') and hasattr(det, 'pred_boxes'):
        scores_all = torch.softmax(det.pred_logits, dim=-1)
        scores, labels = torch.max(scores_all, dim=-1)
        boxes = det.pred_boxes
    elif isinstance(det, dict) and all(k in det for k in ['scores', 'labels', 'boxes']):
        scores = det['scores']
        labels = det['labels']
        boxes = det['boxes']
    else:
        return counts

    for score, label in zip(scores, labels):
        if float(score) >= float(confidence_threshold):
            class_id = int(label.item())
            if class_id in TARGET_CLASSES:
                name = TARGET_CLASSES[class_id]
                counts[f'det_vit_{name}'] += 1
    return counts


def color_for_class(coco_id: int):
    """
    Purpose: Get a deterministic color for a COCO class id
    Arguments: coco_id (int) - COCO class id (1..90 subset)
    Returns: Tuple (r, g, b) in 0..1
    """
    cmap = cm.get_cmap('tab20', 20)
    idx = (coco_id % 20)
    return cmap(idx)[:3]


def draw_detections_pil(image: Image.Image, detections, confidence_threshold: float, original_size=None, model_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Purpose: Draw bounding boxes and labels on image with proper coordinate scaling (mirrors existing OD viz)
    Arguments:
        image - PIL image
        detections - raw detector outputs (list/dict/module output)
        confidence_threshold - score threshold
        original_size - tuple (width,height); default to image.size
        model_size - resize used for model input
    Returns: Annotated PIL image
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    if original_size is None:
        original_size = image.size
    orig_width, orig_height = original_size
    scale_x = orig_width / model_size
    scale_y = orig_height / model_size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Unwrap output
    det = detections
    if isinstance(det, dict) and 'raw' in det:
        det = det['raw']
    if isinstance(det, list) and len(det) > 0:
        det = det[0]

    # Parse formats
    if hasattr(det, 'pred_logits') and hasattr(det, 'pred_boxes'):
        scores_all = torch.softmax(det.pred_logits, dim=-1)
        scores, labels = torch.max(scores_all, dim=-1)
        boxes = det.pred_boxes
    elif isinstance(det, dict) and all(k in det for k in ['scores', 'labels', 'boxes']):
        scores = det['scores']
        labels = det['labels']
        boxes = det['boxes']
    else:
        return img_draw

    colors = ['red','blue','green','yellow','purple','orange','pink','cyan']
    color_idx = 0

    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
        class_id = int(label.item())
        if class_id not in TARGET_CLASSES:
            continue
        # Expect xyxy at model scale; scale to original
        x1, y1, x2, y2 = box.tolist()
        x1_scaled = max(0.0, min(x1 * scale_x, orig_width))
        y1_scaled = max(0.0, min(y1 * scale_y, orig_height))
        x2_scaled = max(0.0, min(x2 * scale_x, orig_width))
        y2_scaled = max(0.0, min(y2 * scale_y, orig_height))

        color = colors[color_idx % len(colors)]
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=3)
        name = TARGET_CLASSES[class_id]
        label_text = f"{name}: {float(score):.2f}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        tx = max(0, x1_scaled)
        ty = max(0, y1_scaled - th - 5)
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=color, outline=color)
        draw.text((tx, ty), label_text, fill='white', font=font)
        color_idx += 1

    return img_draw


def visualize_single_image(detector, image_path: str, csv_row: dict, device: str, output_dir: Path) -> bool:
    """
    Purpose: Visualize detection for a single image and compare with CSV
    Arguments:
        detector - loaded detection model
        image_path - path to image
        csv_row - dict of CSV row values for this image
        device - device string
        output_dir - directory to save visualization
    Returns: True on success, False otherwise
    """
    try:
        image = Image.open(image_path).convert('RGB')
        det = run_detector(detector, image_path, device)
        # Visual counts using thresholded logic
        raw = det['raw']
        visual_counts = count_by_class_from_raw(raw, CONFIDENCE_THRESHOLD)

        # CSV counts
        csv_counts = {k: v for k, v in csv_row.items() if k.startswith('det_vit_')}

        # Figure layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original\n{Path(image_path).stem}", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Detections overlay using PIL scaling logic
        annotated = draw_detections_pil(image, det, CONFIDENCE_THRESHOLD, original_size=image.size, model_size=IMAGE_SIZE)
        axes[0, 1].imshow(annotated)
        axes[0, 1].set_title("Detections (COCO-80)", fontsize=14, fontweight='bold')

        # Top-10 comparison chart
        # Pick top-10 by CSV counts
        top = sorted(csv_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        class_names, csv_vals, vis_vals = [], [], []
        for feat_name, csv_val in top:
            class_name = feat_name.replace('det_vit_', '')
            class_names.append(class_name)
            csv_vals.append(int(csv_val))
            vis_vals.append(int(visual_counts.get(feat_name, 0)))

        x = np.arange(len(class_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, csv_vals, width, label='CSV Counts', color='skyblue', alpha=0.85)
        axes[1, 0].bar(x + width/2, vis_vals, width, label='Visual Counts', color='salmon', alpha=0.85)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Top 10 Classes: CSV vs Visual Counts', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Legend with unique detected classes and counts (from parsed labels)
        # For legend, parse labels from raw and apply threshold
        parsed_unique = []
        det_raw = raw
        if isinstance(det_raw, list) and len(det_raw) > 0:
            det_raw = det_raw[0]
        if hasattr(det_raw, 'pred_logits') and hasattr(det_raw, 'pred_boxes'):
            scores_all = torch.softmax(det_raw.pred_logits, dim=-1)
            scores, labels_ = torch.max(scores_all, dim=-1)
            for s, l in zip(scores.tolist(), labels_.tolist()):
                if s >= float(CONFIDENCE_THRESHOLD) and l in TARGET_CLASSES:
                    parsed_unique.append(l)
        elif isinstance(det_raw, dict) and 'labels' in det_raw and 'scores' in det_raw:
            for s, l in zip(det_raw['scores'].tolist(), det_raw['labels'].tolist()):
                if s >= float(CONFIDENCE_THRESHOLD) and l in TARGET_CLASSES:
                    parsed_unique.append(l)
        unique_labels = sorted(set(parsed_unique))
        legend_elements = []
        for lab in unique_labels:
            name = TARGET_CLASSES[lab]
            color = color_for_class(lab)
            count = int(visual_counts.get(f'det_vit_{name}', 0))
            legend_elements.append(mpatches.Patch(color=color, label=f"{name} ({count})"))
        axes[1, 1].axis('off')
        if legend_elements:
            legend = axes[1, 1].legend(
                handles=legend_elements,
                loc='center', ncol=2, fontsize=10,
                title='Detected Classes (count)', title_fontsize=12
            )
            legend.get_title().set_fontweight('bold')

        plt.tight_layout()
        out_path = output_dir / f"{Path(image_path).stem}_coco80_detection_comparison.jpg"
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {out_path}")

        # Print quick diff summary
        total_diff = sum(abs(csv_vals[i] - vis_vals[i]) for i in range(len(csv_vals)))
        print(f"   Total difference (top 10): {total_diff}")
        return True

    except Exception as e:
        print(f"❌ Error visualizing {image_path}: {e}")
        return False


def main():
    """
    Purpose: Main function to visualize COCO-80 detections for 5 images and compare with CSV
    Arguments: None
    Returns: None
    """
    print("🎯 Starting COCO-80 Detection Visualization")
    print("=" * 60)

    df = load_test_results()
    if df is None or df.empty:
        return

    test_images = get_test_image_paths(max_images=5)

    output_dir = PATHS['visualization_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    detector = load_detector(device)

    print(f"\n📸 Visualizing {len(test_images)} images...")
    ok = 0
    for img_path in tqdm(test_images, desc='Visualizing images'):
        image_name = Path(img_path).name
        rows = df[df['image_id'] == image_name]
        if rows.empty:
            print(f"⚠️  No CSV row for {image_name}")
            continue
        row = rows.iloc[0].to_dict()
        if visualize_single_image(detector, img_path, row, device, output_dir):
            ok += 1

    print(f"\n🎉 Done: {ok}/{len(test_images)} images visualized")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()


