#!/usr/bin/env python3
"""
Purpose: Visualize hybrid segmentation results on test images and compare with CSV area ratios
Arguments: None (uses test results from extract_segmentation_features.py)
Returns: Saves visualization images and prints comparison
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import partial
from tqdm import tqdm

# Define base path for relative imports
BASE_DIR = Path(__file__).parent.parent  # Goes up to Feature_Extractor directory

# Add paths for hybrid segmentation (using relative paths)
sys.path.append(str(BASE_DIR / "dinov3_idd_finetuning/finetuned_image_testing"))
sys.path.append(str(BASE_DIR / "dinov3_idd_finetuning/clean_finetuning_setup"))

# DINOv3 repository path (relative)
DINOV3_REPO = str(BASE_DIR / "dinov3_idd_finetuning/dinov3_github_repo/dinov3")
sys.path.insert(0, DINOV3_REPO)

from dinov3.eval.segmentation.inference import make_inference

# Import IDD class definitions and colormaps
from idd_classes_config_verified import IDD_CLASS_NAMES, IDD_COLORMAP, IGNORE_INDEX

# Import ADE20K definitions for comparison
from ADE20K_colormaps import ADE20K_CLASS_NAMES

# Define weights paths (relative to base directory)
SEGMENTOR_WEIGHTS = BASE_DIR / "weights/dinov3/adapters/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_WEIGHTS = BASE_DIR / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth"
FINETUNED_CHECKPOINT = BASE_DIR / "dinov3_idd_finetuning/clean_finetuning_setup/checkpoints/train_30epoch_full_20250824_143906/best_model.pth"

# Define mapping (same as in extract_segmentation_features.py)
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
    44: 17,  # signboard;sign -> billboard
    137: 19, # traffic light -> traffic light
    94: 20,  # pole -> pole
    2: 22,   # building -> building
    62: 23,  # bridge -> bridge
    5: 24,   # tree -> vegetation
    3: 25,   # sky -> sky
}

# IDD-only classes
IDD_ONLY_CLASSES = {1, 5, 8, 12, 13, 18, 21}

def load_test_results():
    """Load test results from CSV file"""
    csv_path = Path("output/test_segmentation_features.csv")
    if not csv_path.exists():
        print(f"❌ Test results file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded test results: {len(df)} images")
    return df

def get_test_image_paths():
    """Get image paths for the 5 test images"""
    base_path = BASE_DIR
    
    # Load image manifests
    with open(base_path / "data/manifests/pp_images.txt", 'r') as f:
        pp_images = [line.strip() for line in f if line.strip()]
    
    # Get first 5 images
    test_images = pp_images[:5]
    print(f"✅ Found {len(test_images)} test images")
    return test_images

def load_hybrid_models():
    """Load both hybrid segmentation models"""
    print("🔍 Loading hybrid segmentation models...")
    
    # Load ADE20K model on CPU
    ade20k_model = torch.hub.load(
        DINOV3_REPO, 
        'dinov3_vit7b16_ms', 
        source="local",
        weights=str(SEGMENTOR_WEIGHTS),
        backbone_weights=str(BACKBONE_WEIGHTS)
    )
    ade20k_model.eval()
    
    # Load IDD finetuned model
    idd_model = torch.hub.load(
        DINOV3_REPO, 
        'dinov3_vit7b16_ms', 
        source="local",
        weights=str(SEGMENTOR_WEIGHTS),
        backbone_weights=str(BACKBONE_WEIGHTS)
    )
    
    # Load finetuned checkpoint
    checkpoint = torch.load(FINETUNED_CHECKPOINT, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        
        # Replace classification head with 26 IDD classes
        import torch.nn as nn
        in_features = 2048
        new_class_embed = nn.Linear(in_features, 26)
        
        # Load finetuned weights
        if 'segmentation_model.1.predictor.class_embed.weight' in model_state:
            new_class_embed.weight.data = model_state['segmentation_model.1.predictor.class_embed.weight']
            new_class_embed.bias.data = model_state['segmentation_model.1.predictor.class_embed.bias']
        
        # Replace in model
        idd_model.segmentation_model[1].predictor.class_embed = new_class_embed
    
    idd_model.eval()
    
    print("✅ Both models loaded successfully")
    return ade20k_model, idd_model

def make_transform(resize_size=512):
    """Create image transformation pipeline"""
    from torchvision import transforms
    
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def process_ade20k_output(segmentor, batch_img, img_size, target_size):
    """Process ADE20K model output"""
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
    
    # Apply +1 correction for ADE20K class mapping issue
    seg_map = segmentation_map[0, 0].cpu().numpy()
    seg_map = seg_map + 1  # Correct the index shift
    
    return seg_map

def process_idd_output(outputs, target_size):
    """Process IDD finetuned model output"""
    pred_logits = outputs['pred_logits']  # [1, N_queries, N_classes]
    pred_masks = outputs['pred_masks']    # [1, N_queries, H, W]
    
    # Apply softmax to class predictions
    pred_probs = F.softmax(pred_logits, dim=-1)
    
    # Resize masks to target size
    pred_masks_resized = F.interpolate(
        pred_masks, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert to final segmentation map (ensure on CPU)
    batch_size, n_queries, h, w = pred_masks_resized.shape
    final_seg = torch.zeros((h, w), dtype=torch.long)  # Keep on CPU
    
    # Get the best class for each query
    query_classes = pred_probs[0].argmax(dim=-1)  # [N_queries]
    
    # Apply masks and assign classes
    for q in range(n_queries):
        mask_prob = torch.sigmoid(pred_masks_resized[0, q])
        mask_binary = mask_prob > 0.5
        
        if mask_binary.sum() > 0:  # If mask has pixels
            class_id = query_classes[q].item()
            if class_id < 26:  # Valid IDD class
                final_seg[mask_binary] = class_id
    
    return final_seg.numpy()

def combine_segmentations(ade20k_seg, idd_seg, img_shape):
    """Combine ADE20K and IDD segmentations into final hybrid result"""
    
    # Initialize final segmentation map
    final_seg = np.zeros(img_shape[:2], dtype=np.int32)
    
    # First, apply ADE20K results for common classes
    for ade_class, idd_class in ADE20K_TO_IDD_MAPPING.items():
        mask = (ade20k_seg == ade_class)
        final_seg[mask] = idd_class
    
    # Then, apply IDD results for IDD-only classes
    for idd_class in IDD_ONLY_CLASSES:
        mask = (idd_seg == idd_class)
        if mask.sum() > 0:
            final_seg[mask] = idd_class
    
    return final_seg

def create_colored_mask(segmentation_map, colormap):
    """Create colored segmentation mask from label array"""
    colored_mask = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    
    for class_id in range(len(colormap)):
        mask = segmentation_map == class_id
        colored_mask[mask] = colormap[class_id]
    
    return Image.fromarray(colored_mask, "RGB")

def calculate_area_ratios(segmentation_map, num_classes=26):
    """Calculate area percentage for each class (0-100 scale)"""
    total_pixels = segmentation_map.size
    area_ratios = {}
    
    for class_id in range(num_classes):
        class_name = IDD_CLASS_NAMES[class_id] if class_id < len(IDD_CLASS_NAMES) else f"class_{class_id}"
        class_pixels = (segmentation_map == class_id).sum()
        area_percentage = float(class_pixels / total_pixels * 100)  # Convert to percentage
        area_ratios[f'seg_vit_{class_name}'] = area_percentage
    
    return area_ratios

def visualize_single_image(ade20k_model, idd_model, image_path, csv_row, output_dir):
    """Visualize segmentation for a single image and compare with CSV"""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        img_size = 512
        transform = make_transform(img_size)
        
        original_size = image.size[::-1]  # (height, width)
        batch_img = transform(image)[None]  # Keep on CPU
        
        # Run hybrid segmentation
        with torch.inference_mode():
            # ADE20K model
            ade20k_seg = process_ade20k_output(ade20k_model, batch_img, img_size, original_size)
            
            # IDD model
            idd_outputs = idd_model(batch_img)
            idd_seg = process_idd_output(idd_outputs, original_size)
            
            # Combine results
            final_seg = combine_segmentations(ade20k_seg, idd_seg, original_size)
        
        # Calculate area ratios from visualization
        visual_ratios = calculate_area_ratios(final_seg)
        
        # Create visualization
        image_name = Path(image_path).stem
        
        # Create colored segmentation masks
        final_colored = create_colored_mask(final_seg, IDD_COLORMAP)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image\n{image_name}", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Hybrid segmentation result
        axes[0, 1].imshow(final_colored)
        axes[0, 1].set_title("Hybrid Segmentation Result\n(19 ADE20K + 7 IDD Classes)", fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Area ratio comparison - visual vs CSV
        # Get top 10 classes by area from CSV
        csv_features = {k: v for k, v in csv_row.items() if k.startswith('seg_vit_')}
        top_classes = sorted(csv_features.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create comparison bars
        class_names = []
        csv_values = []
        visual_values = []
        
        for feature_name, csv_value in top_classes:
            if feature_name in visual_ratios:
                class_names.append(feature_name.replace('seg_vit_', ''))
                csv_values.append(csv_value)
                visual_values.append(visual_ratios[feature_name])
        
        x = np.arange(len(class_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, csv_values, width, label='CSV Values', alpha=0.8, color='skyblue')
        axes[1, 0].bar(x + width/2, visual_values, width, label='Visual Values', alpha=0.8, color='lightcoral')
        
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Area Percentage (%)')
        axes[1, 0].set_title('Top 10 Classes: CSV vs Visual Area Percentages', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Create legend for segmentation classes
        unique_classes = np.unique(final_seg)
        legend_elements = []
        
        for class_id in sorted(unique_classes):
            if class_id < len(IDD_CLASS_NAMES) and class_id < len(IDD_COLORMAP):
                class_name = IDD_CLASS_NAMES[class_id]
                color = IDD_COLORMAP[class_id]
                color_normalized = tuple(c/255.0 for c in color)
                
                # Determine source model
                is_ade20k = any(idd_id == class_id for idd_id in ADE20K_TO_IDD_MAPPING.values())
                is_idd_only = class_id in IDD_ONLY_CLASSES
                
                if is_ade20k:
                    label = f"{class_name} (ADE20K)"
                elif is_idd_only:
                    label = f"{class_name} (IDD)"
                else:
                    label = f"{class_name}"
                
                legend_elements.append(mpatches.Patch(color=color_normalized, label=label))
        
        axes[1, 1].axis('off')
        if legend_elements:
            legend = axes[1, 1].legend(
                handles=legend_elements[:15],  # Show first 15 classes to fit
                loc='center',
                ncol=2,
                fontsize=10,
                title="Segmentation Classes (Top 15)",
                title_fontsize=12
            )
            legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / f"{image_name}_segmentation_comparison.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        
        # Print comparison stats
        total_diff = sum(abs(csv_values[i] - visual_values[i]) for i in range(len(csv_values)))
        print(f"   Total difference (top 10 classes): {total_diff:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error visualizing {image_path}: {e}")
        return False

def main():
    """Main function to visualize test results"""
    print("🎨 Starting Segmentation Visualization")
    print("=" * 50)
    
    # Load test results
    test_df = load_test_results()
    if test_df is None:
        print("❌ Could not load test results. Please run extract_segmentation_features.py first.")
        return
    
    # Get test image paths
    test_images = get_test_image_paths()
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load models
    ade20k_model, idd_model = load_hybrid_models()
    
    # Process each test image
    print(f"\n🎯 Visualizing {len(test_images)} test images...")
    
    successful_visualizations = 0
    
    for i, image_path in enumerate(tqdm(test_images, desc="Creating visualizations")):
        image_name = Path(image_path).name
        
        # Find corresponding CSV row
        csv_row = test_df[test_df['image_id'] == image_name]
        
        if csv_row.empty:
            print(f"⚠️  No CSV data found for {image_name}")
            continue
        
        csv_row = csv_row.iloc[0].to_dict()
        
        # Visualize
        success = visualize_single_image(ade20k_model, idd_model, image_path, csv_row, output_dir)
        if success:
            successful_visualizations += 1
    
    print(f"\n🎉 Visualization Summary:")
    print(f"   Successfully visualized: {successful_visualizations}/{len(test_images)} images")
    print(f"   Output directory: {output_dir.absolute()}")
    
    # Print summary of test results
    print(f"\n📊 Test Results Summary:")
    print(f"   Images in CSV: {len(test_df)}")
    
    # Show top segmentation features across all test images
    seg_columns = [col for col in test_df.columns if col.startswith('seg_vit_')]
    mean_areas = test_df[seg_columns].mean().sort_values(ascending=False)
    
    print(f"\n🏆 Top 10 Segmentation Classes (Average Area Percentage):")
    for i, (feature, ratio) in enumerate(mean_areas.head(10).items()):
        class_name = feature.replace('seg_vit_', '')
        print(f"   {i+1:2d}. {class_name:<20s}: {ratio:.2f}%")

if __name__ == "__main__":
    main()