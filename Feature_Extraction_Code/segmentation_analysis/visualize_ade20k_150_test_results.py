#!/usr/bin/env python3
"""
Purpose: Visualize ADE20K 150-class segmentation results on test images and compare with CSV area ratios
Arguments: None (uses test results from extract_ade20k_150_features.py)
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

# Add paths for ADE20K segmentation (using relative paths)
sys.path.append(str(BASE_DIR / "dinov3_idd_finetuning/dinov3_github_repo/dinov3"))

from dinov3.eval.segmentation.inference import make_inference

# Import ADE20K class definitions and colormaps
sys.path.append(str(BASE_DIR / "segmentation_analysis/full_ade20k_150_features_extraction"))
from ADE20K_colormaps import ADE20K_CLASS_NAMES, ADE20K_COLORMAP

# Define weights paths (relative to base directory)
SEGMENTOR_WEIGHTS = BASE_DIR / "weights/dinov3/adapters/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_WEIGHTS = BASE_DIR / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth"

def load_test_results():
    """Load test results from CSV file"""
    csv_path = Path("test_ade20k_150_features_5_images.csv")
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

def load_ade20k_model():
    """Load ADE20K segmentation model"""
    print("🔍 Loading ADE20K segmentation model...")
    
    # Load ADE20K model on CPU
    ade20k_model = torch.hub.load(
        str(BASE_DIR / "dinov3_idd_finetuning/dinov3_github_repo/dinov3"), 
        'dinov3_vit7b16_ms', 
        source="local",
        weights=str(SEGMENTOR_WEIGHTS),
        backbone_weights=str(BACKBONE_WEIGHTS)
    )
    ade20k_model.eval()
    
    print("✅ ADE20K model loaded successfully")
    return ade20k_model

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
    
    # Extract segmentation map without +1 correction
    # The +1 correction will be applied in calculate_ade20k_area_ratios when getting class names
    # This matches the official testing approach
    seg_map = segmentation_map[0, 0].cpu().numpy()
    
    return seg_map

def create_colored_mask(segmentation_map, colormap):
    """Create colored segmentation mask from label array"""
    colored_mask = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    
    # The segmentation map has values 0-149, colormap has 151 entries (0-150)
    # Map segmentation values 0-149 to colormap indices 1-150 (ADE20K classes 1-150)
    for seg_class_id in range(150):  # Segmentation map values 0-149
        mask = segmentation_map == seg_class_id
        colormap_idx = seg_class_id + 1  # Map to colormap index 1-150 (ADE20K classes)
        colored_mask[mask] = colormap[colormap_idx]
    
    return Image.fromarray(colored_mask, "RGB")

def calculate_area_ratios(segmentation_map, num_classes=151):
    """Calculate area percentage for each ADE20K class (0-100 scale)"""
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

def visualize_single_image(ade20k_model, image_path, csv_row, output_dir):
    """Visualize segmentation for a single image and compare with CSV"""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        img_size = 512
        transform = make_transform(img_size)
        
        original_size = image.size[::-1]  # (height, width)
        batch_img = transform(image)[None]  # Keep on CPU
        
        # Run ADE20K segmentation
        with torch.inference_mode():
            ade20k_seg = process_ade20k_output(ade20k_model, batch_img, img_size, original_size)
        
        # Calculate area ratios from visualization
        visual_ratios = calculate_area_ratios(ade20k_seg)
        
        # Create visualization
        image_name = Path(image_path).stem
        
        # Create colored segmentation masks
        final_colored = create_colored_mask(ade20k_seg, ADE20K_COLORMAP)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image\n{image_name}", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # ADE20K segmentation result
        axes[0, 1].imshow(final_colored)
        axes[0, 1].set_title("ADE20K 150-Class Segmentation Result", fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Area ratio comparison - visual vs CSV
        # Get top 10 classes by area from CSV
        csv_features = {k: v for k, v in csv_row.items() if k.startswith('ade20k_')}
        top_classes = sorted(csv_features.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create comparison bars
        class_names = []
        csv_values = []
        visual_values = []
        
        for feature_name, csv_value in top_classes:
            if feature_name in visual_ratios:
                class_names.append(feature_name.replace('ade20k_', ''))
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
        
        # Create legend for segmentation classes with percentages
        unique_classes = np.unique(ade20k_seg)
        legend_elements = []
        
        # Calculate percentages for each unique class
        total_pixels = ade20k_seg.size
        class_percentages = {}
        
        for class_id in unique_classes:
            pixel_count = (ade20k_seg == class_id).sum()
            percentage = pixel_count / total_pixels * 100
            class_percentages[class_id] = percentage
        
        # Sort by percentage (descending) and create legend
        sorted_classes = sorted(unique_classes, key=lambda x: class_percentages[x], reverse=True)
        
        for class_id in sorted_classes:
            # Map segmentation value to ADE20K class (seg value 0-149 → ADE20K class 1-150)
            ade20k_class = class_id + 1
            colormap_idx = class_id + 1
            
            if ade20k_class < len(ADE20K_CLASS_NAMES) and colormap_idx < len(ADE20K_COLORMAP):
                class_name = ADE20K_CLASS_NAMES[ade20k_class]
                color = ADE20K_COLORMAP[colormap_idx]
                color_normalized = tuple(c/255.0 for c in color)
                percentage = class_percentages[class_id]
                
                legend_elements.append(mpatches.Patch(
                    color=color_normalized, 
                    label=f"{class_name} ({percentage:.1f}%)"
                ))
        
        axes[1, 1].axis('off')
        if legend_elements:
            legend = axes[1, 1].legend(
                handles=legend_elements,  # Show all detected classes
                loc='center',
                ncol=2,
                fontsize=9,
                title="Detected Classes (with % coverage)",
                title_fontsize=12
            )
            legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / f"{image_name}_ade20k_150_segmentation_comparison.jpg"
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
    print("🎨 Starting ADE20K 150-Class Segmentation Visualization")
    print("=" * 60)
    
    # Load test results
    test_df = load_test_results()
    if test_df is None:
        print("❌ Could not load test results. Please run extract_ade20k_150_features.py first.")
        return
    
    # Get test image paths
    test_images = get_test_image_paths()
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    ade20k_model = load_ade20k_model()
    
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
        success = visualize_single_image(ade20k_model, image_path, csv_row, output_dir)
        if success:
            successful_visualizations += 1
    
    print(f"\n🎉 Visualization Summary:")
    print(f"   Successfully visualized: {successful_visualizations}/{len(test_images)} images")
    print(f"   Output directory: {output_dir.absolute()}")
    
    # Print summary of test results
    print(f"\n📊 Test Results Summary:")
    print(f"   Images in CSV: {len(test_df)}")
    
    # Show top segmentation features across all test images
    seg_columns = [col for col in test_df.columns if col.startswith('ade20k_')]
    mean_areas = test_df[seg_columns].mean().sort_values(ascending=False)
    
    print(f"\n🏆 Top 15 ADE20K Segmentation Classes (Average Area Percentage):")
    for i, (feature, ratio) in enumerate(mean_areas.head(15).items()):
        class_name = feature.replace('ade20k_', '')
        print(f"   {i+1:2d}. {class_name:<30s}: {ratio:.2f}%")

if __name__ == "__main__":
    main()
