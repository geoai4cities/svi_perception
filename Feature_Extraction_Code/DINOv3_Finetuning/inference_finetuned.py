#!/usr/bin/env python3
"""
DINOv3 IDD Finetuned Model Inference Script
Performs semantic segmentation on single images using the finetuned IDD model
"""

import sys
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import argparse

# Add DINOv3 repo to path - UPDATE THIS or set DINOV3_REPO environment variable
DINOV3_REPO = os.environ.get('DINOV3_REPO', str(Path(__file__).parent.parent / "DINOv3"))
sys.path.insert(0, DINOV3_REPO)
sys.path.insert(0, str(Path(__file__).parent))

from dinov3.eval.segmentation.inference import make_inference

# Import IDD configuration
from idd_classes_config_verified import (
    IDD_CLASS_NAMES, 
    IDD_COLORMAP, 
    NUM_CLASSES,
    create_colored_mask
)

# Import training-compatible inference for finetuned model
from scripts.make_inference_training import make_inference_training


class DINOv3IDDInference:
    """Inference class for DINOv3 IDD finetuned model"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to finetuned model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.transform = None
        
        # Base paths - set FEATURE_EXTRACTOR_BASE environment variable or modify here
        BASE_PATH = Path(os.environ.get('FEATURE_EXTRACTOR_BASE',
                                        str(Path(__file__).parent.parent)))
        self.backbone_weights = BASE_PATH / "weights/dinov3/backbones/dinov3_vit7b16_pretrain_lvd1689m.pth"
        self.ade20k_weights = BASE_PATH / "weights/dinov3/adapters/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
        
        self._load_model()
        self._setup_transform()
    
    def _load_model(self):
        """Load the finetuned model"""
        print(f"🔍 Loading DINOv3 finetuned model from {self.checkpoint_path}")
        
        # Load base model
        self.model = torch.hub.load(
            DINOV3_REPO, 
            'dinov3_vit7b16_ms', 
            source="local",
            weights=str(self.ade20k_weights),
            backbone_weights=str(self.backbone_weights)
        )
        
        # Load finetuned checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Load only the modified weights (class embedding layer)
        model_state = checkpoint['model_state_dict']
        
        # Filter to only load the classification head weights
        finetuned_weights = {k: v for k, v in model_state.items() 
                           if 'predictor.class_embed' in k}
        
        # Load the finetuned weights
        missing_keys, unexpected_keys = self.model.load_state_dict(finetuned_weights, strict=False)
        
        print(f"✅ Model loaded successfully")
        print(f"   - Finetuned parameters: {len(finetuned_weights)}")
        print(f"   - Best validation IoU: {checkpoint.get('val_iou', 'N/A')}")
        print(f"   - Training epoch: {checkpoint.get('epoch', 'N/A')}")
        
        self.model = self.model.to(self.device).eval()
    
    def _setup_transform(self):
        """Setup image preprocessing transform"""
        # Use same transform as training (448x448 for memory optimization)
        self.img_size = 448
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate input image"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"📸 Image loaded: {image_path}")
        print(f"   - Original size: {image.size}")
        return image
    
    def predict(self, image: Image.Image) -> tuple:
        """
        Run inference on image
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            tuple: (segmentation_map, prediction_probabilities)
        """
        print("🤖 Running inference...")
        
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Preprocess image
                batch_img = self.transform(image)[None].to(self.device)
                
                # Run inference using training-compatible method
                predictions = make_inference_training(
                    batch_img,
                    self.model,
                    inference_mode="whole",
                    decoder_head_type="m2f",
                    rescale_to=(self.img_size, self.img_size),
                    n_output_channels=NUM_CLASSES,
                    output_activation=partial(torch.nn.functional.softmax, dim=1)
                )
                
                # Get segmentation map
                segmentation_map = predictions.argmax(dim=1).squeeze().cpu().numpy()
                
                # Get prediction probabilities
                pred_probs = predictions.squeeze().cpu().numpy()
        
        print(f"✅ Inference completed")
        print(f"   - Output shape: {segmentation_map.shape}")
        print(f"   - Unique classes: {len(np.unique(segmentation_map))}")
        
        return segmentation_map, pred_probs
    
    def create_visualization(self, 
                           original_image: Image.Image,
                           segmentation_map: np.ndarray,
                           output_path: Path) -> dict:
        """
        Create and save visualization outputs
        
        Args:
            original_image: Original input image
            segmentation_map: Predicted segmentation map
            output_path: Output directory path
            
        Returns:
            dict: Paths to saved files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Resize segmentation map to original image size for visualization
        original_size = original_image.size
        seg_map_resized = np.array(Image.fromarray(segmentation_map.astype(np.uint8))
                                  .resize(original_size, Image.NEAREST))
        
        # Create colored segmentation mask
        colored_mask = create_colored_mask(seg_map_resized)
        
        # Save outputs
        outputs = {}
        
        # 1. Original image
        original_path = output_path / "original.png"
        original_image.save(original_path)
        outputs['original'] = original_path
        
        # 2. Raw segmentation map (for further processing)
        seg_raw_path = output_path / "segmentation_raw.npy"
        np.save(seg_raw_path, seg_map_resized)
        outputs['segmentation_raw'] = seg_raw_path
        
        # 3. Colored segmentation mask
        colored_path = output_path / "segmentation_colored.png"
        Image.fromarray(colored_mask).save(colored_path)
        outputs['segmentation_colored'] = colored_path
        
        # 4. Side-by-side comparison
        comparison_path = output_path / "comparison.png"
        self._create_comparison(original_image, colored_mask, comparison_path)
        outputs['comparison'] = comparison_path
        
        # 5. Class statistics
        stats_path = output_path / "class_statistics.txt"
        self._save_class_statistics(seg_map_resized, stats_path)
        outputs['statistics'] = stats_path
        
        print(f"💾 Outputs saved to: {output_path}")
        for key, path in outputs.items():
            print(f"   - {key}: {path.name}")
        
        return outputs
    
    def _create_comparison(self, original: Image.Image, colored_mask: np.ndarray, save_path: Path):
        """Create side-by-side comparison image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(colored_mask)
        ax2.set_title('IDD Segmentation (26 Classes)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_class_statistics(self, segmentation_map: np.ndarray, save_path: Path):
        """Save class distribution statistics"""
        unique, counts = np.unique(segmentation_map, return_counts=True)
        total_pixels = segmentation_map.size
        
        with open(save_path, 'w') as f:
            f.write("IDD SEGMENTATION STATISTICS\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Total pixels: {total_pixels:,}\\n")
            f.write(f"Classes found: {len(unique)}\\n\\n")
            
            f.write("CLASS DISTRIBUTION:\\n")
            f.write("-" * 50 + "\\n")
            
            for class_id, count in zip(unique, counts):
                percentage = (count / total_pixels) * 100
                class_name = IDD_CLASS_NAMES[class_id] if class_id < len(IDD_CLASS_NAMES) else f"unknown_{class_id}"
                f.write(f"{class_id:2d}: {class_name:25s} - {count:8,} px ({percentage:5.2f}%)\\n")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='DINOv3 IDD Finetuned Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to finetuned model checkpoint (.pth)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='./inference_outputs',
                       help='Output directory (default: ./inference_outputs)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize inference model
    inference = DINOv3IDDInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Load image
    image = inference.load_image(args.image)
    
    # Run inference
    segmentation_map, pred_probs = inference.predict(image)
    
    # Create output directory
    image_name = Path(args.image).stem
    output_dir = Path(args.output) / f"{image_name}_segmentation"
    
    # Create visualizations
    outputs = inference.create_visualization(
        original_image=image,
        segmentation_map=segmentation_map,
        output_path=output_dir
    )
    
    print("\\n🎯 Inference completed successfully!")
    print(f"   📁 Results: {output_dir}")
    
    return outputs


if __name__ == "__main__":
    main()