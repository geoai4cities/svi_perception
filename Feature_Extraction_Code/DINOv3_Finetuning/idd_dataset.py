"""
IDD Segmentation Dataset Loader
Handles the Indian Driving Dataset with 26 semantic classes
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

# Import IDD class mapping
from idd_classes_config_verified import IDD_TO_SEMANTIC_MAPPING, NUM_CLASSES


class IDDSegmentationDataset(Dataset):
    """
    IDD Segmentation dataset loader following DINOv3 preprocessing
    """
    
    # IDD class information (from verified config)
    NUM_CLASSES = NUM_CLASSES  # 26 classes
    IGNORE_INDEX = 255
    
    # Class names for IDD
    CLASS_NAMES = [
        'road', 'parking', 'sidewalk', 'rail track', 'person',
        'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car',
        'truck', 'bus', 'vehicle fallback', 'curb', 'wall',
        'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light',
        'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation',
        'sky'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 518,
        augment: bool = True,
        return_original_size: bool = False
    ):
        """
        Args:
            root_dir: Path to IDD_Segmentation directory
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (518 for ViT-7B, 224 for ViT-L)
            augment: Whether to apply data augmentation
            return_original_size: Whether to return original image size
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.return_original_size = return_original_size
        
        # Setup paths based on IDD structure
        self.img_dir = self.root_dir / 'leftImg8bit' / split
        self.ann_dir = self.root_dir / 'gtFine' / split
        
        # Collect all images and annotations
        self.samples = []
        self._load_samples()
        
        print(f"IDD {split}: {len(self.samples)} samples loaded")
        
        # Define transforms
        self.transform = self._get_transform()
        self.target_transform = self._get_target_transform()
    
    def _load_samples(self):
        """Load all image-annotation pairs"""
        if not self.img_dir.exists():
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not self.ann_dir.exists():
            raise ValueError(f"Annotation directory not found: {self.ann_dir}")
        
        # IDD has subdirectories for different sequences
        for seq_dir in sorted(self.img_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            for img_path in sorted(seq_dir.glob('*.png')):
                # Construct annotation path
                # Image: leftImg8bit/train/seq/frame_leftImg8bit.png
                # Anno: gtFine/train/seq/frame_gtFine_labelIds.png
                ann_name = img_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                ann_path = self.ann_dir / seq_dir.name / ann_name
                
                if ann_path.exists():
                    self.samples.append({
                        'image': str(img_path),
                        'mask': str(ann_path),
                        'seq': seq_dir.name
                    })
                else:
                    print(f"Warning: No annotation for {img_path}")
    
    def _get_transform(self):
        """Get image transforms following DINOv3 preprocessing"""
        # Always use fixed size resize to avoid dimension mismatches
        # The Mask2Former head expects consistent input dimensions
        transform_list = [
            transforms.Resize((self.img_size, self.img_size), antialias=True)
        ]
        
        if self.augment:
            # Training augmentations (applied after resize to maintain size)
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
            ])
        
        # Convert to tensor and normalize (ImageNet stats for LVD-1689M weights)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def _get_target_transform(self):
        """Get target (mask) transforms"""
        # Always resize to consistent size
        return transforms.Resize(
            (self.img_size, self.img_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        orig_size = image.size  # (width, height)
        
        # Load annotation
        mask = Image.open(sample['mask'])
        
        # Apply transforms
        if self.augment:
            # Apply same horizontal flip to both image and mask
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
        
        # Apply image transforms
        image = self.transform(image)
        
        # Apply mask transforms
        mask = self.target_transform(mask)
        
        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        # Apply IDD to semantic mapping (255 -> 0 for road, etc.)
        mask_mapped = torch.full_like(mask, self.IGNORE_INDEX)  # Default to ignore
        for idd_idx, semantic_idx in IDD_TO_SEMANTIC_MAPPING.items():
            mask_mapped[mask == idd_idx] = semantic_idx
        mask = mask_mapped
        
        output = {
            'image': image,
            'mask': mask,
            'path': sample['image'],
            'seq': sample['seq']
        }
        
        if self.return_original_size:
            output['orig_size'] = orig_size
        
        return output


def create_idd_dataloaders(
    root_dir: str,
    batch_size: int = 2,
    img_size: int = 518,
    num_workers: int = 4,
    augment_train: bool = True
):
    """
    Create train and validation dataloaders for IDD
    
    Args:
        root_dir: Path to IDD_Segmentation directory
        batch_size: Batch size for dataloaders
        img_size: Image size for model input
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = IDDSegmentationDataset(
        root_dir=root_dir,
        split='train',
        img_size=img_size,
        augment=augment_train
    )
    
    val_dataset = IDDSegmentationDataset(
        root_dir=root_dir,
        split='val',
        img_size=img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = IDDSegmentationDataset(
        root_dir='/path/to/IDD_Segmentation',
        split='train',
        img_size=518,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.NUM_CLASSES}")
    print(f"Class names: {dataset.CLASS_NAMES}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")