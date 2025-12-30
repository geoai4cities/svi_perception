#!/usr/bin/env python3
"""
Proper DINOv3 fine-tuning implementation using torch.hub.load
Based on official testing.py architecture
"""

import sys
import os
from pathlib import Path

# Add DINOv3 repo to path - UPDATE THIS PATH to your local DINOv3 installation
# Download from: https://github.com/facebookresearch/dinov3
DINOV3_REPO = os.environ.get("DINOV3_REPO", str(Path(__file__).parent.parent.parent / "DINOv3"))
sys.path.insert(0, DINOV3_REPO)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import logging
from typing import Dict, Tuple
from functools import partial

# Import dataset and config
from idd_dataset import IDDSegmentationDataset
from idd_classes_config_verified import (
    IDD_TRAINING_CONFIG,
    NUM_CLASSES,
    IDD_CLASS_NAMES,
    IDD_COLORMAP,
    create_colored_mask
)

# Training-compatible inference
from make_inference_training import make_inference_training


class DINOv3IDDFinetuner:
    """
    Fine-tuning manager for DINOv3 segmentor on IDD dataset
    Uses proper torch.hub loading for best performance
    """
    
    def __init__(
        self,
        backbone_weights: str,
        ade20k_weights: str,
        output_dir: Path,
        num_classes: int = 26,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load the complete segmentor using torch.hub (proper way)
        logging.info("Loading DINOv3 segmentor via torch.hub...")
        self.model = torch.hub.load(
            DINOV3_REPO,
            'dinov3_vit7b16_ms',
            source="local",
            weights=str(ade20k_weights),
            backbone_weights=str(backbone_weights)
        )
        
        # Modify for IDD classes
        self._modify_for_idd_classes()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Print statistics
        self._print_model_stats()
    
    def _modify_for_idd_classes(self):
        """Modify the segmentor for IDD 26 classes"""
        logging.info("Modifying segmentor for IDD 26 classes...")
        
        # Access the segmentation model components
        # segmentation_model[0] = backbone (DINOv3_Adapter)
        # segmentation_model[1] = decoder (Mask2Former)
        
        if self.freeze_backbone:
            # Freeze the backbone adapter
            for param in self.model.segmentation_model[0].parameters():
                param.requires_grad = False
            logging.info("✓ Backbone frozen")
        
        # Modify the decoder's class head for 26 classes
        decoder = self.model.segmentation_model[1]
        
        # Find and modify the class embedding layer
        if hasattr(decoder, 'predictor') and hasattr(decoder.predictor, 'class_embed'):
            old_class_embed = decoder.predictor.class_embed
            in_features = old_class_embed.in_features  # Should be 2048
            
            # Create new class embedding for IDD
            new_class_embed = nn.Linear(in_features, self.num_classes, bias=True)
            nn.init.xavier_uniform_(new_class_embed.weight)
            nn.init.zeros_(new_class_embed.bias)
            
            # Replace the layer
            decoder.predictor.class_embed = new_class_embed
            logging.info(f"✓ Modified class head: {in_features} → {self.num_classes} classes")
        
        # Optionally freeze most of decoder except final layers
        if self.freeze_backbone:
            # First freeze everything in decoder
            for param in decoder.parameters():
                param.requires_grad = False
            
            # Then unfreeze only class-related layers
            for name, param in decoder.named_parameters():
                if 'class' in name.lower() or 'final' in name.lower():
                    param.requires_grad = True
                    logging.info(f"  Unfrozen: {name}")
    
    def _print_model_stats(self):
        """Print model parameter statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logging.info("Model Statistics:")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        logging.info(f"  Frozen parameters: {frozen_params:,}")
        logging.info(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, scheduler=None) -> float:
        """Train for one epoch"""
        self.model.train()
        losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(dtype=torch.bfloat16):
                # Use gradient-friendly training inference
                outputs = make_inference_training(
                    images,
                    self.model,
                    inference_mode="whole",
                    decoder_head_type="m2f",
                    rescale_to=masks.shape[-2:],
                    n_output_channels=self.num_classes,
                    output_activation=None,  # Raw logits for training
                )
                
                # Resize outputs to match mask size if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                loss = self.criterion(outputs, masks)
                current_loss = loss.item()  # Store before deletion
            
            # Backward pass with memory optimization
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Free intermediate tensors
            del outputs, loss
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()  # Clear gradients
            
            if scheduler:
                scheduler.step()
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            losses.append(current_loss)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        return np.mean(losses)
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        losses = []
        ious = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                with autocast(dtype=torch.bfloat16):
                    # Use gradient-friendly training inference for validation
                    outputs = make_inference_training(
                        images,
                        self.model,
                        inference_mode="whole",
                        decoder_head_type="m2f",
                        rescale_to=masks.shape[-2:],
                        n_output_channels=self.num_classes,
                        output_activation=partial(torch.nn.functional.softmax, dim=1),
                    )
                    
                    # Get class predictions
                    preds = outputs.argmax(dim=1)
                    
                    # Resize for loss calculation
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = nn.functional.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    loss = self.criterion(outputs, masks)
                
                losses.append(loss.item())
                
                # Calculate IoU
                iou = self._calculate_iou(preds, masks)
                ious.append(iou)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
        
        return np.mean(losses), np.mean(ious)
    
    def _calculate_iou(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean IoU"""
        ious = []
        valid = (targets != 255)
        
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid
            true_c = (targets == c) & valid
            
            intersection = (pred_c & true_c).sum().item()
            union = (pred_c | true_c).sum().item()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0
    
    def save_checkpoint(self, epoch: int, val_iou: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_iou': val_iou,
            'num_classes': self.num_classes
        }
        
        # Save regular checkpoint
        path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            logging.info(f"Saved best model: {path} (IoU: {val_iou:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DINOv3 for IDD segmentation (Proper Method)')
    parser.add_argument('--backbone-weights', type=str, required=True,
                        help='Path to backbone checkpoint')
    parser.add_argument('--ade20k-weights', type=str, required=True,
                        help='Path to ADE20K head checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to IDD dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Log configuration
    logging.info("Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    
    # Create finetuner
    finetuner = DINOv3IDDFinetuner(
        backbone_weights=args.backbone_weights,
        ade20k_weights=args.ade20k_weights,
        output_dir=output_dir,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        learning_rate=args.lr
    )
    
    # Create datasets
    train_dataset = IDDSegmentationDataset(
        root_dir=args.data_dir,
        split='train',
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = IDDSegmentationDataset(
        root_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logging.info(f"Dataset: Train {len(train_dataset)}, Val {len(val_dataset)}")
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        finetuner.optimizer,
        T_max=total_steps
    )
    
    # Setup TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    best_iou = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = finetuner.train_epoch(train_loader, epoch, scheduler)
        
        # Validate
        val_loss, val_iou = finetuner.validate(val_loader, epoch)
        
        # Log metrics
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/IoU', val_iou, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Only save best model to save disk space (each checkpoint is ~29GB)
        if is_best:
            finetuner.save_checkpoint(epoch, val_iou, is_best)
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logging.info(f"\nTraining completed! Best IoU: {best_iou:.4f}")
    writer.close()


if __name__ == "__main__":
    main()