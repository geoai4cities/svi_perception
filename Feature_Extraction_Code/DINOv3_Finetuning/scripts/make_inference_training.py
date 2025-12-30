"""
Training-compatible version of make_inference that preserves gradients
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple


def make_inference_training(
    x: torch.Tensor,
    segmentation_model: torch.nn.Module,
    inference_mode: str = "whole",
    decoder_head_type: str = "m2f",
    rescale_to=(512, 512),
    n_output_channels: int = 26,
    output_activation: Callable | None = None,
):
    """
    Training-compatible version of make_inference that preserves gradients.
    
    Args:
        x: input image tensor [B, C, H, W]
        segmentation_model: DINOv3 segmentation model
        inference_mode: "whole" inference mode
        decoder_head_type: "m2f" for Mask2Former
        rescale_to: output size tuple
        n_output_channels: number of output classes
        output_activation: optional activation function
        
    Returns:
        Dense segmentation predictions [B, C, H, W]
    """
    assert inference_mode == "whole", "Only 'whole' mode supported for training"
    
    # Resize input to standard size
    x_resized = F.interpolate(
        x,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    
    # Forward pass through the model (preserves gradients)
    # Call the model directly instead of .predict() to maintain gradients
    outputs = segmentation_model(x_resized)
    
    # Handle Mask2Former outputs
    if decoder_head_type == "m2f":
        if isinstance(outputs, dict):
            mask_pred = outputs["pred_masks"]    # [B, N, H, W]
            mask_cls = outputs["pred_logits"]    # [B, N, C]
            
            # Convert to dense segmentation (same as original make_inference)
            # Apply softmax to class predictions (keep all classes - no background removal)
            mask_cls = F.softmax(mask_cls, dim=-1)  # Keep all classes for IDD
            
            # Apply sigmoid to mask predictions
            mask_pred = mask_pred.sigmoid()
            
            # Combine class and mask predictions
            pred = torch.einsum("bqc,bqhw->bchw", 
                              mask_cls.to(torch.float), 
                              mask_pred.to(torch.float))
        else:
            # Direct tensor output
            pred = outputs
    else:
        pred = outputs
    
    # Rescale to target size
    if pred.shape[-2:] != rescale_to:
        pred = F.interpolate(
            pred,
            size=rescale_to,
            mode="bilinear",
            align_corners=False,
        )
    
    # Apply output activation if specified
    if output_activation:
        pred = output_activation(pred)
        
    return pred