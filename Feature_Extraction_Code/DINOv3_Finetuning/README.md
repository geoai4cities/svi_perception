# DINOv3 Finetuning for Semantic Segmentation

This directory contains the finetuning code used to train DINOv3 for 36-feature extraction (26 segmentation + 10 detection aggregated features).

## Overview

The finetuning process adapts the pretrained DINOv3 model to extract optimized features for urban perception prediction.

**Key Achievement**: 42.59% mIoU validation accuracy with head-only finetuning (53,274 trainable parameters out of 7.6B total).

## Architecture

```
DINOv3 ViT-7B/16 Segmentation Model
├── Backbone: DINOv3 ViT-7B/16 (FROZEN)
│   ├── Total Parameters: 7,643,154,240
│   ├── Weights: dinov3_vit7b16_pretrain_lvd1689m.pth
│   └── Status: Frozen during finetuning
├── Decoder: Mask2Former Head (PARTIALLY TRAINABLE)
│   ├── Pretrained: ADE20K → 150 classes
│   ├── Finetuned: Custom → 26 classes
│   ├── Trainable: predictor.class_embed only
│   └── Parameters: 53,274 trainable
└── Output: 26 semantic classes (aggregated from 150)
```

## Directory Structure

```
DINOv3_Finetuning/
├── README.md                        # This file
├── USAGE_GUIDE.md                   # Detailed usage guide
├── idd_dataset.py                   # Dataset loader
├── idd_classes_config_verified.py   # Class configuration
├── inference_finetuned.py           # Inference script
└── scripts/
    ├── train_dinov3_idd_proper.py   # Core training logic
    ├── make_inference_training.py   # Gradient-friendly inference
    └── train_20epoch_recommended.sh # Training runner script
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- DINOv3 repository (https://github.com/facebookresearch/dinov3)
- NVIDIA GPU with 40GB+ VRAM (A100 recommended)

## Weights Required

Download from Meta's DINOv3 repository (requires form):

```
weights/dinov3/
├── backbones/
│   └── dinov3_vit7b16_pretrain_lvd1689m.pth
└── adapters/segmentation/
    └── dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
```

## Quick Start

### 1. Setup Environment

```bash
# Follow DINOv3 setup
# https://github.com/facebookresearch/dinov3

# Or use setup script from parent directory
cd ../
./setup_environment.sh
conda activate dinov3_env
```

### 2. Prepare Dataset

Prepare your segmentation dataset in the following structure:
```
data/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### 3. Train Model

```bash
cd scripts/

# Edit paths in train_20epoch_recommended.sh
# Then run:
./train_20epoch_recommended.sh
```

### 4. Run Inference

```bash
python inference_finetuned.py \
    --checkpoint checkpoints/best_model.pth \
    --image /path/to/image.jpg \
    --output ./outputs
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | DINOv3 ViT-7B/16 | Frozen |
| Image Size | 512×512 | Optimal for 40GB GPU |
| Batch Size | 1 | Memory constrained |
| Learning Rate | 1e-4 | AdamW optimizer |
| Epochs | 20 | Early stopping at patience=10 |
| Mixed Precision | bfloat16 | Essential for 7B model |

## Key Technical Innovations

### 1. Gradient-Preserving Inference

The original DINOv3 `make_inference()` uses `torch.no_grad()` internally, breaking gradient flow. Our `make_inference_training.py` bypasses this:

```python
# Direct model call preserves gradients
outputs = segmentation_model(x_resized)  # Instead of .predict()
```

### 2. Head-Only Finetuning

Only the final classification layer is trained:
- Freezes 7.6B backbone parameters
- Trains only 53,274 parameters (0.0007%)
- Achieves 42.59% mIoU in 5 epochs

### 3. Memory Optimization

- Mixed precision (bfloat16)
- Gradient accumulation
- Periodic cache clearing
- Optimized image size (512×512)

## Expected Performance

| Training Duration | Validation mIoU | Use Case |
|-------------------|-----------------|----------|
| 5 epochs (~4h) | ~42% | Quick validation |
| 20 epochs (~16h) | ~57-63% | Production use |
| 30 epochs (~24h) | ~62-68% | Maximum accuracy |

## Feature Aggregation

After finetuning, the 150 ADE20K classes are aggregated to 26 semantic features:

```python
aggregation_map = {
    'building': ['building', 'house', 'skyscraper', 'shop'],
    'tree': ['tree', 'palm', 'plant'],
    'road': ['road', 'highway', 'street'],
    # ... etc for all 26 categories
}
```

Combined with 10 selected detection features → **36 total features**.

## Citation

```bibtex
@article{chauhan2025data,
  title={A Data-Driven framework for pedestrian oriented route planning leveraging deep learning and spatial perception},
  author={Chauhan, Pyare Lal and Baswal, Tanishq Kumar and Kumar, Vaibhav},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={144},
  pages={104932},
  year={2025},
  publisher={Elsevier}
}
```

## References

- **DINOv3**: https://github.com/facebookresearch/dinov3
- **Mask2Former**: Universal image segmentation architecture
- **ADE20K**: http://groups.csail.mit.edu/vision/datasets/ADE20K/
