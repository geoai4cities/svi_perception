# Feature Extraction Code (Reference Implementation)

## 📋 Overview

This directory contains **reference implementations** of the feature extraction pipeline used to convert urban street view images into numerical features for perception prediction.

**Note**: This code is **optional**. The main repository already includes pre-extracted features in `Input_Data/dinov3_all_classes/`. Use this code if you want to:
- Extract features from your own images
- Understand the feature extraction process
- Reproduce the feature extraction from scratch
- Experiment with different extraction methods

## 📂 Directory Structure

```
Feature_Extraction_Code/
├── setup_environment.sh            # Environment setup script
├── segmentation_analysis/          # Semantic segmentation (ADE20K)
│   ├── 00_make_manifests.py        # Create image file lists
│   ├── run_ade20k_150_extraction.sh
│   ├── extract_ade20k_150_features.py
│   ├── config.py
│   ├── visualize_ade20k_150_test_results.py
│   ├── visualize_segmentation_test_results.py
│   └── README.md
│
├── object_detection_analysis/      # Object detection (COCO)
│   ├── run_80class_extraction.sh
│   ├── extract_detection_features_80_classes.py
│   ├── config_80_classes.py
│   ├── visualize_coco80_test_results.py
│   └── README.md
│
├── DINOv3/                         # DINOv3 repository (clone from Meta)
│
├── DINOv3_Finetuning/              # Finetuning code for 36 features
│   ├── scripts/                    # Training scripts
│   │   ├── train_dinov3_idd_proper.py
│   │   ├── make_inference_training.py
│   │   └── train_20epoch_recommended.sh
│   ├── idd_dataset.py              # Dataset loader
│   ├── idd_classes_config_verified.py
│   ├── inference_finetuned.py
│   ├── USAGE_GUIDE.md
│   └── README.md
│
└── README.md                        # This file
```

## Environment Setup

For DINOv3 feature extraction, follow: https://github.com/facebookresearch/dinov3

```bash
# Run setup script
./setup_environment.sh

# Activate environment
conda activate dinov3_env
```

## Download Weights

Download pre-trained weights from Meta's official DINOv3 repository:
https://github.com/facebookresearch/dinov3#pretrained-models

**Required weights:**
- **Backbone**: `dinov3_vit7b16_pretrain_lvd1689m.pth` (ViT-7B/16)
- **Segmentation Head**: `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth` (ADE20K Mask2Former)
- **Detection Head**: `dinov3_vit7b16_coco_detr_head-b0235ff7.pth` (COCO DETR)

**Directory structure:**
```
weights/dinov3/
├── backbones/
│   └── dinov3_vit7b16_pretrain_lvd1689m.pth
└── adapters/
    ├── segmentation/
    │   └── dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
    └── detection/
        └── dinov3_vit7b16_coco_detr_head-b0235ff7.pth
```

> **Note**: You need to fill out Meta's access form to download the weights.

## 🎯 What This Code Does

### Semantic Segmentation (ADE20K)
- **Input**: Street view images
- **Model**: DINOv3 with ADE20K Mask2Former head (150 classes)
- **Process**: Per-pixel classification
- **Output**: 150 semantic features (class percentages)
- **Examples**: building, sky, road, tree, car, sidewalk, etc.

### Object Detection (COCO)
- **Input**: Street view images
- **Model**: DINOv3 with COCO DETR head (80 classes)
- **Process**: Object bounding box detection
- **Output**: 80 object count features
- **Examples**: car, person, traffic_light, bicycle, truck, etc.

### Feature Extraction Options

**Option 1: Full Extraction (230 features)**
- 150 semantic segmentation features (all ADE20K classes)
- 80 object detection features (all COCO classes)
- Best for comprehensive analysis

**Option 2: Finetuned DINOv3 (36 features)**
- 26 aggregated semantic features
- 10 selected detection features
- Optimized for perception prediction (used in paper)

### Output Format
- **Format**: CSV files ready for perception prediction
- **Size**: ~300MB for 111,268 images

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Install dependencies
pip install torch torchvision pillow pandas numpy tqdm
```

### Option 1: Test Mode (Quick Validation)

```bash
# Test semantic segmentation (5 images)
cd segmentation_analysis
python3 extract_ade20k_150_features.py --test

# Test object detection (5 images)
cd ../object_detection_analysis
python3 extract_detection_features_80_classes.py --test
```

**Time**: 1-2 minutes
**Output**: Small test CSV files with visualizations

### Option 2: Full Extraction

```bash
# Full semantic segmentation (~111K images)
cd segmentation_analysis
./run_ade20k_150_extraction.sh start
# Wait 3-5 hours (GPU) or 1-2 days (CPU)

# Full object detection (can run in parallel)
cd ../object_detection_analysis
./run_80class_extraction.sh start
# Wait 3-5 hours (GPU) or 1-2 days (CPU)
```

**Time**: 6-10 hours (GPU) or 3-4 days (CPU)
**Output**: Complete feature CSV files

### Monitor Progress

```bash
# Check segmentation progress
cd segmentation_analysis
./run_ade20k_150_extraction.sh status

# Check detection progress
cd object_detection_analysis
./run_80class_extraction.sh status

# Real-time monitoring
./run_ade20k_150_extraction.sh monitor
```

## ⚙️ Configuration

### Segmentation Configuration

Edit `segmentation_analysis/config.py`:

```python
# Input
IMAGE_DIR = "/path/to/street/view/images"
MANIFEST_FILE = "/path/to/image_list.csv"

# Output
OUTPUT_DIR = "./output"
FEATURE_OUTPUT_FILE = "ade20k_150_features.csv"

# Processing
BATCH_SIZE = 16        # GPU memory dependent
NUM_WORKERS = 4        # CPU cores
USE_GPU = True         # Set False for CPU-only
DEVICE = "cuda"        # or "cpu"

# Resume
SAVE_PROGRESS_EVERY = 100  # Save every N images
RESUME_FROM_PARTIAL = True
```

### Detection Configuration

Edit `object_detection_analysis/config_80_classes.py`:

```python
# Similar structure to segmentation
# Adjust BATCH_SIZE based on GPU memory
# Supports same resume and progress features
```

## 📊 Output Format

Both extractors produce CSV files with this structure:

```csv
unique_id,image_path,perception,rating_score,feature1,feature2,...,featureN
PP_001,/path/img1.jpg,beautiful,7.5,25.3,15.2,...,0.8
PP_002,/path/img2.jpg,lively,6.2,30.1,12.5,...,1.2
...
```

**Columns**:
- `unique_id`: Unique identifier (e.g., Place Pulse ID)
- `image_path`: Path to source image
- `perception`: Perception attribute (beautiful/lively/boring/safe)
- `rating_score`: Human rating score (0-10 scale)
- `feature1...featureN`: Extracted visual features

## 🔄 Integration with Main Repository

### Step 1: Extract Features

```bash
# Run both extractors
cd Feature_Extraction_Code
./run_all_extractions.sh  # (you can create this wrapper)
```

### Step 2: Verify Output

```python
import pandas as pd

# Check segmentation features
seg_df = pd.read_csv('segmentation_analysis/output/features.csv')
print(f"Segmentation: {seg_df.shape}")  # (111268, 30)

# Check detection features
det_df = pd.read_csv('object_detection_analysis/output/features.csv')
print(f"Detection: {det_df.shape}")  # (111268, 14)
```

### Step 3: Merge Features (if needed)

```python
# Simple merge by unique_id
merged = pd.merge(seg_df, det_df, on='unique_id')
# Result: 36 features + 4 metadata = 40 columns
```

### Step 4: Place in Input_Data

```bash
# Copy to main repository
cp merged_features/beautiful_features.csv \
   ../Input_Data/my_features/beautiful_input.xlsx

# Repeat for lively, boring, safe
```

### Step 5: Run Perception Prediction

```bash
cd ..
export INPUT_DATA_DIR="./Input_Data/my_features"
export FEATURE_COUNT=36
./run_experiment.sh --full
```

## 📈 Performance

### Typical Performance (GPU: NVIDIA A100)

| Component | Images/sec | Total Time (111K images) |
|-----------|-----------|--------------------------|
| Segmentation | 5-10 | 3-5 hours |
| Detection | 3-8 | 3-5 hours |
| **Total** | **-** | **6-10 hours** |

### CPU Performance (Intel Xeon or equivalent)

| Component | Images/sec | Total Time (111K images) |
|-----------|-----------|--------------------------|
| Segmentation | 0.5-1 | 1-2 days |
| Detection | 0.3-0.7 | 1-2 days |
| **Total** | **-** | **3-4 days** |

### Disk Usage

- Input images: ~50GB
- Output features: ~300MB (CSV)
- Temporary files: ~1GB
- **Total**: ~51GB

## 🔧 Troubleshooting

### Issue: Out of GPU Memory

```bash
# Solution: Reduce batch size
BATCH_SIZE = 8  # or 4, or 2
```

### Issue: Models not downloading

```bash
# Solution: Manual download from PyTorch
# Models are downloaded automatically from torchvision.models
# If blocked, use manual download:
python3 -c "import torchvision.models as models; \
    models.segmentation.fcn_resnet50(pretrained=True); \
    models.detection.fasterrcnn_resnet50_fpn(pretrained=True)"
```

### Issue: Missing dependencies

```bash
# Solution: Install all requirements
pip install torch torchvision pillow pandas numpy tqdm

# For visualization
pip install matplotlib seaborn
```

### Issue: Extraction too slow

```bash
# Solution 1: Use GPU
USE_GPU = True

# Solution 2: Reduce image resolution
IMAGE_RESIZE = (512, 512)  # Instead of (1024, 1024)

# Solution 3: Increase batch size (if memory allows)
BATCH_SIZE = 32
```

### Issue: Process interrupted

```bash
# Solution: Resume automatically
# Both extractors support resume capability
./run_ade20k_150_extraction.sh start
# Will automatically skip processed images
```

## 🎨 Visualization

Both extractors include visualization tools:

```bash
# Visualize segmentation results
cd segmentation_analysis
python3 visualize_ade20k_150_test_results.py

# Visualize detection results
cd object_detection_analysis
python3 visualize_coco80_test_results.py
```

**Output**: Image grids showing:
- Original image
- Segmentation mask (colored by class)
- Detection boxes with labels
- Feature statistics

## 📚 Technical Details

### Segmentation Model
- **Architecture**: DINOv3 ViT-7B/16 + ADE20K Mask2Former head
- **Training Data**: ADE20K dataset (20K+ images, 150 classes)
- **Input Size**: 896×896 (resized)
- **Output**: Per-pixel class probabilities (150 classes)
- **Weights**: `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`

### Detection Model
- **Architecture**: DINOv3 ViT-7B/16 + COCO DETR head
- **Training Data**: MS COCO dataset (118K+ images, 80 classes)
- **Input Size**: 896×896 (resized)
- **Output**: Bounding boxes + class labels + confidence scores
- **Weights**: `dinov3_vit7b16_coco_detr_head-b0235ff7.pth`

### Feature Engineering

**Full Extraction (230 features):**
- All 150 ADE20K segmentation classes (pixel percentages)
- All 80 COCO detection classes (object counts)

**Finetuned DINOv3 (36 features):**

*Segmentation (150 → 26 features):*
```python
# Aggregate semantically similar classes
aggregation_map = {
    'building': ['building', 'house', 'skyscraper', 'shop'],
    'tree': ['tree', 'palm', 'plant'],
    # ... etc for all 26 categories
}
```

*Detection (80 → 10 features):*
```python
# Select most relevant urban objects
selected_classes = [
    'car', 'traffic_light', 'person', 'bicycle',
    'truck', 'bird', 'bench', 'dog', 'boat', 'motorcycle'
]
```

## 🔗 Additional Resources

- **Main Documentation**: [../docs/FEATURE_EXTRACTION.md](../docs/FEATURE_EXTRACTION.md)
- **Complete Pipeline**: [../docs/PIPELINE.md](../docs/PIPELINE.md)
- **Perception Prediction**: [../README.md](../README.md)

### Model References

- **ADE20K**: http://groups.csail.mit.edu/vision/datasets/ADE20K/
- **MS COCO**: https://cocodataset.org/
- **PyTorch Models**: https://pytorch.org/vision/stable/models.html

## 📝 Notes

1. **Optional Code**: This is reference code. Pre-extracted features are included in the main repository.

2. **Flexibility**: You can use any feature extraction method. See [FEATURE_EXTRACTION.md](../docs/FEATURE_EXTRACTION.md) for alternatives.

3. **Resume Capability**: Both extractors support resume after interruption. Progress is saved every 100 images.

4. **Parallel Execution**: You can run segmentation and detection in parallel to save time.

5. **Testing**: Always test on a few images first before running full extraction.

## 🆘 Support

For issues with feature extraction:
1. Check configuration files
2. Verify GPU availability: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Test on small batch first
4. Check logs in `logs/` directory
5. Refer to `README.md` in each subdirectory

For perception prediction issues:
- See main [README.md](../README.md)
- Use pre-extracted features in `Input_Data/`

---

**Last Updated**: December 2025
**Version**: 1.0.0
**Status**: Reference Implementation
