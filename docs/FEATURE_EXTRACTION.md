# Feature Extraction Pipeline

## 📋 Overview

This document describes the complete feature extraction pipeline used to convert urban street view images into numerical features for perception prediction. The pipeline uses **DINOv3** (Facebook Research) with two extraction heads:

1. **Semantic Segmentation** (ADE20K 150 classes)
2. **Object Detection** (COCO 80 classes)

## 🎯 Feature Extraction Options

| Option | Segmentation | Detection | Total | Description |
|--------|-------------|-----------|-------|-------------|
| **Full Extraction** | 150 features | 80 features | **230 features** | All class outputs |
| **Finetuned DINOv3** | 26 features | 10 features | **36 features** | Optimized for perception (used in paper) |

## 🎯 Complete Pipeline Flow

```
Raw Images (Street View Photos)
         ↓
    ┌─────────────────────────────────────────────┐
    │   DINOv3 Feature Extraction Pipeline        │
    │   (ViT-7B/16 backbone)                      │
    │                                             │
    │  ┌─────────────────────────────┐            │
    │  │  Semantic Segmentation      │            │
    │  │  (ADE20K Mask2Former head)  │            │
    │  │  - Building, Sky, Road...   │            │
    │  │  → 150 classes              │            │
    │  └─────────────────────────────┘            │
    │               +                              │
    │  ┌─────────────────────────────┐            │
    │  │  Object Detection           │            │
    │  │  (COCO DETR head)           │            │
    │  │  - Car, Person, Sign...     │            │
    │  │  → 80 classes               │            │
    │  └─────────────────────────────┘            │
    └─────────────────────────────────────────────┘
         ↓
    ┌────────────────┬─────────────────┐
    │                │                 │
    ↓                ↓
Full Extraction   Finetuned DINOv3
(230 features)    (36 features)
         ↓
    ┌────────────────────────────────────┐
    │   Perception Prediction            │
    │   (This Repository)                │
    │   - Beautiful, Lively,             │
    │   - Boring, Safe                   │
    └────────────────────────────────────┘
         ↓
    Perception Predictions
```

## 🔧 Feature Extraction Methods

### Method 1: Semantic Segmentation (ADE20K)

**Purpose**: Extract pixel-level understanding of scene composition

**Model**: DINOv3 ViT-7B/16 + ADE20K Mask2Former head
- Architecture: Vision Transformer (7B parameters)
- Pre-trained backbone + ADE20K segmentation head
- Outputs per-pixel class probabilities (150 classes)
- Weights: `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`

**Process**:
1. Input: Street view image (RGB, resized to 896×896)
2. Run DINOv3 segmentation model
3. Extract pixel counts for each of 150 classes
4. Normalize by total pixels → percentage coverage
5. Option: Aggregate related classes into 26 semantic categories

**Full Output (150 features)**: All ADE20K class percentages

**Aggregated Output (26 features)**:
```
building, sky, tree, road, grass, sidewalk, person, earth,
plant, car, water, fence, wall, pole, door, chair, sand,
floor, sign, field, ground, mountain, house, sea, rug, ceiling
```

**Key Characteristics**:
- Captures overall scene composition
- Spatial coverage information
- Environment type (urban, natural, mixed)
- Infrastructure density

### Method 2: Object Detection (COCO)

**Purpose**: Detect and count discrete objects in the scene

**Model**: DINOv3 ViT-7B/16 + COCO DETR head
- Architecture: Vision Transformer (7B parameters) + DETR detection head
- Pre-trained backbone + COCO detection head
- Outputs bounding boxes and class labels (80 classes)
- Weights: `dinov3_vit7b16_coco_detr_head-b0235ff7.pth`

**Process**:
1. Input: Street view image (RGB, resized to 896×896)
2. Run DINOv3 detection model
3. Count instances of each object class
4. Normalize counts (log transformation or binning)
5. Option: Select 10 most relevant urban objects

**Full Output (80 features)**: All COCO class object counts

**Selected Output (10 features)**:
```
car_detection, traffic_light, person_detection, bicycle,
truck, bird, bench, dog, boat, motorcycle
```

**Key Characteristics**:
- Counts discrete objects
- Transportation infrastructure
- Human activity indicators
- Urban furniture and amenities

## 📊 Feature Aggregation

### Option 1: Full Features (230)
- **150 Semantic Segmentation features** (all ADE20K classes)
- **80 Object Detection features** (all COCO classes)

### Option 2: Finetuned Features (36)
- **26 Semantic Segmentation features** (aggregated)
- **10 Object Detection features** (selected urban objects)

### Feature Normalization
- **Segmentation**: Pixel percentages (0-100%)
- **Detection**: Object counts (0-N, often log-transformed)

### Feature Naming Convention
```
<class_name>                  # Segmentation features
<class_name>_detection        # Object detection features
```

Examples:
- `car` → Segmentation (% of pixels)
- `car_detection` → Detection (count of cars)
- `person` → Segmentation (% of pixels)
- `person_detection` → Detection (count of people)

## 🔬 Technical Implementation

### Included Feature Extraction Code

The repository includes **reference implementations** in `Feature_Extraction_Code/`:

```
Feature_Extraction_Code/
├── segmentation_analysis/
│   ├── run_ade20k_150_extraction.sh         # Runner script
│   ├── extract_ade20k_150_features.py       # Main extraction
│   ├── config.py                            # Configuration
│   ├── visualize_ade20k_150_test_results.py # Visualization
│   └── README.md                            # Documentation
│
├── object_detection_analysis/
│   ├── run_80class_extraction.sh            # Runner script
│   ├── extract_detection_features_80_classes.py
│   ├── config_80_classes.py                 # Configuration
│   ├── visualize_coco80_test_results.py     # Visualization
│   └── README.md                            # Documentation
│
└── docs/
    └── USAGE.md                             # Usage instructions
```

## 🚀 Using the Provided Extraction Code

### Prerequisites
- Python 3.8+
- PyTorch + torchvision
- PIL/Pillow
- NumPy, Pandas
- Pre-trained models (downloaded automatically)

### Quick Start

#### 1. Semantic Segmentation Extraction
```bash
cd Feature_Extraction_Code/segmentation_analysis

# Test mode (few images)
python3 extract_ade20k_150_features.py --test

# Full extraction (all images)
./run_ade20k_150_extraction.sh start

# Monitor progress
./run_ade20k_150_extraction.sh status

# Stop extraction
./run_ade20k_150_extraction.sh stop
```

#### 2. Object Detection Extraction
```bash
cd Feature_Extraction_Code/object_detection_analysis

# Test mode (few images)
python3 extract_detection_features_80_classes.py --test

# Full extraction (all images)
./run_80class_extraction.sh start

# Monitor progress
./run_80class_extraction.sh status

# Stop extraction
./run_80class_extraction.sh stop
```

### Configuration

Edit `config.py` (segmentation) or `config_80_classes.py` (detection):

```python
# Input images
IMAGE_DIR = "/path/to/street/view/images"
MANIFEST_FILE = "/path/to/image_list.csv"

# Output
OUTPUT_DIR = "./output"
FEATURE_OUTPUT_FILE = "extracted_features.csv"

# Processing
BATCH_SIZE = 16
NUM_WORKERS = 4
USE_GPU = True

# Resume capability
SAVE_PROGRESS_EVERY = 100  # Save every N images
```

### Output Format

Both extractors produce CSV files with this structure:

```csv
unique_id,image_path,perception,rating_score,feature1,feature2,...,featureN
PP_001,/path/img1.jpg,beautiful,7.5,25.3,15.2,...,0.8
PP_002,/path/img2.jpg,lively,6.2,30.1,12.5,...,1.2
...
```

**Columns**:
- `unique_id`: Unique image identifier
- `image_path`: Path to source image
- `perception`: Perception attribute (if available)
- `rating_score`: Human rating (if available)
- `feature1...featureN`: Extracted features

## 🔄 Using Your Own Feature Extractor

You can use **any feature extraction method** as long as the output format matches the expected structure.

### Requirements

1. **CSV Format**: Features must be in CSV format
2. **Required Columns**:
   ```
   unique_id, image_path, perception, rating_score, [feature_columns]
   ```
3. **Feature Count**: Must match expected count (default: 36)
   - Can be changed via `FEATURE_COUNT` environment variable
4. **Feature Names**: Must be consistent across files

### Supported Alternative Extractors

#### Vision Transformers
- **DINOv3**: Self-supervised vision features - [GitHub](https://github.com/facebookresearch/dinov3) (**used in this paper**)
- **DINOv2**: Self-supervised vision features
- **CLIP**: OpenAI CLIP embeddings
- **ViT**: Vision Transformer features

> **Note**: Complete feature extraction code using DINOv3 is included in `Feature_Extraction_Code/`.

#### Semantic Segmentation
- **Mask2Former**: Advanced segmentation
- **SegFormer**: Efficient segmentation
- **SAM**: Segment Anything Model

#### Object Detection
- **YOLO v8/v9**: Modern object detection
- **DETR**: Detection Transformer
- **Detectron2**: Facebook's detection framework

#### Scene Understanding
- **PlacesCNN**: Scene classification features
- **SUN**: Scene understanding features
- **ImageNet**: Pre-trained CNN features

### Integration Steps

1. **Extract Features**:
   ```bash
   python your_extractor.py --input images/ --output features.csv
   ```

2. **Verify Format**:
   ```python
   import pandas as pd
   df = pd.read_csv('features.csv')
   print(df.columns)  # Check column names
   print(df.shape)    # Check dimensions
   ```

3. **Place in Input_Data**:
   ```bash
   mkdir -p Input_Data/my_custom_features
   cp beautiful_features.csv Input_Data/my_custom_features/beautiful_input.xlsx
   # Convert to Excel if needed, or keep as CSV
   ```

4. **Update Configuration**:
   ```bash
   export INPUT_DATA_DIR="./Input_Data/my_custom_features"
   export FEATURE_COUNT=<your_feature_count>
   ./run_experiment.sh --full
   ```

## 📐 Feature Engineering Details

### Semantic Segmentation Feature Engineering

**Raw Output**: 150 ADE20K classes
**Final Output**: 26 aggregated features

**Aggregation Strategy**:
```python
# Example: Aggregate transportation
transportation = building_pixels + road_pixels + sidewalk_pixels

# Example: Aggregate nature
nature = tree_pixels + grass_pixels + plant_pixels + flower_pixels
```

**Why Aggregate?**
- Reduces dimensionality (150 → 26)
- Groups semantically similar classes
- Reduces noise and variability
- Improves model interpretability

### Object Detection Feature Engineering

**Raw Output**: 80 COCO classes with counts
**Final Output**: 10 selected urban features

**Selection Criteria**:
1. Urban relevance (cars, traffic lights, etc.)
2. Frequency of appearance (> 1% of images)
3. Perception correlation (empirically determined)
4. Class distinctiveness (avoid redundancy)

**Count Normalization**:
```python
# Option 1: Log transformation
normalized_count = np.log1p(raw_count)

# Option 2: Binning
normalized_count = np.minimum(raw_count, max_threshold)

# Option 3: Standardization
normalized_count = (count - mean) / std
```

## 📊 Feature Statistics

### Typical Feature Ranges

**Segmentation Features (%):**
- Building: 10-40% (urban scenes)
- Sky: 5-30% (weather dependent)
- Road: 15-35% (street view)
- Tree: 5-25% (greenery)
- Car: 5-20% (traffic)

**Detection Features (counts):**
- Cars: 0-30 per image
- People: 0-20 per image
- Traffic lights: 0-5 per image
- Bicycles: 0-5 per image

### Feature Correlation

High correlation pairs (often correlated):
- `building` ↔ `road` (urban scenes)
- `sky` ↔ `tree` (open spaces)
- `car` ↔ `car_detection` (different measures of same)

Low correlation pairs (independent):
- `water` ↔ `traffic_light`
- `mountain` ↔ `bicycle`

## 🔍 Quality Assurance

### Validation Checks

1. **Feature Range Check**:
   ```python
   # Segmentation: 0-100%
   assert (features >= 0).all() and (features <= 100).all()

   # Detection: ≥ 0
   assert (features >= 0).all()
   ```

2. **Missing Values**:
   ```python
   assert not features.isnull().any()
   ```

3. **Feature Consistency**:
   ```python
   # Check feature count
   assert len(features.columns) == expected_feature_count
   ```

### Visualization

Both extractors include visualization tools:

```bash
# Visualize segmentation results
python3 visualize_ade20k_150_test_results.py

# Visualize detection results
python3 visualize_coco80_test_results.py
```

**Output**: Side-by-side comparison images showing:
- Original image
- Segmentation mask (colored)
- Detection boxes (with labels)
- Feature statistics

## ⚙️ Performance Optimization

### GPU Acceleration
```python
# Enable GPU
USE_GPU = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch processing
BATCH_SIZE = 16  # Adjust based on GPU memory
```

### Parallel Processing
```bash
# Run both extractors in parallel
./Feature_Extraction_Code/segmentation_analysis/run_ade20k_150_extraction.sh start &
./Feature_Extraction_Code/object_detection_analysis/run_80class_extraction.sh start &

# Monitor both
watch -n 5 'ps aux | grep extract'
```

### Resume Capability
Both extractors support resume after interruption:
- Progress saved every 100 images
- Automatic detection of partial results
- Skip already processed images

### Processing Speed

**Typical Performance** (GPU: NVIDIA A100):
- Segmentation: ~5-10 images/sec
- Detection: ~3-8 images/sec
- Total time for 111,268 images: 6-10 hours

**CPU Performance**:
- Segmentation: ~0.5-1 images/sec
- Detection: ~0.3-0.7 images/sec
- Total time for 111,268 images: 3-4 days

## 📚 Reference Implementation

The provided code uses DINOv3 from Facebook Research:

### DINOv3 Backbone
- **Paper**: DINOv3 (Facebook Research)
- **Model**: Vision Transformer ViT-7B/16
- **Backbone Weights**: `dinov3_vit7b16_pretrain_lvd1689m.pth`
- **Repository**: https://github.com/facebookresearch/dinov3

### Segmentation Head
- **Model**: Mask2Former with ADE20K training
- **Dataset**: ADE20K (150 semantic classes)
- **Weights**: `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`
- **Reference**: "Scene Parsing through ADE20K Dataset" (Zhou et al., CVPR 2017)

### Detection Head
- **Model**: DETR with COCO training
- **Dataset**: MS COCO (80 object classes)
- **Weights**: `dinov3_vit7b16_coco_detr_head-b0235ff7.pth`
- **Reference**: "Microsoft COCO: Common Objects in Context" (Lin et al., ECCV 2014)

## 🔗 Integration with Perception Prediction

### Workflow

1. **Extract Features** (this document):
   ```bash
   cd Feature_Extraction_Code
   ./run_all_extractions.sh
   ```

2. **Merge Features**:
   ```bash
   python merge_features.py \
       --segmentation segmentation_analysis/output/ \
       --detection object_detection_analysis/output/ \
       --output merged_features.csv
   ```

3. **Format for Perception**:
   ```bash
   python format_for_perception.py \
       --input merged_features.csv \
       --output Input_Data/my_features/
   ```

4. **Run Perception Prediction** (main repository):
   ```bash
   export INPUT_DATA_DIR="./Input_Data/my_features"
   ./run_experiment.sh --full
   ```

## 🆘 Troubleshooting

### Common Issues

**Issue**: Out of GPU memory
```bash
# Solution: Reduce batch size
BATCH_SIZE = 8  # or 4
```

**Issue**: Models not downloading
```bash
# Solution: Manual download
wget <model_url>
mv model.pth ~/.cache/torch/hub/checkpoints/
```

**Issue**: Missing dependencies
```bash
# Solution: Install requirements
pip install torch torchvision pillow pandas numpy
```

**Issue**: Extraction too slow
```bash
# Solution: Use GPU or reduce image resolution
USE_GPU = True
IMAGE_RESIZE = (512, 512)  # Smaller than 1024x1024
```

## 📝 Citation

If you use the provided feature extraction code, please cite:

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

## 📖 Additional Resources

- **ADE20K Dataset**: http://groups.csail.mit.edu/vision/datasets/ADE20K/
- **MS COCO Dataset**: https://cocodataset.org/
- **PyTorch Vision Models**: https://pytorch.org/vision/stable/models.html
- **Main Repository README**: [README.md](README.md)

---

**Note**: The feature extraction code is provided as a **reference implementation**. Users are encouraged to experiment with different feature extractors and compare results. The perception prediction pipeline is **modular** and works with any properly formatted feature set.

**Last Updated**: October 28, 2025
**Version**: 1.0.0
