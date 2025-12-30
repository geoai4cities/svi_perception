# Complete Pipeline: From Images to Predictions

## 🎯 Overview

This document describes the **complete end-to-end pipeline** from raw street view images to urban perception predictions. The pipeline consists of two main stages:

1. **Feature Extraction** (Optional - pre-computed features provided)
2. **Perception Prediction** (This repository)

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STAGE 1                                 │
│                   Feature Extraction                            │
│                    (Optional/Reference)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Raw Street View Images                                         │
│  - Urban scenes from Place Pulse 2.0                            │
│  - ~111,268 images                                              │
│  - Various cities worldwide                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ↓                           ↓
┌───────────────────────────┐   ┌───────────────────────────┐
│  Semantic Segmentation    │   │   Object Detection        │
│  (ADE20K 150 classes)     │   │   (COCO 80 classes)       │
│                           │   │                           │
│  → 26 features            │   │   → 10 features           │
│  (scene composition)      │   │   (discrete objects)      │
└───────────────────────────┘   └───────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ↓
                ┌─────────────────────────┐
                │  36 or 230 Visual Features      │
                │  per Image               │
                └─────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         STAGE 2                                 │
│                 Perception Prediction                           │
│                   (This Repository)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Data Preparation                                               │
│  - Load 36-feature vectors                                      │
│  - Add perception ratings (beautiful, lively, boring, safe)     │
│  - City-based train/test split                                  │
│  - Delta-based label generation                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Delta Sensitivity Analysis                                     │
│  - 7 threshold values (δ = 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8) │
│  - Binary labels: score >= (median + δ * std)                  │
│  - Multi-class: Low/Medium/High                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┬──────────────┐
                │             │             │              │
                ↓             ↓             ↓              ↓
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Random   │  │   SVM    │  │ XGBoost  │  │ RealMLP  │
        │  Forest  │  │          │  │          │  │    TD    │
        └──────────┘  └──────────┘  └──────────┘  └──────────┘
                │             │             │              │
                └─────────────┴─────────────┴──────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model Evaluation                                               │
│  - F1 Score, Accuracy, Precision, Recall                        │
│  - ROC-AUC, PR-AUC                                              │
│  - Cross-validation                                             │
│  - Feature importance                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Results & Visualization                                        │
│  - 112 trained models (4 × 7 × 4)                              │
│  - 12+ publication figures                                      │
│  - Comprehensive metrics CSV                                    │
│  - Statistical reports                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
              ┌──────────────────────────┐
              │  Perception Predictions   │
              │  - Beautiful: 0-10        │
              │  - Lively: 0-10           │
              │  - Boring: 0-10           │
              │  - Safe: 0-10             │
              └──────────────────────────┘
```

## 🔢 Data Flow Dimensions

### Stage 1: Feature Extraction

**Input**:
- **Images**: 111,268 street view photos
- **Format**: JPEG/PNG (RGB)
- **Resolution**: Variable (resized to 1024×1024)
- **Size**: ~50GB total

**Processing**:
- **Semantic Segmentation**: 150 classes → 26 features
- **Object Detection**: 80 classes → 10 features
- **Time**: 6-10 hours (GPU) or 3-4 days (CPU)

**Output**:
- **Format**: CSV files (4 files, one per perception)
- **Dimensions**: 111,268 rows × 40 columns
  - 4 metadata columns (ID, path, perception, rating)
  - 36 feature columns
- **Size**: ~300MB total (compressed: ~50MB)

### Stage 2: Perception Prediction

**Input**:
- **Features**: 36 per image
- **Labels**: 4 perception attributes (0-10 scale)
- **Samples**: 111,268 total

**Processing**:
- **Experiments**: 112 (4 perceptions × 7 deltas × 4 models)
- **Split**: 80% train / 20% test (city-based or random)
- **Time**: 2-6 hours for all experiments

**Output**:
- **Models**: 112 trained classifiers
- **Metrics**: F1, Accuracy, ROC-AUC, PR-AUC per experiment
- **Figures**: 12+ publication-ready visualizations
- **Size**: ~500MB (models + results)

## 📁 File Structure & Data Flow

```
Project Root
│
├── Feature_Extraction_Code/           # [STAGE 1 - Optional]
│   ├── segmentation_analysis/
│   │   ├── extract_ade20k_150_features.py
│   │   └── output/
│   │       └── *_ade20k_150_features.csv    # 26 features
│   │
│   ├── object_detection_analysis/
│   │   ├── extract_detection_features_80_classes.py
│   │   └── output/
│   │       └── *_detection_features.csv      # 10 features
│   │
│   └── merge_features.py                     # Combine → 36 features
│
├── Input_Data/                        # [STAGE 2 INPUT]
│   └── dinov3_all_classes/            # Example pre-extracted features
│       ├── beautiful_input.xlsx       # 111,268 × 40
│       ├── lively_input.xlsx          # 111,268 × 40
│       ├── boring_input.xlsx          # 111,268 × 40
│       └── safe_input.xlsx            # 111,268 × 40
│
├── run_experiment.sh                  # [STAGE 2 EXECUTION]
│
├── experiments/                       # [STAGE 2 OUTPUT]
│   └── <dataset>/<city>/perception_delta_sensitivity_*/
│       ├── 02_models/                 # 112 .pkl files
│       ├── 03_results/
│       │   ├── metrics/
│       │   │   └── all_results.csv    # 112 rows × 10+ cols
│       │   └── visualizations/        # 12+ figures
│       └── experiment_summary.json
│
└── Feature_Importance/                # [OPTIONAL ANALYSIS]
    ├── outputs/
    │   └── feature_importance_*.csv
    └── saved_models/
```

## 🚀 Quick Start: Complete Pipeline

### Option A: Use Pre-Extracted Features (Recommended)

**Time**: 5 minutes setup + 2-6 hours experiment

```bash
# 1. Setup
cd perception_prediction_gitrepo
source setup_experiment.sh

# 2. Run perception prediction (features already included)
./run_experiment.sh --full --background

# 3. Monitor
./monitor_experiment.sh

# 4. View results
cd experiments/*/03_results/
```

### Option B: Extract Features from Scratch

**Time**: 6-10 hours extraction + 2-6 hours experiment

```bash
# 1. Extract semantic segmentation features
cd Feature_Extraction_Code/segmentation_analysis
./run_ade20k_150_extraction.sh start
# Wait 3-5 hours...

# 2. Extract object detection features (parallel)
cd ../object_detection_analysis
./run_80class_extraction.sh start
# Wait 3-5 hours...

# 3. Merge features
cd ..
python3 merge_features.py \
    --segmentation segmentation_analysis/output/ \
    --detection object_detection_analysis/output/ \
    --output merged_features/

# 4. Format for perception prediction
python3 format_for_perception.py \
    --input merged_features/ \
    --output ../Input_Data/custom_features/

# 5. Run perception prediction
cd ..
export INPUT_DATA_DIR="./Input_Data/custom_features"
./run_experiment.sh --full --background
```

## 🔄 Data Transformations

### 1. Image → Raw Features

**Semantic Segmentation**:
```
Image (1024×1024×3)
    → Segmentation Model
    → Pixel Mask (1024×1024) with 150 classes
    → Count pixels per class
    → Normalize by total pixels
    → Features (150 percentages)
    → Aggregate to 26 semantic categories
```

**Object Detection**:
```
Image (1024×1024×3)
    → Detection Model
    → Bounding Boxes + Labels (N detections × 80 classes)
    → Count objects per class
    → Features (80 counts)
    → Select 10 most relevant urban classes
```

### 2. Features → Labels

**Binary Classification**:
```python
# For each delta value δ
threshold = median(ratings) + δ * std(ratings)
label = 1 if rating >= threshold else 0
```

**Multi-Class Classification**:
```python
low_threshold = median(ratings) - 0.5 * std(ratings)
high_threshold = median(ratings) + 0.5 * std(ratings)

if rating < low_threshold:
    label = "Low"
elif rating < high_threshold:
    label = "Medium"
else:
    label = "High"
```

### 3. Models → Predictions

**Training**:
```python
# For each perception, delta, and model
X_train = features[train_indices]  # 36 features
y_train = labels[train_indices]    # binary or multi-class
model.fit(X_train, y_train)
```

**Prediction**:
```python
# Test set prediction
X_test = features[test_indices]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
```

**Evaluation**:
```python
metrics = {
    'f1': f1_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob[:, 1]),
    'pr_auc': average_precision_score(y_test, y_prob[:, 1])
}
```

## 📊 Data Quality & Validation

### Feature Extraction Validation

```python
# Check feature ranges
assert (seg_features >= 0).all() and (seg_features <= 100).all()
assert (det_features >= 0).all()

# Check completeness
assert not features.isnull().any()
assert len(features) == expected_count

# Check feature consistency
assert list(features.columns) == expected_feature_names
```

### Perception Prediction Validation

```python
# Check train/test split
assert set(train_ids).isdisjoint(set(test_ids))
assert len(train_ids) + len(test_ids) == total_samples

# Check label distribution
print(f"Positive class: {(labels == 1).sum() / len(labels) * 100:.1f}%")

# Check model performance
assert metrics['accuracy'] > 0.5  # Better than random
```

## ⚙️ Configuration Options

### Feature Extraction

```bash
# config.py
IMAGE_DIR = "/path/to/images"
OUTPUT_DIR = "./output"
BATCH_SIZE = 16
USE_GPU = True
SAVE_PROGRESS_EVERY = 100
```

### Perception Prediction

```bash
# run_experiment.sh
INPUT_DATA_DIR="./Input_Data/dinov3_all_classes"
FEATURE_COUNT=36
TEST_CITY_NAME="Mumbai"
USE_CITY_BASED_SPLIT=true

# Run
./run_experiment.sh --full --test-cities Mumbai
```

### Feature Importance

```yaml
# Feature_Importance/experiment_config.yaml
data:
  feature_count: 36
  test_size: 280

perceptions:
  beautiful:
    model: random_forest
    delta: 1.2
```

## 📈 Performance Benchmarks

### Full Pipeline Execution

| Stage | Component | Duration | Resource |
|-------|-----------|----------|----------|
| 1 | Semantic Segmentation | 3-5 hours | GPU (A100) |
| 1 | Object Detection | 3-5 hours | GPU (A100) |
| 1 | Feature Merging | 5-10 min | CPU |
| 2 | Perception Training | 2-6 hours | CPU/GPU |
| 2 | Result Generation | 10-15 min | CPU |
| **Total** | **End-to-End** | **8-16 hours** | **Mixed** |

### Resource Requirements

| Component | CPU | RAM | GPU | Disk |
|-----------|-----|-----|-----|------|
| Feature Extraction | 4+ cores | 16GB | 16GB VRAM | 100GB |
| Perception Prediction | 4+ cores | 8GB | Optional | 5GB |
| Total | 8+ cores | 24GB | 16GB VRAM | 105GB |

## 🔧 Troubleshooting

### Common Issues

**Problem**: Out of GPU memory during feature extraction
```bash
# Solution: Reduce batch size
BATCH_SIZE = 8  # or 4
```

**Problem**: Feature count mismatch
```bash
# Solution: Update feature count
export FEATURE_COUNT=<your_count>
./run_experiment.sh --full
```

**Problem**: City not found in test split
```bash
# Solution: Check valid cities
cat config/cities.yaml
# Or use random split
./run_experiment.sh --full --use-last-280
```

## 📚 Academic Pipeline

For academic publication, we recommend:

### 1. Feature Extraction (Document Thoroughly)
```markdown
- Model: ADE20K Semantic FPN + COCO Faster R-CNN
- Implementation: PyTorch 1.10+
- Hardware: NVIDIA A100 GPU
- Processing time: 6-10 hours
- Code: Available in Feature_Extraction_Code/
```

### 2. Feature Engineering (Report Details)
```markdown
- Input: 150 semantic + 80 detection classes
- Aggregation: Manual grouping based on semantic similarity
- Output: 26 semantic + 10 detection = 36 features
- Normalization: Percentages (0-100) for segmentation, counts for detection
```

### 3. Perception Prediction (Main Contribution)
```markdown
- Models: RF, SVM, XGBoost, RealMLP
- Delta sensitivity: 7 thresholds (0.5-1.8)
- Evaluation: 5-fold cross-validation
- Metrics: F1, Accuracy, ROC-AUC, PR-AUC
```

## 📖 Citation

If you use this complete pipeline, please cite:

```bibtex
@article{yourname2024perception,
  title={Urban Perception Prediction using Delta Sensitivity Analysis},
  author={Your Name and Co-Authors},
  journal={Journal Name},
  year={2024},
  note={Complete pipeline: feature extraction + perception prediction}
}
```

## 🔗 Related Documentation

- **Feature Extraction Details**: [FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md)
- **Perception Prediction**: [README.md](README.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Feature Importance**: [Feature_Importance/feature_importance.md](Feature_Importance/feature_importance.md)

---

**Last Updated**: October 28, 2025
**Version**: 1.0.0
**Pipeline Status**: Production Ready ✅
