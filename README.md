# A Data-Driven Framework for Pedestrian Oriented Route Planning Leveraging Deep Learning and Spatial Perception

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-green.svg)](https://www.sciencedirect.com/science/article/pii/S1569843225005795)

**Official Implementation**

A framework for predicting urban perception attributes (beautiful, lively, boring, safe) using machine learning models with delta sensitivity analysis.

## Authors

- **Pyare Lal Chauhan** ([LinkedIn](https://www.linkedin.com/in/pyarelaldse/))
- **Tanishq Kumar Baswal**
- **Vaibhav Kumar**

**Affiliation**: [GeoAI4Cities Lab](https://geoai4cities.com/), IISER Bhopal | [Netrica.ai](https://www.linkedin.com/company/netrica-ai/)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Feature Extraction                  │
│                       (Optional/Reference)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Raw Street View Images (~111,268 from Place Pulse 2.0)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ↓                           ↓
┌───────────────────────────┐   ┌───────────────────────────┐
│  Semantic Segmentation    │   │   Object Detection        │
│  (ADE20K - 150 classes)   │   │   (COCO - 80 classes)     │
└───────────────────────────┘   └───────────────────────────┘
                └─────────────┬─────────────┘
                              ↓
         ┌────────────────────┴────────────────────┐
         ↓                                         ↓
┌─────────────────────────┐         ┌─────────────────────────┐
│  Full Extraction        │         │  Finetuned DINOv3       │
│  (150 SS + 80 OD = 230) │         │  (36 features)          │
└─────────────────────────┘         └─────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 2: Perception Prediction                  │
│                      (This Repository)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Delta Sensitivity Analysis                                     │
│  7 thresholds (δ = 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ↓          ↓          ↓          ↓
   Random Forest  SVM     XGBoost    RealMLP
                              │
                              ↓
              Perception Predictions & Visualizations
              (112 trained models, 12+ figures)
```

**Pre-extracted features included** - start immediately!

## Installation

```bash
git clone https://github.com/geoai4cities/svi_perception.git
cd svi_perception
source setup_experiment.sh
```

## Quick Start

```bash
# Test run (5-15 min)
./run_experiment.sh --test

# Full experiment (2-6 hours)
./run_experiment.sh --full --background

# Monitor progress
./monitor_experiment.sh
```

## Data

Pre-extracted features in `Input_Data/dinov3_all_classes/`:
- 111,268 images with 36 finetuned DINOv3 features
- 4 perception files: `beautiful_input.xlsx`, `lively_input.xlsx`, `boring_input.xlsx`, `safe_input.xlsx`

**Feature Extraction Options** (see `Feature_Extraction_Code/`):
- **Full extraction**: 230 features (150 ADE20K segmentation + 80 COCO detection)
- **Finetuned DINOv3**: 36 optimized features (used in this repo)

## Results

Results saved in `experiments/<dataset>/<city>/`:
- `03_results/metrics/all_results.csv` - All 112 experiment results
- `03_results/visualizations/` - Publication-ready figures

## Documentation

- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [docs/PIPELINE.md](docs/PIPELINE.md) - Complete pipeline details
- [docs/FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md) - Feature extraction guide

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

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

- **Email**: pyarelal@iiserb.ac.in
- **Lab**: [GeoAI4Cities](https://geoai4cities.com/)
- **Paper**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843225005795)
