# Paper Information

## Title
**A Data-Driven Framework for Pedestrian Oriented Route Planning Leveraging Deep Learning and Spatial Perception**

## Authors
1. **Pyare Lal Chauhan** 
   - Affiliation: GeoAI4Cities Lab, Data Science and Engineering, IISER Bhopal
   - Organization: Netrica.ai and IISER Bhopal
   - Email: [pyarelal@iiserb.ac.in]
   - ORCID: [Your ORCID]
   - Role: Lead researcher, methodology design, implementation

2. **Tanishq Kumar Baswal**
   - Affiliation: GeoAI4Cities Lab, Data Science and Engineering, IISER Bhopal
   - Email: [Email]

3. **Vaibhav Kumar**
   - Affiliation: GeoAI4Cities Lab, Data Science and Engineering, IISER Bhopal
   - Email: [Email]
   - ORCID: [ORCID]

## Research Group
**GeoAI4Cities Lab**
Data Science and Engineering
Indian Institute of Science Education and Research (IISER) Bhopal
Madhya Pradesh, India

**[Netrica.ai](https://www.linkedin.com/company/netrica-ai/)**

## Abstract
[Add your paper abstract here]

This work presents a comprehensive framework for pedestrian-oriented route planning that leverages deep learning and spatial perception analysis. The framework predicts urban perception attributes (beautiful, lively, boring, safe) using delta sensitivity analysis across multiple machine learning models, enabling data-driven route recommendations based on pedestrian preferences.

## Key Contributions

1. **Delta Sensitivity Analysis**: Novel approach to test model robustness across 7 threshold values (δ = 0.5-1.8)

2. **Dual Classification Approach**:
   - Binary classification for above/below threshold predictions
   - Multi-class classification (Low/Medium/High) for perception categories

3. **Comprehensive Model Evaluation**:
   - 4 ML models (Random Forest, SVM, XGBoost, RealMLP)
   - 112 total experiments (4 perceptions × 7 deltas × 4 models)
   - City-based geographical splits for generalization

4. **Feature Extraction Pipeline**:
   - Semantic segmentation (ADE20K: 150 classes → 26 features)
   - Object detection (COCO: 80 classes → 10 features)
   - 36 visual features per urban scene

5. **Feature Importance Analysis**:
   - Permutation-based importance with statistical testing
   - Identification of key urban elements driving perception

6. **Publication-Ready Visualizations**:
   - 12+ figures analyzing model performance
   - Performance curves, heatmaps, and statistical summaries

## Dataset

**Source**: Place Pulse 2.0 dataset with urban street view images

**Size**: 111,268 street view images from 62 cities worldwide

**Perceptions**: 4 attributes rated on 0-10 scale
- Beautiful
- Lively
- Boring
- Safe

**Features**: 36 visual features
- 26 semantic segmentation features (scene composition)
- 10 object detection features (discrete objects)

**Split Strategy**:
- City-based splits (geographic generalization)
- Random splits (standard validation)
- 80/20 train/test ratio

## Methodology

### Stage 1: Feature Extraction
- **Semantic Segmentation**: ADE20K model (MIT)
- **Object Detection**: COCO Faster R-CNN
- **Processing**: 6-10 hours (GPU) for 111K images
- **Output**: 36 numerical features per image

### Stage 2: Perception Prediction
- **Binary Classification**: Dynamic threshold (median + δ × std)
- **Multi-class Classification**: Low/Medium/High categories
- **Models**: RF, SVM, XGBoost, RealMLP TD
- **Evaluation**: F1, Accuracy, ROC-AUC, PR-AUC

### Delta Sensitivity Analysis
Testing across 7 threshold values:
- δ = 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8
- Assesses model robustness to classification boundary changes
- Identifies optimal delta per perception/model combination

## Results

[Add your key results here]

### Best Performing Models
- **Beautiful**: [Model] with δ=[value], F1=[score]
- **Lively**: [Model] with δ=[value], F1=[score]
- **Boring**: [Model] with δ=[value], F1=[score]
- **Safe**: [Model] with δ=[value], F1=[score]

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]
4. [Finding 4]

### Feature Importance Insights
- Most important features for perception prediction
- Urban elements that drive different perceptions
- Correlation between features and perception ratings

## Applications

1. **Pedestrian Route Planning**
   - Recommend routes based on desired perceptions
   - Optimize for safety, beauty, or liveliness
   - Personalized navigation

2. **Urban Design**
   - Identify areas needing improvement
   - Data-driven design decisions
   - Perception-aware planning

3. **Smart Cities**
   - Real-time perception monitoring
   - Automated urban quality assessment
   - Citizen feedback integration

4. **Tourism & Navigation**
   - Scenic route recommendations
   - Experience-based navigation
   - Cultural exploration paths

## Implementation Details

**Language**: Python 3.8+

**Key Libraries**:
- scikit-learn (RF, SVM)
- XGBoost
- PyTorch (RealMLP, feature extraction)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)

**Hardware**:
- GPU: NVIDIA A100 (recommended for RealMLP)
- RAM: 8GB minimum, 16GB recommended
- Storage: 5GB for perception prediction, 100GB for feature extraction

**Runtime**:
- Perception prediction: 2-6 hours (full 112 experiments)
- Feature extraction: 6-10 hours (GPU) or 3-4 days (CPU)

## Repository Structure

See [REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md) for complete structure.

**Main Components**:
- `core_scripts/`: Perception prediction modules
- `Feature_Extraction_Code/`: Reference feature extraction (optional)
- `Feature_Importance/`: Feature importance analysis
- `Input_Data/`: Pre-extracted features (111,268 samples)
- `experiments/`: Results directory (created during runs)

## Documentation

- **README.md**: Main documentation
- **QUICK_START.md**: 5-minute setup guide
- **FEATURE_EXTRACTION.md**: Feature extraction details
- **PIPELINE.md**: End-to-end pipeline
- **VERIFICATION_CHECKLIST.md**: Testing guide

## Reproducibility

All experiments are fully reproducible:
1. Pre-extracted features included
2. Random seeds fixed
3. Complete configuration logs
4. Automated experiment runner
5. Documented hyperparameters

**To reproduce main results**:
```bash
source setup_experiment.sh
./run_experiment.sh --full --test-cities Mumbai
```

## License

MIT License - See [LICENSE](LICENSE) file

## Acknowledgments

- Place Pulse 2.0 dataset creators
- ADE20K and MS COCO dataset teams
- PyTorch and scikit-learn communities
- IISER Bhopal for computational resources
- [Add other acknowledgments]

## Related Publications

[Add related publications from your group]

1. [Publication 1]
2. [Publication 2]

## Contact

**Corresponding Author**: Pyare Lal Chauhan

**Lab**: GeoAI4Cities Lab
**Institution**: Indian Institute of Science Education and Research (IISER) Bhopal
**Email**: [pyarelal@iiserb.ac.in or appropriate email]

**Co-Authors**:
- Tanishq Kumar Baswal: [email@iiserb.ac.in]
- Vaibhav Kumar: [email@iiserb.ac.in]

**Project Page**: [Add project website URL]
**Lab Website**: [GeoAI4Cities](https://geoai4cities.com/)
**GitHub**: [Add repository URL]
**Netrica.ai**: [LinkedIn](https://www.linkedin.com/company/netrica-ai/)

---

**Published**: [Publication Date]
**Last Updated**: October 28, 2025
**Version**: 1.0.0
