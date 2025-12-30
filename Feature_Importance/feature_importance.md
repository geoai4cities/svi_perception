## Feature Importance (Unified Pipeline)

This document explains the unified, configuration-driven pipeline for feature importance across models (Random Forest, SVM, XGBoost, RealMLP TD), aligned with the main experiment settings (same deltas, splits, and feature count).

### How to Run

1) Make outputs mirror main experiments (dataset/city):
```bash
export OUTPUT_SUBDIR="<dataset>/<city>"   # e.g., cnn/Paris or dinov3_finetuned/Mumbai
python3 Feature_Importance/run_feature_importance_pipeline.py --config Feature_Importance/experiment_config.yaml
```

2) Models are retrained per perception using the config, then permutation importance is computed and visualized.

### Configuration
- File: `Feature_Importance/experiment_config.yaml`
- Key sections:
  - `data`: input_dir (e.g., ../Input_Data), feature_count (default 36), test_size (280)
  - `perceptions`: per-perception `model` and `delta` (e.g., lively → svm, δ=1.2)
  - `models`: exact parameters matching the main experiments
  - `analysis`: permutation importance settings (n_repeats, α)

### Outputs
- If `OUTPUT_SUBDIR` is set: `output/<dataset>/<city>/`
  - `saved_models/`: trained models per perception
  - `figure_9_feature_importance.(png|pdf|svg)`
  - `feature_importance_detailed.csv`
  - `feature_importance_summary.csv`
  - `feature_importance_report_*.md`
- Otherwise (fallback): `Feature_importance/results/` and `Feature_importance/saved_models/`

### Method Summary
1) Retrain best-configured model per perception using delta-based labels and the same binary filtering as the main experiment.
2) Compute permutation importance with repeats (e.g., n_repeats=30) using F1 as the scoring metric.
3) Perform one-sample t-tests against 0 to obtain p-values; report significant features.
4) Produce publication-ready plots (PNG, PDF, SVG) and CSV summaries.

### Notes
- GPU usage for RealMLP TD is automatic when available (PyTorch Lightning). To pin a GPU:
```bash
export CUDA_VISIBLE_DEVICES="0"
```
- Feature count can be changed globally by setting `FEATURE_COUNT` in the main runner; the pipeline uses `data.feature_count`.

### References
- Code: `Feature_Importance/retrain_and_save_models.py`, `Feature_Importance/feature_importance_analysis.py`
- Config: `Feature_Importance/experiment_config.yaml`, `Feature_Importance/example_config.yaml`