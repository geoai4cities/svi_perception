# Feature Importance Pipeline - Configuration System

## Overview

The feature importance pipeline now uses a standardized YAML-based configuration system for maximum flexibility and generalizability. This allows easy customization of experiments without modifying code.

## Configuration File Structure

### Main Configuration: `experiment_config.yaml`

```yaml
# Experiment metadata
experiment:
  name: "perception_feature_importance"
  description: "Feature importance analysis for urban perception prediction"
  version: "1.0"
  author: "Perception Research Team"

# Data configuration
data:
  input_dir: "Input_Data"
  feature_count: 36  # 36 features + rating_score column
  test_size: 280
  random_state: 42
  
# Perceptions and their optimal configurations
perceptions:
  beautiful:
    model: "random_forest"
    delta: 1.8
    enabled: true
    
  lively:
    model: "random_forest" 
    delta: 1.2
    enabled: true
    
  boring:
    model: "random_forest"
    delta: 1.8
    enabled: true
    
  safe:
    model: "xgboost"
    delta: 1.4
    enabled: true

# Model configurations
models:
  random_forest:
    enabled: true
    parameters:
      n_estimators: 1000
      max_depth: 20
      min_samples_split: 2
      min_samples_leaf: 1
      random_state: 42
      n_jobs: -1
      
  xgboost:
    enabled: true
    parameters:
      n_estimators: 800
      max_depth: 6
      learning_rate: 0.1
      subsample: 1.0
      random_state: 42
      n_jobs: -1
      eval_metric: "logloss"

# Feature importance analysis settings
analysis:
  method: "permutation_importance"
  n_repeats: 30
  significance_level: 0.05
  performance_metric: "f1"
  top_features: 10

# Output configuration
output:
  results_dir: "Feature_importance/results"
  report_format: "markdown"
  visualization_formats: ["png", "pdf", "svg"]
  figure_dpi: 300
```

## Usage

### Basic Usage (Default Configuration)
```bash
python3 run_feature_importance_pipeline.py
```

### Custom Configuration
```bash
python3 run_feature_importance_pipeline.py --config custom_config.yaml
```

### Example Custom Configuration
```bash
python3 run_feature_importance_pipeline.py --config vit_finetuned_best_config.yaml
```

### Running with GPU for RealMLP TD
```bash
# Set GPU device (0, 1, 2, etc.)
export CUDA_VISIBLE_DEVICES=1
python3 run_feature_importance_pipeline.py --config your_config.yaml
```

### Running in Background
```bash
# Run in background and save output to log file
nohup python3 run_feature_importance_pipeline.py --config your_config.yaml > pipeline_output.log 2>&1 &
```

## Output Directory Structure

The pipeline creates the following structured output:

```
Feature_Importance/
├── outputs/
│   └── feature_importance_analysis/     # Main output directory
│       ├── results/                     # Analysis results
│       │   ├── figure_9_feature_importance.png
│       │   ├── figure_9_feature_importance.pdf
│       │   ├── figure_9_feature_importance.svg
│       │   ├── feature_importance_detailed.csv
│       │   ├── feature_importance_summary.csv
│       │   └── feature_importance_report_perception_feature_importance.md
│       ├── saved_models/                # Trained models
│       │   ├── beautiful_random_forest_delta_1.8_YYYYMMDD_HHMMSS.pkl
│       │   ├── lively_random_forest_delta_1.8_YYYYMMDD_HHMMSS.pkl
│       │   ├── boring_svm_delta_1.6_YYYYMMDD_HHMMSS.pkl
│       │   └── safe_random_forest_delta_1.8_YYYYMMDD_HHMMSS.pkl
│       └── logs/                        # Pipeline logs
│           └── feature_importance_pipeline.log
└── config files...
```

### Output Files Description

- **`saved_models/`**: Contains trained models for each perception with timestamps
- **`figure_9_feature_importance.*`**: Publication-ready visualizations in multiple formats
- **`feature_importance_detailed.csv`**: Complete analysis results with all features
- **`feature_importance_summary.csv`**: Summary of top features per perception
- **`feature_importance_report_*.md`**: Comprehensive markdown report
- **`feature_importance_pipeline.log`**: Detailed execution log with timestamps

## Configuration Sections

### 1. Experiment Metadata
- **name**: Experiment identifier
- **description**: Human-readable description
- **version**: Configuration version
- **author**: Author information

### 2. Data Configuration
- **input_dir**: Directory containing input data files
- **feature_count**: Number of features (36 for current setup)
- **test_size**: Number of samples for test set (IGNORED when test_city_name is specified)
- **test_city_name**: City name for city-based split (uses ALL samples from this city as test set)
- **random_state**: Random seed for reproducibility

#### Data Splitting Strategy
The pipeline supports two data splitting methods:

1. **City-based Split (Recommended)**:
   - When `test_city_name` is specified (e.g., "Paris")
   - Uses ALL samples from the specified city as test set
   - Uses ALL samples from other cities as training pool
   - The `test_size` parameter is ignored in this case

2. **Fixed Size Split (Fallback)**:
   - When no `test_city_name` is provided
   - Uses the last N samples (specified by `test_size`) as test set
   - Uses remaining samples as training pool

### 3. Perceptions Configuration
- **model**: Model type for each perception (`random_forest` or `xgboost`)
- **delta**: Delta value for binary classification
- **enabled**: Whether to include this perception in analysis

### 4. Model Configurations
- **enabled**: Whether model type is available
- **parameters**: Model-specific hyperparameters

#### Supported Models
- **random_forest**: Random Forest Classifier
- **xgboost**: XGBoost Classifier
- **svm**: Support Vector Machine
- **realmlp_td**: RealMLP Tabular Data (supports GPU)
- **realmlp_hpo**: RealMLP with Hyperparameter Optimization

#### GPU Configuration for RealMLP TD
RealMLP TD models can utilize GPU acceleration. To enable GPU usage:

1. **Set GPU device**:
   ```bash
   export CUDA_VISIBLE_DEVICES=1  # Use GPU 1 (or 0, 2, etc.)
   ```

2. **Configure in YAML**:
   ```yaml
   models:
     realmlp_td:
       enabled: true
       parameters:
         random_state: 42
         n_cv: 0
         device: "cuda"  # Optional: explicitly set device
   ```

3. **Memory considerations**:
   - RealMLP TD automatically detects available GPU
   - Falls back to CPU if GPU is not available
   - Monitor GPU memory usage for large datasets

### 5. Analysis Configuration
- **method**: Feature importance method (currently only `permutation_importance`)
- **n_repeats**: Number of repeats for permutation importance
- **significance_level**: Statistical significance threshold
- **performance_metric**: Metric for importance calculation
- **top_features**: Number of top features to report

### 6. Output Configuration
- **base_dir**: Main output directory (e.g., "outputs/feature_importance_analysis")
- **results_dir**: Results subdirectory (e.g., "results")
- **models_dir**: Models subdirectory (e.g., "saved_models")
- **logs_dir**: Logs subdirectory (e.g., "logs")
- **report_format**: Report format (currently only `markdown`)
- **visualization_formats**: Image formats for plots
- **figure_dpi**: Resolution for saved figures

## Key Benefits

1. **Generalizability**: Easy to adapt for different experiments
2. **Reproducibility**: All settings in one file
3. **Flexibility**: Enable/disable perceptions and models
4. **Customization**: Adjust all parameters without code changes
5. **Documentation**: Self-documenting configuration files

## Migration from Previous System

The pipeline automatically uses the new configuration system. The old `best_config.txt` file is no longer needed as all settings are now in `experiment_config.yaml`.

## Example Customizations

### Disable a Perception
```yaml
perceptions:
  safe:
    model: "xgboost"
    delta: 1.4
    enabled: false  # This perception will be skipped
```

### Use Different Model Parameters
```yaml
models:
  random_forest:
    parameters:
      n_estimators: 500  # Fewer trees for faster training
      max_depth: 15
      min_samples_split: 5
```

### Change Analysis Settings
```yaml
analysis:
  n_repeats: 20  # Fewer repeats for faster analysis
  significance_level: 0.01  # More strict significance
  top_features: 15  # More top features
```

### City-based Split Configuration
```yaml
data:
  input_dir: "../Input_Data/dinov3_finetuned"
  test_city_name: "Paris"  # Use ALL Paris samples as test set
  test_size: 0.2  # IGNORED when test_city_name is specified
```

### GPU Configuration for RealMLP
```yaml
perceptions:
  boring:
    model: "realmlp_td"  # Use RealMLP TD with GPU support
    delta: 1.8
    enabled: true

models:
  realmlp_td:
    enabled: true
    parameters:
      random_state: 42
      n_cv: 0
      device: "cuda"  # Explicitly use GPU
```

### Structured Output Configuration
```yaml
output:
  base_dir: "outputs/feature_importance_analysis"  # Main output directory
  results_dir: "results"                           # Results subdirectory
  models_dir: "saved_models"                       # Models subdirectory
  logs_dir: "logs"                                 # Logs subdirectory
  report_format: "markdown"
  visualization_formats: ["png", "pdf", "svg"]
  figure_dpi: 300
```

## Troubleshooting

### Common Issues

1. **"No such file or directory" error**:
   - Check that input data files exist in the specified `input_dir`
   - Ensure file names match the pattern: `{perception}_input.xlsx`

2. **"Invalid configuration loaded" error**:
   - Verify all required sections are present in YAML
   - Check that model types are supported (random_forest, xgboost, svm, realmlp_td, realmlp_hpo)
   - Ensure perception names are valid (beautiful, lively, boring, safe)

3. **GPU not detected for RealMLP**:
   - Check CUDA installation: `nvidia-smi`
   - Set GPU device: `export CUDA_VISIBLE_DEVICES=1`
   - RealMLP will fall back to CPU if GPU is unavailable

4. **Memory issues with large datasets**:
   - Reduce `n_repeats` in analysis configuration
   - Use smaller model parameters (fewer estimators, smaller max_depth)
   - Consider using CPU instead of GPU for very large datasets

5. **City-based split not working**:
   - Ensure data files contain a `city_name` column
   - Check that the specified city name exists in the data
   - Verify city name spelling and case sensitivity

### Performance Tips

- **Faster training**: Reduce `n_estimators` for Random Forest, use fewer `n_repeats` for analysis
- **Memory optimization**: Use `n_jobs: -1` for parallel processing, monitor GPU memory
- **Reproducibility**: Always set `random_state` parameters consistently

## File Structure

```
Feature_Importance/
├── experiment_config.yaml      # Main configuration
├── example_config.yaml         # Example custom configuration
├── config_loader.py            # Configuration loader
├── run_feature_importance_pipeline.py  # Main pipeline
├── retrain_and_save_models.py  # Model training
├── feature_importance_analysis.py  # Analysis
└── model_saver.py              # Model saving utilities
```
