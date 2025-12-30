#!/usr/bin/env python
"""
Enhanced Delta Sensitivity Experiment Runner with Progress Tracking
Dual Approach: Binary Classification + Multi-class Classification
Fixes: Progress bar, experiment folder location, timestamp generation, version warnings
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import time
from tqdm import tqdm
import warnings

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')
warnings.filterwarnings('ignore', message='A NumPy version')

# Import multiclass module
from multiclass_delta_sensitivity import MultiClassDeltaSensitivity
from multiclass_evaluator import MultiClassEvaluator

# Import visualization and report modules
try:
    from enhanced_publication_visualizer import EnhancedPublicationVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    try:
        from publication_visualizer import PublicationVisualizer as EnhancedPublicationVisualizer
        VISUALIZER_AVAILABLE = True
    except ImportError:
        VISUALIZER_AVAILABLE = False
        print("Warning: Publication visualizer not available")

try:
    from report_generator import ReportGenerator
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False
    print("Warning: Report generator not available")

# Import existing modules
sys.path.append(str(Path(__file__).parent))
try:
    from enhanced_main_experiment import EnhancedPerceptionDataLoader
    from model_trainer import ModelTrainer
except ImportError:
    print("Warning: Could not import enhanced modules, using fallback")


class EnhancedExperimentRunner:
    """Enhanced experiment runner with progress tracking and proper folder management."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the experiment runner.
        
        Args:
            base_dir: Base directory for experiments (default: current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Dynamic timestamp
        self.experiment_name = f"perception_delta_sensitivity_{self.timestamp}"
        
        # Feature columns count (from env, default 36)
        try:
            self.feature_count = int(os.environ.get("FEATURE_COUNT", "36"))
        except Exception:
            self.feature_count = 36

        # Ensure experiments folder is inside base directory; allow optional nested subdir via env
        experiments_root = self.base_dir / "experiments"
        experiments_subdir = os.environ.get("EXPERIMENTS_SUBDIR", "").strip()
        if experiments_subdir:
            experiments_root = experiments_root / experiments_subdir
        self.experiment_dir = experiments_root / self.experiment_name
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Configuration
        self.config = self._load_config()
        
        # Progress tracking
        self.total_experiments = 0
        self.completed_experiments = 0
        self.progress_file = self.experiment_dir / "05_logs" / "checkpoints" / "progress.json"
        
    def _setup_directories(self):
        """Create experiment directory structure."""
        dirs = [
            "00_configs",
            "01_data/splits",
            "01_data/class_distributions",
            "02_models",
            "03_results/metrics",
            "03_results/visualizations",
            "04_analysis/reports",
            "05_logs/checkpoints",
            "06_scripts",
            "07_backup"
        ]
        
        for dir_path in dirs:
            (self.experiment_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created experiment directory: {self.experiment_dir}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / "05_logs" / f"experiment_{self.timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info(f"ENHANCED DELTA SENSITIVITY EXPERIMENT - {self.timestamp}")
        self.logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.logger.info("=" * 80)
        
    def _load_config(self) -> dict:
        """Load or create configuration."""
        config_file = self.experiment_dir / "00_configs" / "experiment_config.yaml"
        
        # Check if we're in test mode by looking for test flag in sys.argv
        is_test_mode = '--test' in sys.argv
        
        # Default configuration
        default_config = {
            'experiment': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'random_state': 42,
                'n_jobs': -1
            },
            'data': {
                'input_dir': os.environ.get('INPUT_DATA_DIR', str(self.base_dir / "Input_Data")),
                'perceptions': ['beautiful'] if is_test_mode else ['beautiful', 'lively', 'boring', 'safe'],
                'test_size': 280,
                'validation_split': 0.2
            },
            'delta_values': [1.8] if is_test_mode else [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            'models': {
                'random_forest': {
                    'enabled': True,
                    'n_estimators': [500, 1000],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4]
                },
                'svm': {
                    'enabled': False if is_test_mode else True,
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 0.001, 0.01],
                    'kernel': ['rbf']
                },
                'xgboost': {
                    'enabled': False if is_test_mode else True,
                    'n_estimators': [400, 800],
                    'max_depth': [3, 6],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                },
                'realmlp_td': {
                    'enabled': True,
                    'random_state': [42],
                    'n_cv': [1]
                },
                'realmlp_hpo': {
                    'enabled': False,
                    'random_state': [42],
                    'n_hyperopt_steps': [50]
                },
                'mlp': {
                    'enabled': False
                }
            },
            'multiclass': {
                'enabled': False if is_test_mode else True,
                'n_classes': [3, 5]
            },
            'visualization': {
                'dpi': 300,
                'style': 'seaborn-v0_8-whitegrid',
                'figsize': [10, 6]
            }
        }
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        self.logger.info(f"Configuration saved to: {config_file}")
        return default_config
        
    def _update_progress(self, perception: str, delta: float, model: str, status: str):
        """Update progress tracking."""
        progress_data = {
            'total_experiments': self.total_experiments,
            'completed_experiments': self.completed_experiments,
            'progress_percentage': (self.completed_experiments / self.total_experiments * 100) if self.total_experiments > 0 else 0,
            'current': {
                'perception': perception,
                'delta': delta,
                'model': model,
                'status': status
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def run_experiment(self):
        """Run the complete delta sensitivity experiment with progress tracking."""
        
        # Calculate total experiments
        perceptions = self.config['data']['perceptions']
        delta_values = self.config['delta_values']
        models = [m for m, cfg in self.config['models'].items() if cfg.get('enabled', True)]
        
        self.total_experiments = len(perceptions) * len(delta_values) * len(models)
        
        self.logger.info(f"Total experiments to run: {self.total_experiments}")
        self.logger.info(f"Perceptions: {perceptions}")
        self.logger.info(f"Delta values: {delta_values}")
        self.logger.info(f"Models: {models}")
        
        # Results storage
        all_results = []
        
        # Main progress bar
        with tqdm(total=self.total_experiments, desc="Overall Progress", position=0) as pbar_main:
            
            # Iterate through perceptions
            for perception in perceptions:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing Perception: {perception.upper()}")
                self.logger.info(f"{'='*60}")
                
                # Load data for this perception
                data_file = Path(self.config['data']['input_dir']) / f"{perception}_input.xlsx"
                if not data_file.exists():
                    self.logger.error(f"Data file not found: {data_file}")
                    continue
                    
                # Load and prepare data
                self.logger.info(f"Loading data from: {data_file}")
                df = pd.read_excel(data_file)
                
                # Split data - check if using city-based split
                if 'test_cities' in self.config['data'] and self.config['data']['test_cities']:
                    # City-based split
                    test_cities = self.config['data']['test_cities']
                    if 'city_name' not in df.columns:
                        self.logger.error("city_name column not found in data for city-based split")
                        continue
                    
                    # Create test set mask
                    test_mask = df['city_name'].isin(test_cities)
                    local_test = df[test_mask].copy()
                    pp_pool = df[~test_mask].copy()
                    
                    self.logger.info(f"Test cities: {test_cities}")
                    self.logger.info(f"Test set city distribution: {local_test['city_name'].value_counts().to_dict()}")
                else:
                    # Original last N entries split
                    test_size = self.config['data']['test_size']
                    local_test = df.iloc[-test_size:]
                    pp_pool = df.iloc[:-test_size]
                
                self.logger.info(f"Place Pulse pool: {len(pp_pool)} samples")
                self.logger.info(f"Local test set: {len(local_test)} samples")
                
                # Progress bar for delta values
                with tqdm(delta_values, desc=f"{perception.capitalize()} - Delta", position=1, leave=False) as pbar_delta:
                    
                    for delta in pbar_delta:
                        pbar_delta.set_description(f"{perception.capitalize()} - Delta={delta}")
                        self.logger.info(f"Processing delta={delta} for {perception}")
                        
                        # Create analyzer for this delta
                        analyzer = MultiClassDeltaSensitivity(random_state=42)
                        
                        # Apply delta thresholds
                        delta_results = analyzer.create_delta_based_labels(
                            pp_pool['rating_score'].values, 
                            delta
                        )
                        
                        # Get binary labels
                        binary_labels = delta_results['binary']
                        
                        # Filter out mid-range samples
                        mask = binary_labels != 1  # Remove mid-range
                        if mask.sum() < 100:  # Skip if too few samples
                            self.logger.warning(f"Skipping delta={delta} for {perception} - too few samples ({mask.sum()})")
                            continue
                            
                        # Log binary distribution for analysis
                        binary_labels_filtered = binary_labels[mask]
                        n_neg = np.sum(binary_labels_filtered == 0)
                        n_pos = np.sum(binary_labels_filtered == 2)
                        self.logger.info(f"Delta={delta} binary distribution: neg={n_neg}, pos={n_pos}, mid={len(binary_labels) - mask.sum()}")
                            
                        # Prepare training data with train/val split
                        X_full = pp_pool.iloc[:, :self.feature_count].values[mask]
                        y_full = binary_labels[mask]
                        y_full[y_full == 2] = 1  # Convert to 0/1 labels
                        
                        # Create train/validation split (80/20)
                        from sklearn.model_selection import train_test_split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                        )
                        
                        # Progress bar for models
                        with tqdm(models, desc=f"Delta={delta} - Models", position=2, leave=False) as pbar_model:
                            
                            for model_name in pbar_model:
                                pbar_model.set_description(f"Delta={delta} - {model_name}")
                                
                                # Update progress
                                self._update_progress(perception, delta, model_name, "running")
                                self.logger.info(f"Training {model_name} for {perception} with delta={delta}")
                                
                                try:
                                    # Train model
                                    trainer = ModelTrainer(
                                        model_type=model_name,
                                        random_state=42
                                    )
                                    
                                    # Quick training for demonstration
                                    start_time = time.time()
                                    model = trainer.train_quick(X_train, y_train)
                                    train_time = time.time() - start_time
                                    
                                    self.logger.info(f"Model {model_name} trained in {train_time:.2f}s")
                                    
                                    # NEW: Evaluate on train/val/test sets for comprehensive analysis
                                    # 1. Training set performance
                                    train_metrics = self._evaluate_binary_classification(
                                        model, X_train, y_train, delta=None,  # No delta needed for binary labels
                                        pp_mean=None, pp_std=None, is_binary_labels=True
                                    )
                                    
                                    # 2. Validation set performance
                                    val_metrics = self._evaluate_binary_classification(
                                        model, X_val, y_val, delta=None,
                                        pp_mean=None, pp_std=None, is_binary_labels=True
                                    )
                                    
                                    # 3. Test set performance (evaluation on local test set)
                                    X_test = local_test.iloc[:, :self.feature_count].values
                                    y_test = local_test['rating_score'].values
                                    
                                    # Binary Classification Evaluation
                                    binary_metrics = self._evaluate_binary_classification(
                                        model, X_test, y_test, 
                                        delta=delta, 
                                        pp_mean=pp_pool['rating_score'].mean(), 
                                        pp_std=pp_pool['rating_score'].std()
                                    )
                                    
                                    # Multi-class Classification Evaluation (only if enabled)
                                    multiclass_metrics = {}
                                    if self.config['multiclass']['enabled']:
                                        evaluator = MultiClassEvaluator(n_classes=5)  # 5-class system
                                        
                                        # Convert continuous scores to multi-class labels
                                        multiclass_labels = analyzer.create_multiclass_labels(y_test)
                                        
                                        # Get predictions and evaluate
                                        y_pred = model.predict(X_test)
                                        y_proba = model.predict_proba(X_test)
                                        
                                        # Since model is binary, we need to map binary predictions to multiclass
                                        # This is a simplified mapping - in practice you'd train separate multiclass models
                                        multiclass_metrics = {
                                            'accuracy_multiclass': np.mean(y_pred == (multiclass_labels > 2).astype(int)),
                                            'n_classes': 5,
                                            'class_distribution': np.bincount(multiclass_labels, minlength=5).tolist()
                                        }
                                    
                                    # Combine all metrics with train/val/test prefixes
                                    metrics = {
                                        # Training set metrics
                                        'f1_train': train_metrics['f1_binary'],
                                        'accuracy_train': train_metrics['accuracy_binary'],
                                        'roc_auc_train': train_metrics.get('roc_auc_binary', 0.5),
                                        
                                        # Validation set metrics
                                        'f1_val': val_metrics['f1_binary'],
                                        'accuracy_val': val_metrics['accuracy_binary'],
                                        'roc_auc_val': val_metrics.get('roc_auc_binary', 0.5),
                                        
                                        # Test set metrics (original)
                                        **binary_metrics,
                                        **multiclass_metrics
                                    }
                                    
                                    # Store results
                                    result = {
                                        'perception': perception,
                                        'delta': delta,
                                        'model': model_name,
                                        'train_time': train_time,
                                        **metrics
                                    }
                                    all_results.append(result)
                                    
                                    # Log key metrics with train/val/test comparison
                                    self.logger.info(
                                        f"{perception} | δ={delta} | {model_name} | "
                                        f"Train: F1={metrics['f1_train']:.3f} | "
                                        f"Val: F1={metrics['f1_val']:.3f} | "
                                        f"Test: F1={metrics.get('f1_binary', 0):.3f}, ROC-AUC={metrics.get('roc_auc_binary', 0):.3f} | "
                                        f"Time={train_time:.2f}s"
                                    )
                                    
                                except Exception as e:
                                    self.logger.error(f"Error training {model_name} for {perception} δ={delta}: {e}")
                                    self._update_progress(perception, delta, model_name, "failed")
                                    continue
                                
                                # Update progress
                                self.completed_experiments += 1
                                self._update_progress(perception, delta, model_name, "completed")
                                pbar_main.update(1)
                                
                                # Update main progress bar description
                                pbar_main.set_description(
                                    f"Progress: {self.completed_experiments}/{self.total_experiments} "
                                    f"({self.completed_experiments/self.total_experiments*100:.1f}%)"
                                )
                
                # Log perception completion
                self.logger.info(f"Completed perception: {perception}")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_file = self.experiment_dir / "03_results" / "metrics" / "all_results.csv"
        results_df.to_csv(results_file, index=False)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT COMPLETED!")
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Total experiments completed: {self.completed_experiments}/{self.total_experiments}")
        self.logger.info("="*80)
        
        # Generate visualizations
        self._generate_visualizations(results_df)
        
        # Generate comprehensive report
        self._generate_report(results_df)
        
        # Generate summary
        self._generate_summary(results_df)
        
    def _generate_visualizations(self, results_df: pd.DataFrame):
        """Generate publication-ready visualizations."""
        if not VISUALIZER_AVAILABLE:
            self.logger.warning("Visualizer not available, skipping visualization generation")
            return
            
        if results_df.empty:
            self.logger.warning("No results data available for visualization")
            return
            
        try:
            self.logger.info("Generating publication-ready visualizations...")
            
            # Initialize enhanced visualizer
            viz_output_dir = self.experiment_dir / "03_results" / "visualizations"
            visualizer = EnhancedPublicationVisualizer(
                output_dir=str(viz_output_dir),
                style='publication',
                dpi=300
            )
            
            # Generate all figures including mean analysis
            figures = visualizer.create_enhanced_publication_figures(results_df)
            
            # Log created figures
            self.logger.info(f"Generated {len(figures)} publication figures:")
            for fig_name, fig_path in figures.items():
                self.logger.info(f"  - {fig_name}: {fig_path}")
                
            # Save figure list to metadata
            viz_metadata = {
                'total_figures': len(figures),
                'figure_list': figures,
                'output_directory': str(viz_output_dir),
                'style': 'publication',
                'dpi': 300,
                'formats': ['png', 'pdf', 'svg']
            }
            
            metadata_file = viz_output_dir / "visualization_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(viz_metadata, f, indent=2)
                
            self.logger.info(f"Visualization metadata saved to: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            self.logger.info("Experiment results are still available in CSV format")
        
    def _generate_report(self, results_df: pd.DataFrame):
        """Generate comprehensive analysis report."""
        if not REPORT_AVAILABLE:
            self.logger.warning("Report generator not available, skipping report generation")
            return
            
        if results_df.empty:
            self.logger.warning("No results data available for report generation")
            return
            
        try:
            self.logger.info("Generating comprehensive analysis report...")
            
            # Initialize report generator
            report_gen = ReportGenerator(str(self.experiment_dir))
            
            # Generate reports in multiple formats
            report_paths = report_gen.generate_comprehensive_report(
                results_df, 
                config=self.config
            )
            
            # Log generated reports
            self.logger.info(f"Generated {len(report_paths)} report formats:")
            for format_name, path in report_paths.items():
                self.logger.info(f"  - {format_name}: {path}")
                
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            self.logger.info("Results are still available in CSV and visualization formats")
    
    def _generate_summary(self, results_df: pd.DataFrame):
        """Generate experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'total_experiments': self.total_experiments,
            'completed_experiments': self.completed_experiments,
            'success_rate': f"{(self.completed_experiments/self.total_experiments*100):.1f}%" if self.total_experiments > 0 else "0%",
            'best_results': {}
        }
        
        # Check if we have results to summarize
        if results_df.empty or 'perception' not in results_df.columns:
            self.logger.warning("No results to summarize")
            summary_file = self.experiment_dir / "experiment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Summary saved to: {summary_file}")
            return
        
        # Find best configuration for each perception
        for perception in results_df['perception'].unique():
            perception_df = results_df[results_df['perception'] == perception]
            best_row = perception_df.loc[perception_df['f1_binary'].idxmax()]
            summary['best_results'][perception] = {
                'delta': float(best_row['delta']),
                'model': best_row['model'],
                'f1_score': float(best_row['f1_binary']),
                'roc_auc': float(best_row.get('roc_auc_binary', 0))
            }
        
        # Save summary
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for perception, results in summary['best_results'].items():
            print(f"\n{perception.upper()}:")
            print(f"  Best Delta: {results['delta']}")
            print(f"  Best Model: {results['model']}")
            print(f"  F1 Score: {results['f1_score']:.3f}")
            print(f"  ROC-AUC: {results['roc_auc']:.3f}")
    
    def _evaluate_binary_classification(self, model, X_test, y_test_input, 
                                       delta=None, pp_mean=None, pp_std=None,
                                       is_binary_labels=False):
        """
        Evaluate binary classification performance on test data.
        
        Args:
            model: Trained binary classifier
            X_test: Test features
            y_test_input: Test labels (either binary 0/1 or continuous 0-10)
            delta: Delta value used for thresholding (for continuous labels)
            pp_mean: Mean of Place Pulse training data (for continuous labels)
            pp_std: Standard deviation of Place Pulse training data (for continuous labels)
            is_binary_labels: If True, y_test_input is already binary (0/1)
            
        Returns:
            Dictionary of binary classification metrics
        """
        from sklearn.metrics import (
            f1_score, accuracy_score, precision_score, recall_score,
            roc_auc_score, average_precision_score
        )
        
        # Handle binary vs continuous labels
        if is_binary_labels:
            # Labels are already binary (for train/val evaluation)
            y_test_binary = y_test_input
            X_test_filtered = X_test
            mask_valid = np.ones(len(y_test_binary), dtype=bool)
        else:
            # Convert continuous scores to binary (for test evaluation)
            # Calculate thresholds on test set itself, not from Place Pulse data
            test_mean = np.mean(y_test_input)
            test_std = np.std(y_test_input)
            if test_std == 0 or np.isnan(test_std):
                # If test set has no variance, use a small epsilon
                test_std = 1e-6
            neg_threshold = test_mean - delta * test_std
            pos_threshold = test_mean + delta * test_std
            
            # Log test set binary distribution for analysis
            test_neg_samples = np.sum(y_test_input < neg_threshold)
            test_pos_samples = np.sum(y_test_input > pos_threshold)
            test_mid_samples = len(y_test_input) - test_neg_samples - test_pos_samples
            self.logger.info(f"Test set delta={delta} distribution: neg={test_neg_samples}, pos={test_pos_samples}, mid={test_mid_samples}")
            
            # Create binary labels for test data
            y_test_binary = np.zeros(len(y_test_input), dtype=int)
            mask_neg = y_test_input < neg_threshold
            mask_pos = y_test_input > pos_threshold
            mask_mid = (~mask_neg) & (~mask_pos)
            
            y_test_binary[mask_neg] = 0
            y_test_binary[mask_pos] = 1
            
            # Filter out mid-range samples for binary evaluation
            mask_valid = ~mask_mid
        
        if mask_valid.sum() == 0:
            return {
                'f1_binary': 0.0,
                'accuracy_binary': 0.0,
                'precision_binary': 0.0,
                'recall_binary': 0.0,
                'roc_auc_binary': 0.5,
                'n_samples_binary': 0,
                'n_pos_binary': 0,
                'n_neg_binary': 0,
                'n_mid_binary': mask_mid.sum() if not is_binary_labels else 0 if not is_binary_labels else 0
            }
        
        X_test_filtered = X_test[mask_valid] if not is_binary_labels else X_test
        y_test_filtered = y_test_binary[mask_valid] if not is_binary_labels else y_test_binary
        
        try:
            # Get predictions
            y_pred = model.predict(X_test_filtered)
            y_proba = model.predict_proba(X_test_filtered)
            
            # Calculate metrics
            metrics = {
                'f1_binary': f1_score(y_test_filtered, y_pred, average='binary'),
                'accuracy_binary': accuracy_score(y_test_filtered, y_pred),
                'precision_binary': precision_score(y_test_filtered, y_pred, average='binary', zero_division=0),
                'recall_binary': recall_score(y_test_filtered, y_pred, average='binary', zero_division=0),
                'n_samples_binary': len(y_test_filtered),
                'n_pos_binary': (y_test_filtered == 1).sum(),
                'n_neg_binary': (y_test_filtered == 0).sum(),
                'n_mid_binary': mask_mid.sum() if not is_binary_labels else 0
            }
            
            # Add ROC-AUC if we have both classes
            if len(np.unique(y_test_filtered)) > 1:
                metrics['roc_auc_binary'] = roc_auc_score(y_test_filtered, y_proba[:, 1])
                metrics['pr_auc_binary'] = average_precision_score(y_test_filtered, y_proba[:, 1])
            else:
                metrics['roc_auc_binary'] = 0.5
                metrics['pr_auc_binary'] = 0.5
                
        except Exception as e:
            self.logger.error(f"Error in binary evaluation: {e}")
            metrics = {
                'f1_binary': 0.0,
                'accuracy_binary': 0.0,
                'roc_auc_binary': 0.5,
                'n_samples_binary': 0,
                'n_pos_binary': 0,
                'n_neg_binary': 0,
                'n_mid_binary': mask_mid.sum() if not is_binary_labels else 0
            }
            
        return metrics


class ModelTrainer:
    """Simple model trainer for quick experiments."""
    
    def __init__(self, model_type: str, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        
    def train_quick(self, X_train, y_train):
        """Quick training without extensive hyperparameter tuning."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            model = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except ImportError:
                # Fallback to RandomForest if XGBoost not available
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        elif self.model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=100,
                random_state=self.random_state
            )
        elif self.model_type == 'realmlp_td':
            try:
                # Import locally to avoid hard dependency if not used
                from pytabkit import RealMLP_TD_Classifier
                model = RealMLP_TD_Classifier(random_state=self.random_state)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"RealMLP_TD not available ({e}); falling back to MLPClassifier"
                )
                model = MLPClassifier(
                    hidden_layer_sizes=(64,),
                    max_iter=100,
                    random_state=self.random_state
                )
        elif self.model_type == 'realmlp_hpo':
            try:
                from pytabkit import RealMLP_HPO_Classifier
                # Default steps; could be overridden by config in a fuller refactor
                model = RealMLP_HPO_Classifier(n_hyperopt_steps=50, random_state=self.random_state)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"RealMLP_HPO not available ({e}); falling back to RealMLP_TD"
                )
                try:
                    from pytabkit import RealMLP_TD_Classifier
                    model = RealMLP_TD_Classifier(random_state=self.random_state)
                except Exception:
                    model = MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=100,
                        random_state=self.random_state
                    )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        model.fit(X_train, y_train)
        return model


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Delta Sensitivity Experiment Runner')
    parser.add_argument('--base-dir', type=str, default=None, help='Base directory for experiments')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced experiments')
    parser.add_argument('--test-cities', nargs='+', default=None, help='List of city names to use as test set (enables city-based split)')
    parser.add_argument('--realmlp', type=str, default='td', choices=['td', 'hpo', 'both', 'off'],
                        help='Select RealMLP variant: td (default), hpo, both, or off')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only by ignoring any GPU')
    
    args = parser.parse_args()
    
    # Create runner
    runner = EnhancedExperimentRunner(base_dir=args.base_dir)
    
    # Update configuration if test cities are specified
    if args.test_cities:
        runner.config['data']['test_cities'] = args.test_cities
        runner.logger.info(f"Using city-based test set: {args.test_cities}")
    
    # Apply RealMLP selection flags
    if args.realmlp == 'off':
        if 'realmlp_td' in runner.config['models']:
            runner.config['models']['realmlp_td']['enabled'] = False
        if 'realmlp_hpo' in runner.config['models']:
            runner.config['models']['realmlp_hpo']['enabled'] = False
    elif args.realmlp == 'td':
        if 'realmlp_td' in runner.config['models']:
            runner.config['models']['realmlp_td']['enabled'] = True
        if 'realmlp_hpo' in runner.config['models']:
            runner.config['models']['realmlp_hpo']['enabled'] = False
    elif args.realmlp == 'hpo':
        if 'realmlp_td' in runner.config['models']:
            runner.config['models']['realmlp_td']['enabled'] = False
        if 'realmlp_hpo' in runner.config['models']:
            runner.config['models']['realmlp_hpo']['enabled'] = True
    elif args.realmlp == 'both':
        if 'realmlp_td' in runner.config['models']:
            runner.config['models']['realmlp_td']['enabled'] = True
        if 'realmlp_hpo' in runner.config['models']:
            runner.config['models']['realmlp_hpo']['enabled'] = True

    # Enforce CPU-only if requested
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.test:
        # Test mode: reduce experiments for quick testing
        runner.config['data']['perceptions'] = ['beautiful']
        runner.config['delta_values'] = [1.8]  # Only 1 delta value for quick test
        # Only enable two models for quick test
        runner.config['models'] = {
            'random_forest': {'enabled': True},
            'svm': {'enabled': False},
            'xgboost': {'enabled': False},
            'realmlp_td': {'enabled': True},
            'realmlp_hpo': {'enabled': False},
            'mlp': {'enabled': False}
        }
        # Disable multiclass for quick test
        runner.config['multiclass']['enabled'] = False
        
        # Re-save the updated configuration
        config_file = runner.experiment_dir / "00_configs" / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(runner.config, f, default_flow_style=False)
        
        runner.logger.info("Applied test mode configuration and re-saved config")
        
    # Run experiment
    runner.run_experiment()


if __name__ == "__main__":
    main()