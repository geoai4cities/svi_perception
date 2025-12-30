#!/usr/bin/env python
"""
Enhanced Dual Approach Experiment Runner
Trains BOTH Binary and Multi-class models for comprehensive evaluation
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import multiclass modules
from multiclass_delta_sensitivity import MultiClassDeltaSensitivity
from multiclass_evaluator import MultiClassEvaluator


class DualApproachExperiment:
    """
    Dual Approach: Trains both binary and multi-class models
    Provides comprehensive evaluation across both paradigms
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the dual approach experiment."""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"dual_approach_{self.timestamp}"
        
        # Create experiment directory
        self.experiment_dir = self.base_dir / "experiments" / self.experiment_name
        self._setup_directories()
        self._setup_logging()
        
        # Configuration
        self.config = self._load_config()
        
    def _setup_directories(self):
        """Create experiment directory structure."""
        dirs = [
            "00_configs",
            "01_data",
            "02_models/binary",
            "02_models/multiclass",
            "03_results/binary",
            "03_results/multiclass",
            "03_results/comparison",
            "04_analysis",
            "05_logs"
        ]
        
        for dir_path in dirs:
            (self.experiment_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / "05_logs" / f"dual_experiment_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info(f"DUAL APPROACH EXPERIMENT - {self.timestamp}")
        self.logger.info("Binary Classification + Multi-class Classification")
        self.logger.info(f"Experiment Directory: {self.experiment_dir}")
        self.logger.info("="*80)
        
    def _load_config(self) -> dict:
        """Load or create configuration."""
        config = {
            'experiment': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'random_state': 42
            },
            'data': {
                'input_dir': str(self.base_dir / "Input_data"),
                'perceptions': ['beautiful', 'lively', 'boring', 'safe'],
                'test_size': 280,
                'validation_split': 0.2
            },
            'delta_values': [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            'multiclass': {
                'n_classes': [3, 5]  # Test with 3 and 5 class systems
            },
            'models': ['random_forest', 'svm', 'mlp']
        }
        
        # Save configuration
        config_file = self.experiment_dir / "00_configs" / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return config
        
    def run_experiment(self):
        """Run the complete dual approach experiment."""
        
        perceptions = self.config['data']['perceptions']
        delta_values = self.config['delta_values']
        n_classes_list = self.config['multiclass']['n_classes']
        
        all_results = []
        
        # Main experiment loop
        for perception in tqdm(perceptions, desc="Perceptions"):
            self.logger.info(f"\nProcessing {perception.upper()}")
            
            # Load data
            data_file = Path(self.config['data']['input_dir']) / f"{perception}_input.xlsx"
            df = pd.read_excel(data_file)
            
            # Split data
            test_size = self.config['data']['test_size']
            local_test = df.iloc[-test_size:]
            pp_pool = df.iloc[:-test_size]
            
            X_test = local_test.iloc[:, :36].values
            y_test = local_test['rating_score'].values
            
            for delta in tqdm(delta_values, desc=f"  Delta values", leave=False):
                
                # === BINARY CLASSIFICATION ===
                binary_results = self._train_binary_models(
                    pp_pool, X_test, y_test, delta, perception
                )
                
                # === MULTI-CLASS CLASSIFICATION ===
                for n_classes in n_classes_list:
                    multiclass_results = self._train_multiclass_models(
                        pp_pool, X_test, y_test, n_classes, delta, perception
                    )
                    
                    # Combine results
                    for binary_res in binary_results:
                        for multi_res in multiclass_results:
                            if binary_res['model'] == multi_res['model']:
                                combined = {
                                    'perception': perception,
                                    'delta': delta,
                                    'model': binary_res['model'],
                                    'n_classes': n_classes,
                                    # Binary metrics
                                    'binary_f1': binary_res['f1'],
                                    'binary_accuracy': binary_res['accuracy'],
                                    'binary_roc_auc': binary_res['roc_auc'],
                                    'binary_n_samples': binary_res['n_samples'],
                                    # Multi-class metrics
                                    'multi_f1_macro': multi_res['f1_macro'],
                                    'multi_f1_weighted': multi_res['f1_weighted'],
                                    'multi_accuracy': multi_res['accuracy'],
                                    'multi_balanced_accuracy': multi_res['balanced_accuracy'],
                                    # Comparison
                                    'accuracy_improvement': multi_res['accuracy'] - binary_res['accuracy'],
                                    'f1_comparison': multi_res['f1_weighted'] - binary_res['f1']
                                }
                                all_results.append(combined)
                                
                                self.logger.info(
                                    f"  {perception} | δ={delta} | {binary_res['model']} | "
                                    f"Binary F1={binary_res['f1']:.3f} | "
                                    f"Multi-class({n_classes}) F1={multi_res['f1_weighted']:.3f}"
                                )
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_file = self.experiment_dir / "03_results" / "comparison" / "dual_approach_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Generate analysis
        self._analyze_results(results_df)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("DUAL APPROACH EXPERIMENT COMPLETED!")
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info("="*80)
        
    def _train_binary_models(self, pp_pool, X_test, y_test, delta, perception):
        """Train binary classification models."""
        results = []
        
        # Create binary labels using delta thresholds
        analyzer = MultiClassDeltaSensitivity(random_state=42)
        delta_results = analyzer.create_delta_based_labels(
            pp_pool['rating_score'].values, delta
        )
        
        binary_labels = delta_results['binary']
        mask = binary_labels != 1  # Remove mid-range
        
        if mask.sum() < 100:
            return results
            
        X_train = pp_pool.iloc[:, :36].values[mask]
        y_train = binary_labels[mask]
        y_train[y_train == 2] = 1  # Convert to 0/1
        
        # Train models
        for model_name in self.config['models']:
            try:
                model = self._get_model(model_name, 'binary')
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                metrics = self._evaluate_binary(
                    model, X_test, y_test, delta,
                    delta_results['thresholds']['mean'],
                    delta_results['thresholds']['std']
                )
                
                metrics['model'] = model_name
                results.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Error training binary {model_name}: {e}")
                
        return results
        
    def _train_multiclass_models(self, pp_pool, X_test, y_test, n_classes, delta, perception):
        """Train multi-class classification models."""
        results = []
        
        # Create multi-class labels
        analyzer = MultiClassDeltaSensitivity(random_state=42, n_classes=n_classes)
        
        # For training: convert scores to multi-class labels
        y_train_multi = analyzer.create_multiclass_labels(pp_pool['rating_score'].values)
        X_train = pp_pool.iloc[:, :36].values
        
        # Train models
        for model_name in self.config['models']:
            try:
                model = self._get_model(model_name, 'multiclass', n_classes)
                model.fit(X_train, y_train_multi)
                
                # Evaluate on test set
                y_test_multi = analyzer.create_multiclass_labels(y_test)
                metrics = self._evaluate_multiclass(model, X_test, y_test_multi, n_classes)
                
                metrics['model'] = model_name
                results.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Error training multiclass {model_name}: {e}")
                
        return results
        
    def _get_model(self, model_name: str, model_type: str, n_classes: int = 2):
        """Get model instance based on name and type."""
        if model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif model_name == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def _evaluate_binary(self, model, X_test, y_test_continuous, delta, pp_mean, pp_std):
        """Evaluate binary classification model."""
        # Apply thresholds to test data
        neg_threshold = pp_mean - delta * pp_std
        pos_threshold = pp_mean + delta * pp_std
        
        y_test_binary = np.zeros(len(y_test_continuous), dtype=int)
        mask_neg = y_test_continuous < neg_threshold
        mask_pos = y_test_continuous > pos_threshold
        mask_mid = (~mask_neg) & (~mask_pos)
        
        y_test_binary[mask_neg] = 0
        y_test_binary[mask_pos] = 1
        
        mask_valid = ~mask_mid
        if mask_valid.sum() == 0:
            return {'f1': 0, 'accuracy': 0, 'roc_auc': 0.5, 'n_samples': 0}
            
        X_test_filtered = X_test[mask_valid]
        y_test_filtered = y_test_binary[mask_valid]
        
        y_pred = model.predict(X_test_filtered)
        y_proba = model.predict_proba(X_test_filtered)
        
        return {
            'f1': f1_score(y_test_filtered, y_pred),
            'accuracy': accuracy_score(y_test_filtered, y_pred),
            'roc_auc': roc_auc_score(y_test_filtered, y_proba[:, 1]) if len(np.unique(y_test_filtered)) > 1 else 0.5,
            'n_samples': len(y_test_filtered)
        }
        
    def _evaluate_multiclass(self, model, X_test, y_test_multi, n_classes):
        """Evaluate multi-class classification model."""
        y_pred = model.predict(X_test)
        
        return {
            'f1_macro': f1_score(y_test_multi, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test_multi, y_pred, average='weighted'),
            'accuracy': accuracy_score(y_test_multi, y_pred),
            'balanced_accuracy': accuracy_score(y_test_multi, y_pred)
        }
        
    def _analyze_results(self, results_df):
        """Analyze and summarize dual approach results."""
        summary = {}
        
        # Best configurations
        for perception in results_df['perception'].unique():
            perc_df = results_df[results_df['perception'] == perception]
            
            # Best binary configuration
            best_binary = perc_df.loc[perc_df['binary_f1'].idxmax()]
            
            # Best multi-class configuration
            best_multi = perc_df.loc[perc_df['multi_f1_weighted'].idxmax()]
            
            summary[perception] = {
                'best_binary': {
                    'delta': float(best_binary['delta']),
                    'model': best_binary['model'],
                    'f1': float(best_binary['binary_f1']),
                    'roc_auc': float(best_binary['binary_roc_auc'])
                },
                'best_multiclass': {
                    'delta': float(best_multi['delta']),
                    'model': best_multi['model'],
                    'n_classes': int(best_multi['n_classes']),
                    'f1_weighted': float(best_multi['multi_f1_weighted']),
                    'accuracy': float(best_multi['multi_accuracy'])
                },
                'comparison': {
                    'avg_accuracy_improvement': float(perc_df['accuracy_improvement'].mean()),
                    'best_improvement': float(perc_df['accuracy_improvement'].max())
                }
            }
            
        # Save summary
        summary_file = self.experiment_dir / "04_analysis" / "dual_approach_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print summary
        print("\n" + "="*80)
        print("DUAL APPROACH SUMMARY")
        print("="*80)
        
        for perception, results in summary.items():
            print(f"\n{perception.upper()}:")
            print(f"  Best Binary: δ={results['best_binary']['delta']}, "
                  f"{results['best_binary']['model']}, F1={results['best_binary']['f1']:.3f}")
            print(f"  Best Multi-class: δ={results['best_multiclass']['delta']}, "
                  f"{results['best_multiclass']['model']}, "
                  f"{results['best_multiclass']['n_classes']} classes, "
                  f"F1={results['best_multiclass']['f1_weighted']:.3f}")
            print(f"  Average Improvement: {results['comparison']['avg_accuracy_improvement']:.3f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Approach Experiment')
    parser.add_argument('--base-dir', type=str, default=None, help='Base directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    experiment = DualApproachExperiment(base_dir=args.base_dir)
    
    if args.quick:
        # Quick test mode
        experiment.config['data']['perceptions'] = ['beautiful']
        experiment.config['delta_values'] = [1.2]
        experiment.config['multiclass']['n_classes'] = [5]
        experiment.config['models'] = ['random_forest']
        
    experiment.run_experiment()


if __name__ == "__main__":
    main()