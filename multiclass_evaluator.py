#!/usr/bin/env python
"""
Multi-class Classification Evaluator for Delta Sensitivity Analysis
Focuses on multi-class metrics without regression components
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, 
    roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MultiClassEvaluator:
    """
    Evaluator for multi-class classification models.
    Removes all regression metrics and focuses on classification performance.
    """
    
    def __init__(self, n_classes: int = 5):
        """
        Initialize evaluator.
        
        Args:
            n_classes: Number of classes (3, 5, or 10)
        """
        self.n_classes = n_classes
        self.logger = logging.getLogger(__name__)
        
        # Define class labels based on n_classes
        if n_classes == 3:
            self.class_labels = ['Low', 'Medium', 'High']
        elif n_classes == 5:
            self.class_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        elif n_classes == 10:
            self.class_labels = [f'Class {i}' for i in range(10)]
        else:
            self.class_labels = [f'Class {i}' for i in range(n_classes)]
    
    def evaluate_dual_approach(self, model: Any, X_test: np.ndarray, y_test_continuous: np.ndarray,
                              delta: float, pp_mean: float, pp_std: float) -> Dict[str, float]:
        """
        Evaluate model using both binary classification and regression approaches.
        
        Args:
            model: Trained binary classifier
            X_test: Test features
            y_test_continuous: Test labels in original 0-10 scale
            delta: Delta value used for thresholding
            pp_mean: Mean of Place Pulse training data
            pp_std: Standard deviation of Place Pulse training data
            
        Returns:
            Dictionary containing both binary and regression metrics
        """
        metrics = {}
        
        # Apply delta thresholds to create binary labels
        neg_threshold = pp_mean - delta * pp_std
        pos_threshold = pp_mean + delta * pp_std
        
        # Create binary labels for test data
        y_test_binary = np.zeros(len(y_test_continuous), dtype=int)
        mask_neg = y_test_continuous < neg_threshold
        mask_pos = y_test_continuous > pos_threshold
        mask_mid = (~mask_neg) & (~mask_pos)
        
        y_test_binary[mask_neg] = 0
        y_test_binary[mask_pos] = 1
        
        # Filter out mid-range samples for binary evaluation
        mask_valid = ~mask_mid
        
        if mask_valid.sum() > 0:
            X_test_filtered = X_test[mask_valid]
            y_test_filtered = y_test_binary[mask_valid]
            
            try:
                # Get predictions
                y_pred = model.predict(X_test_filtered)
                y_proba = model.predict_proba(X_test_filtered)
                
                # Binary classification metrics
                metrics['f1_binary'] = f1_score(y_test_filtered, y_pred, average='binary')
                metrics['accuracy_binary'] = accuracy_score(y_test_filtered, y_pred)
                
                if len(np.unique(y_test_filtered)) > 1:
                    metrics['roc_auc_binary'] = roc_auc_score(y_test_filtered, y_proba[:, 1])
                else:
                    metrics['roc_auc_binary'] = 0.5
                    
            except Exception as e:
                self.logger.warning(f"Binary evaluation error: {e}")
                metrics['f1_binary'] = 0.0
                metrics['accuracy_binary'] = 0.0
                metrics['roc_auc_binary'] = 0.5
        else:
            metrics['f1_binary'] = 0.0
            metrics['accuracy_binary'] = 0.0
            metrics['roc_auc_binary'] = 0.5
            
        # Add sample counts
        metrics['n_pos_binary'] = mask_pos.sum()
        metrics['n_neg_binary'] = mask_neg.sum()
        metrics['n_mid_binary'] = mask_mid.sum()
        
        # Regression evaluation (probability to score conversion)
        try:
            y_proba_all = model.predict_proba(X_test)
            predicted_scores = y_proba_all[:, 1] * 10  # Convert to 0-10 scale
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics['mse_regression'] = mean_squared_error(y_test_continuous, predicted_scores)
            metrics['mae_regression'] = mean_absolute_error(y_test_continuous, predicted_scores)
            metrics['r2_regression'] = r2_score(y_test_continuous, predicted_scores)
            
        except Exception as e:
            self.logger.warning(f"Regression evaluation error: {e}")
            metrics['mse_regression'] = 999.0
            metrics['mae_regression'] = 999.0
            metrics['r2_regression'] = -1.0
            
        return metrics
    
    def evaluate_multiclass(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive multi-class classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Agreement metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred)
        metrics['per_class'] = per_class_metrics
        
        # Probability-based metrics if available
        if y_proba is not None:
            prob_metrics = self._compute_probability_metrics(y_true, y_proba)
            metrics.update(prob_metrics)
        
        # Class distribution in predictions
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = dict(zip(unique.tolist(), counts.tolist()))
        metrics['predicted_distribution'] = pred_distribution
        
        # Actual class distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        true_distribution = dict(zip(unique_true.tolist(), counts_true.tolist()))
        metrics['true_distribution'] = true_distribution
        
        return metrics
    
    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute metrics for each class individually.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Per-class metrics dictionary
        """
        per_class = {}
        
        for class_idx in range(self.n_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            class_name = self.class_labels[class_idx] if class_idx < len(self.class_labels) else f'Class {class_idx}'
            
            per_class[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true == class_idx)
            }
        
        return per_class
    
    def _compute_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Compute probability-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Probability-based metrics
        """
        prob_metrics = {}
        
        # Log loss
        try:
            prob_metrics['log_loss'] = log_loss(y_true, y_proba)
        except:
            prob_metrics['log_loss'] = np.nan
        
        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            # Binarize labels for multi-class ROC-AUC
            y_true_bin = label_binarize(y_true, classes=list(range(self.n_classes)))
            
            # Ensure y_proba has the right shape
            if y_proba.shape[1] == self.n_classes:
                prob_metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_proba, 
                                                           multi_class='ovr', average='macro')
                prob_metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_proba, 
                                                           multi_class='ovo', average='macro')
        except:
            prob_metrics['roc_auc_ovr'] = np.nan
            prob_metrics['roc_auc_ovo'] = np.nan
        
        # Top-k accuracy
        prob_metrics['top_2_accuracy'] = self._top_k_accuracy(y_true, y_proba, k=2)
        if self.n_classes >= 3:
            prob_metrics['top_3_accuracy'] = self._top_k_accuracy(y_true, y_proba, k=3)
        
        # Confidence metrics
        max_probs = np.max(y_proba, axis=1)
        prob_metrics['mean_confidence'] = np.mean(max_probs)
        prob_metrics['std_confidence'] = np.std(max_probs)
        
        # Calibration error
        prob_metrics['expected_calibration_error'] = self._compute_ece(y_true, y_proba)
        
        return prob_metrics
    
    def _top_k_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy score
        """
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def _compute_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        y_pred = np.argmax(y_proba, axis=1)
        max_probs = np.max(y_proba, axis=1)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_pred[bin_mask] == y_true[bin_mask])
                bin_confidence = np.mean(max_probs[bin_mask])
                bin_weight = np.sum(bin_mask) / len(y_true)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def evaluate_model_generalization(self, train_metrics: Dict, val_metrics: Dict, 
                                    test_metrics: Dict) -> Dict:
        """
        Analyze model generalization across train/val/test sets.
        
        Args:
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            test_metrics: Test set metrics
            
        Returns:
            Generalization analysis
        """
        generalization = {
            'train_val_gap': {},
            'val_test_gap': {},
            'train_test_gap': {},
            'overfitting_score': 0.0,
            'generalization_score': 0.0
        }
        
        # Key metrics for comparison
        key_metrics = ['accuracy', 'f1_macro', 'balanced_accuracy']
        
        for metric in key_metrics:
            if metric in train_metrics and metric in val_metrics:
                generalization['train_val_gap'][metric] = train_metrics[metric] - val_metrics[metric]
            
            if metric in val_metrics and metric in test_metrics:
                generalization['val_test_gap'][metric] = val_metrics[metric] - test_metrics[metric]
            
            if metric in train_metrics and metric in test_metrics:
                generalization['train_test_gap'][metric] = train_metrics[metric] - test_metrics[metric]
        
        # Compute overfitting score (higher means more overfitting)
        if 'accuracy' in train_metrics and 'accuracy' in test_metrics:
            generalization['overfitting_score'] = train_metrics['accuracy'] - test_metrics['accuracy']
        
        # Compute generalization score (higher is better)
        if 'f1_macro' in test_metrics and 'f1_macro' in val_metrics:
            generalization['generalization_score'] = 1.0 - abs(test_metrics['f1_macro'] - val_metrics['f1_macro'])
        
        return generalization
    
    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    title: str = "Confusion Matrix") -> plt.Figure:
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_labels[:cm.shape[1]], 
                   yticklabels=self.class_labels[:cm.shape[0]],
                   ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        return fig
    
    def create_generalization_curves(self, results_df: pd.DataFrame, 
                                    metric: str = 'f1_macro') -> plt.Figure:
        """
        Create generalization curves showing train/val/test performance.
        
        Args:
            results_df: DataFrame with columns: delta, split (train/val/test), metric values
            metric: Metric to plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot for each model
        models = results_df['model'].unique()
        
        for idx, model in enumerate(models[:4]):  # Max 4 models for 2x2 grid
            ax = axes[idx // 2, idx % 2]
            
            model_data = results_df[results_df['model'] == model]
            
            # Plot lines for train, val, test
            for split in ['train', 'val', 'test']:
                split_data = model_data[model_data['split'] == split]
                if not split_data.empty:
                    ax.plot(split_data['delta'], split_data[metric], 
                           marker='o', label=split.capitalize())
            
            ax.set_xlabel('Delta (δ)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{model.upper()} - Generalization Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Model Generalization Across Delta Values\nMetric: {metric}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


class ModelSaver:
    """
    Saves best models for each perception based on multi-class performance.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize model saver.
        
        Args:
            save_dir: Directory to save models
        """
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)
        self.best_models = {}  # Track best model per perception
    
    def update_best_model(self, perception: str, delta: float, model_name: str,
                         model: Any, metrics: Dict, save_criterion: str = 'f1_macro'):
        """
        Update and save best model if current model is better.
        
        Args:
            perception: Perception name
            delta: Delta value
            model_name: Model name
            model: Trained model
            metrics: Model metrics
            save_criterion: Metric to use for comparison
        """
        import joblib
        
        current_score = metrics.get(save_criterion, 0)
        
        # Check if this is the best model so far
        if perception not in self.best_models:
            self.best_models[perception] = {
                'score': current_score,
                'model_name': model_name,
                'delta': delta,
                'model': model,
                'metrics': metrics
            }
            is_best = True
        else:
            if current_score > self.best_models[perception]['score']:
                self.best_models[perception] = {
                    'score': current_score,
                    'model_name': model_name,
                    'delta': delta,
                    'model': model,
                    'metrics': metrics
                }
                is_best = True
            else:
                is_best = False
        
        if is_best:
            # Save the best model
            model_dir = os.path.join(self.save_dir, perception, 'best_model')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, f'{model_name}_delta_{delta:.1f}.joblib')
            joblib.dump(model, model_path)
            
            # Save metrics
            metrics_path = os.path.join(model_dir, 'best_model_metrics.json')
            import json
            with open(metrics_path, 'w') as f:
                save_metrics = {
                    'model_name': model_name,
                    'delta': delta,
                    'score': current_score,
                    'criterion': save_criterion,
                    'all_metrics': {k: v for k, v in metrics.items() 
                                   if not isinstance(v, (list, np.ndarray))}
                }
                json.dump(save_metrics, f, indent=2)
            
            self.logger.info(f"New best model for {perception}: {model_name} "
                           f"(δ={delta}, {save_criterion}={current_score:.4f})")
    
    def get_best_models_summary(self) -> pd.DataFrame:
        """
        Get summary of best models for all perceptions.
        
        Returns:
            DataFrame with best model information
        """
        summary = []
        for perception, info in self.best_models.items():
            summary.append({
                'perception': perception,
                'best_model': info['model_name'],
                'delta': info['delta'],
                'score': info['score'],
                'accuracy': info['metrics'].get('accuracy', 0),
                'f1_macro': info['metrics'].get('f1_macro', 0),
                'balanced_accuracy': info['metrics'].get('balanced_accuracy', 0)
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Initialize evaluator
    evaluator = MultiClassEvaluator(n_classes=n_classes)
    
    # Evaluate
    metrics = evaluator.evaluate_multiclass(y_true, y_pred, y_proba)
    
    print("Multi-class Classification Metrics:")
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'per_class']:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nPer-class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}: F1={class_metrics['f1']:.3f}, "
              f"Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}")