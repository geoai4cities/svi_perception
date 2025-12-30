#!/usr/bin/env python
"""
Enhanced Delta Sensitivity Analysis with Multi-class Classification Support
Supports both binary and multi-class classification approaches
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MultiClassDeltaSensitivity:
    """
    Enhanced Delta Sensitivity Analyzer supporting multi-class classification.
    
    Instead of binary classification (-1, 1), this creates multiple classes
    based on score ranges for more nuanced prediction.
    """
    
    def __init__(self, random_state: int = 42, n_classes: int = 5):
        """
        Initialize analyzer.
        
        Args:
            random_state: Random seed for reproducibility
            n_classes: Number of classes to create (3, 5, or 10)
        """
        self.random_state = random_state
        self.n_classes = n_classes
        self.logger = logging.getLogger(__name__)
        
        # Define class ranges based on n_classes
        if n_classes == 3:
            # Low (0-3.33), Medium (3.33-6.67), High (6.67-10)
            self.class_boundaries = [0, 3.33, 6.67, 10]
            self.class_labels = ['low', 'medium', 'high']
        elif n_classes == 5:
            # Very Low, Low, Medium, High, Very High
            self.class_boundaries = [0, 2, 4, 6, 8, 10]
            self.class_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
        elif n_classes == 10:
            # Integer classes 0-9
            self.class_boundaries = list(range(11))
            self.class_labels = [f'class_{i}' for i in range(10)]
        else:
            raise ValueError(f"n_classes must be 3, 5, or 10, got {n_classes}")
    
    def create_multiclass_labels(self, scores: np.ndarray) -> np.ndarray:
        """
        Create multi-class labels from continuous scores.
        
        Args:
            scores: Continuous scores (0-10 scale)
            
        Returns:
            Integer class labels (0 to n_classes-1)
        """
        labels = np.digitize(scores, self.class_boundaries[1:-1])
        return labels
    
    def create_delta_based_labels(self, scores: np.ndarray, delta: float) -> Dict:
        """
        Create both binary and multi-class labels using delta thresholds.
        
        Args:
            scores: Continuous scores (0-10 scale)
            delta: Delta value for threshold calculation
            
        Returns:
            Dictionary with binary and multi-class labels
        """
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Binary classification with delta thresholds
        neg_threshold = mean_score - delta * std_score
        pos_threshold = mean_score + delta * std_score
        
        binary_labels = np.zeros(len(scores), dtype=int)
        binary_labels[scores < neg_threshold] = 0  # Negative class
        binary_labels[scores > pos_threshold] = 2  # Positive class
        binary_labels[(scores >= neg_threshold) & (scores <= pos_threshold)] = 1  # Mid-range
        
        # Multi-class classification
        multiclass_labels = self.create_multiclass_labels(scores)
        
        # Delta-enhanced multi-class (combines delta concept with multi-class)
        # Creates finer distinctions near the thresholds
        delta_multiclass = np.zeros(len(scores), dtype=int)
        
        if self.n_classes == 5:
            # 5-class system based on delta regions
            delta_multiclass[scores < neg_threshold] = 0  # Very negative
            delta_multiclass[(scores >= neg_threshold) & (scores < mean_score - 0.5*std_score)] = 1  # Negative
            delta_multiclass[(scores >= mean_score - 0.5*std_score) & (scores <= mean_score + 0.5*std_score)] = 2  # Neutral
            delta_multiclass[(scores > mean_score + 0.5*std_score) & (scores <= pos_threshold)] = 3  # Positive
            delta_multiclass[scores > pos_threshold] = 4  # Very positive
        else:
            # Use standard multi-class for other configurations
            delta_multiclass = multiclass_labels
        
        return {
            'binary': binary_labels,
            'binary_filtered': binary_labels[binary_labels != 1],  # Remove mid-range
            'multiclass': multiclass_labels,
            'delta_multiclass': delta_multiclass,
            'thresholds': {
                'mean': mean_score,
                'std': std_score,
                'negative': neg_threshold,
                'positive': pos_threshold
            }
        }
    
    def analyze_delta_sensitivity(self, pp_features: np.ndarray, pp_scores: np.ndarray,
                                 local_features: np.ndarray, local_scores: np.ndarray,
                                 delta_values: List[float]) -> Dict:
        """
        Analyze sensitivity across multiple delta values.
        
        Args:
            pp_features: Place Pulse features
            pp_scores: Place Pulse scores
            local_features: Local test features
            local_scores: Local test scores
            delta_values: List of delta values to test
            
        Returns:
            Dictionary with results for each delta value
        """
        results = {}
        
        for delta in delta_values:
            self.logger.info(f"Processing delta={delta}")
            
            # Create labels for PP data
            pp_labels = self.create_delta_based_labels(pp_scores, delta)
            
            # Analyze class distributions
            pp_dist = self.analyze_class_distribution(pp_labels, 'PP')
            
            # Check viability
            is_viable = self.check_delta_viability(pp_labels)
            
            if is_viable:
                # Create train/val splits for each label type
                splits = {}
                
                # Binary classification (filtered)
                binary_mask = pp_labels['binary'] != 1
                if np.sum(binary_mask) > 100:
                    binary_features = pp_features[binary_mask]
                    binary_labels = pp_labels['binary'][binary_mask]
                    binary_labels[binary_labels == 2] = 1  # Convert to 0,1
                    
                    splits['binary'] = self.create_stratified_split(
                        binary_features, binary_labels, 0.25
                    )
                
                # Multi-class classification
                splits['multiclass'] = self.create_stratified_split(
                    pp_features, pp_labels['multiclass'], 0.25
                )
                
                # Delta-enhanced multi-class
                splits['delta_multiclass'] = self.create_stratified_split(
                    pp_features, pp_labels['delta_multiclass'], 0.25
                )
                
                # Store results
                results[delta] = {
                    'is_viable': True,
                    'pp_labels': pp_labels,
                    'pp_distribution': pp_dist,
                    'splits': splits,
                    'thresholds': pp_labels['thresholds']
                }
            else:
                results[delta] = {
                    'is_viable': False,
                    'pp_distribution': pp_dist,
                    'reason': 'Insufficient samples or class imbalance'
                }
        
        return results
    
    def analyze_class_distribution(self, labels: Dict, name: str = "") -> Dict:
        """
        Analyze class distribution for all label types.
        
        Args:
            labels: Dictionary with different label types
            name: Dataset name for logging
            
        Returns:
            Distribution statistics
        """
        distributions = {}
        
        for label_type, label_array in labels.items():
            if label_type == 'thresholds':
                continue
                
            if isinstance(label_array, np.ndarray):
                unique, counts = np.unique(label_array, return_counts=True)
                dist = dict(zip(unique.tolist(), counts.tolist()))
                
                distributions[label_type] = {
                    'distribution': dist,
                    'n_classes': len(unique),
                    'total_samples': len(label_array),
                    'min_class_size': min(counts) if len(counts) > 0 else 0,
                    'max_class_size': max(counts) if len(counts) > 0 else 0,
                    'balance_ratio': min(counts) / max(counts) if len(counts) > 0 and max(counts) > 0 else 0
                }
                
                self.logger.debug(f"{name} - {label_type}: {dist}")
        
        return distributions
    
    def check_delta_viability(self, labels: Dict, min_samples_per_class: int = 30,
                            min_total_samples: int = 100) -> bool:
        """
        Check if delta value produces viable splits.
        
        Args:
            labels: Label dictionary
            min_samples_per_class: Minimum samples per class
            min_total_samples: Minimum total samples
            
        Returns:
            True if viable, False otherwise
        """
        # Check binary viability
        binary_filtered = labels['binary'][labels['binary'] != 1]
        if len(binary_filtered) < min_total_samples:
            return False
        
        unique, counts = np.unique(binary_filtered, return_counts=True)
        if len(unique) < 2 or min(counts) < min_samples_per_class:
            return False
        
        # Check multi-class viability
        unique_mc, counts_mc = np.unique(labels['multiclass'], return_counts=True)
        if len(unique_mc) < 2:
            return False
        
        return True
    
    def create_stratified_split(self, features: np.ndarray, labels: np.ndarray,
                               val_size: float = 0.25) -> Dict:
        """
        Create stratified train/validation split.
        
        Args:
            features: Feature array
            labels: Label array
            val_size: Validation set proportion
            
        Returns:
            Split data dictionary
        """
        try:
            # Balance classes if needed
            balanced_features, balanced_labels = self.balance_classes(features, labels)
            
            # Create stratified split
            X_train, X_val, y_train, y_val = train_test_split(
                balanced_features, balanced_labels,
                test_size=val_size,
                stratify=balanced_labels,
                random_state=self.random_state
            )
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'n_train': len(y_train),
                'n_val': len(y_val)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create split: {e}")
            return None
    
    def balance_classes(self, features: np.ndarray, labels: np.ndarray,
                       strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using appropriate strategy.
        
        Args:
            features: Feature array
            labels: Label array
            strategy: 'auto', 'upsample', 'downsample', or 'none'
            
        Returns:
            Balanced features and labels
        """
        if strategy == 'none':
            return features, labels
        
        unique, counts = np.unique(labels, return_counts=True)
        
        if strategy == 'auto':
            # Decide strategy based on class distribution
            min_count = min(counts)
            max_count = max(counts)
            
            if min_count / max_count < 0.5:
                # Significant imbalance, use upsampling
                strategy = 'upsample'
            else:
                # Reasonable balance, no action needed
                return features, labels
        
        balanced_features = []
        balanced_labels = []
        
        if strategy == 'upsample':
            max_count = max(counts)
            for class_label in unique:
                class_mask = labels == class_label
                class_features = features[class_mask]
                class_labels = labels[class_mask]
                
                if len(class_labels) < max_count:
                    # Upsample minority class
                    indices = resample(
                        np.arange(len(class_labels)),
                        n_samples=max_count,
                        random_state=self.random_state
                    )
                    balanced_features.append(class_features[indices])
                    balanced_labels.append(class_labels[indices])
                else:
                    balanced_features.append(class_features)
                    balanced_labels.append(class_labels)
        
        elif strategy == 'downsample':
            min_count = min(counts)
            for class_label in unique:
                class_mask = labels == class_label
                class_features = features[class_mask]
                class_labels = labels[class_mask]
                
                if len(class_labels) > min_count:
                    # Downsample majority class
                    indices = resample(
                        np.arange(len(class_labels)),
                        n_samples=min_count,
                        random_state=self.random_state,
                        replace=False
                    )
                    balanced_features.append(class_features[indices])
                    balanced_labels.append(class_labels[indices])
                else:
                    balanced_features.append(class_features)
                    balanced_labels.append(class_labels)
        
        if balanced_features:
            return np.vstack(balanced_features), np.hstack(balanced_labels)
        else:
            return features, labels
    
    def evaluate_multiclass_local_test(self, model: Any, X_test: np.ndarray, 
                                      y_test_scores: np.ndarray, delta: float,
                                      pp_stats: Dict) -> Dict:
        """
        Evaluate model on local test data with multi-class approach.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test_scores: Test scores (0-10 scale)
            delta: Delta value used for training
            pp_stats: PP statistics (thresholds)
            
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                    f1_score, confusion_matrix, mean_squared_error,
                                    mean_absolute_error, r2_score)
        
        # Create test labels using same approach as training
        test_labels = self.create_delta_based_labels(y_test_scores, delta)
        
        metrics = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Multi-class classification metrics
        metrics['accuracy'] = accuracy_score(test_labels['multiclass'], y_pred)
        metrics['precision_macro'] = precision_score(test_labels['multiclass'], y_pred, 
                                                    average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(test_labels['multiclass'], y_pred, 
                                              average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(test_labels['multiclass'], y_pred, 
                                      average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels['multiclass'], y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Convert predictions back to continuous scores for regression metrics
        if hasattr(model, 'predict_proba'):
            # Use probability-weighted average for score prediction
            y_proba = model.predict_proba(X_test)
            
            # Calculate expected score based on class probabilities
            class_centers = [(self.class_boundaries[i] + self.class_boundaries[i+1])/2 
                           for i in range(self.n_classes)]
            predicted_scores = np.sum(y_proba * np.array(class_centers), axis=1)
        else:
            # Use class centers for predicted classes
            class_centers = [(self.class_boundaries[i] + self.class_boundaries[i+1])/2 
                           for i in range(self.n_classes)]
            predicted_scores = np.array([class_centers[int(p)] for p in y_pred])
        
        # Regression metrics on original scale
        metrics['mse'] = mean_squared_error(y_test_scores, predicted_scores)
        metrics['mae'] = mean_absolute_error(y_test_scores, predicted_scores)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test_scores, predicted_scores)
        
        # Correlation
        correlation = np.corrcoef(y_test_scores, predicted_scores)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0
        
        return metrics


if __name__ == "__main__":
    # Test the multi-class implementation
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    features = np.random.randn(n_samples, 10)
    scores = np.random.uniform(0, 10, n_samples)
    
    # Initialize analyzer
    analyzer = MultiClassDeltaSensitivity(n_classes=5)
    
    # Test label creation
    labels = analyzer.create_delta_based_labels(scores, delta=1.0)
    
    print(f"Binary distribution: {np.unique(labels['binary'], return_counts=True)}")
    print(f"Multi-class distribution: {np.unique(labels['multiclass'], return_counts=True)}")
    print(f"Delta multi-class distribution: {np.unique(labels['delta_multiclass'], return_counts=True)}")