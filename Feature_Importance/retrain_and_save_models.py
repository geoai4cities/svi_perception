#!/usr/bin/env python3
"""
Retrain and Save Models for Feature Importance Analysis
Purpose: Retrain Random Forest models for delta=1.8 and save them for feature importance analysis
Arguments: None (uses configuration from experiment)
Returns: Saves trained models to Feature_importance/saved_models/
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
import warnings
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Only Random Forest will be used.")

# Try to import RealMLP
try:
    from pytabkit import RealMLP_TD_Classifier, RealMLP_HPO_Classifier
    REALMLP_AVAILABLE = True
except ImportError:
    REALMLP_AVAILABLE = False
    print("Warning: RealMLP not available. Only Random Forest and XGBoost will be used.")

# Add core_scripts to path
sys.path.append(str(Path(__file__).parent.parent / "core_scripts"))

# Import our modules
from model_saver import ModelSaver
from multiclass_delta_sensitivity import MultiClassDeltaSensitivity
from config_loader import ConfigLoader

class ModelRetrainer:
    """Retrain and save models for feature importance analysis."""
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize the model retrainer.
        
        Args:
            base_dir: Base directory for the project
            config_file: Path to configuration file (default: best_delta/best_config.txt)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Load configuration first
        self.config_loader = ConfigLoader(config_file)
        
        # Configure model saver with structured output
        output_config = self.config_loader.get_output_config()
        base_dir = output_config.get('base_dir', 'outputs/feature_importance_analysis')
        models_dir = output_config.get('models_dir', 'saved_models')
        structured_models_dir = self.base_dir / base_dir / models_dir
        self.model_saver = ModelSaver(str(self.base_dir))
        # Override the models directory with structured path
        self.model_saver.models_dir = structured_models_dir
        self.model_saver.models_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.config_loader.config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configuration from file
        self.perceptions = self.config_loader.get_perceptions()
        data_config = self.config_loader.get_data_config()
        self.test_size = data_config['test_size']
        self.random_state = data_config['random_state']
        
        # Validate configuration
        if not self.config_loader.validate_config():
            raise ValueError("Invalid configuration loaded")
        
        self.logger.info(f"Using configuration-based training for {len(self.perceptions)} perceptions")
        
    def load_data(self, perception: str) -> tuple:
        """
        Load and prepare data for a specific perception.
        
        Args:
            perception: Perception name
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        # Load data using configuration
        data_config = self.config_loader.get_data_config()
        input_data_dir = self.base_dir / data_config['input_dir']
        data_file = input_data_dir / f"{perception}_input.xlsx"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.logger.info(f"Loading data from: {data_file}")
        df = pd.read_excel(data_file)
        
        # Determine feature count (env overrides config)
        try:
            feature_count = int(os.environ.get('FEATURE_COUNT', data_config.get('feature_count', 36)))
        except Exception:
            feature_count = data_config.get('feature_count', 36)
        
        # City-based split if TEST_CITY_NAME is provided and column exists
        # TEST_CITY_NAME precedence: env > config
        test_city = os.environ.get('TEST_CITY_NAME', '').strip() or str(data_config.get('test_city_name', '')).strip()
        if test_city and 'city_name' in df.columns:
            self.logger.info(f"Using city-based split for test set: city='{test_city}'")
            local_test = df[df['city_name'] == test_city].copy()
            pp_pool = df[df['city_name'] != test_city].copy()
        else:
            # Fallback to last-N split
            local_test = df.iloc[-self.test_size:].copy()
            pp_pool = df.iloc[:-self.test_size].copy()
        
        self.logger.info(f"Place Pulse pool: {len(pp_pool)} samples")
        self.logger.info(f"Local test set: {len(local_test)} samples")
        
        # Create delta-based labels using configuration
        delta = self.config_loader.get_delta(perception)
        analyzer = MultiClassDeltaSensitivity(random_state=self.random_state)
        delta_results = analyzer.create_delta_based_labels(
            pp_pool['rating_score'].values, 
            delta
        )
        
        # Get binary labels
        binary_labels = delta_results['binary']
        
        # Filter out mid-range samples
        mask = binary_labels != 1  # Remove mid-range
        if mask.sum() < 100:
            raise ValueError(f"Too few samples for {perception} with delta={delta}: {mask.sum()}")
        
        # Prepare training data
        X_full = pp_pool.iloc[:, :feature_count].values[mask]
        y_full = binary_labels[mask]
        y_full[y_full == 2] = 1  # Convert to 0/1 labels
        
        # Get feature names
        feature_names = pp_pool.columns[:feature_count].tolist()
        
        # Create train/validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=self.random_state, stratify=y_full
        )
        
        # Prepare test data
        X_test = local_test.iloc[:, :feature_count].values
        y_test = local_test['rating_score'].values
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    
    def train_model(self, X_train, y_train, X_val, y_val, perception: str):
        """
        Train a model based on the configuration for the specific perception.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            perception: Perception name to get model type from config
            
        Returns:
            Trained model
        """
        model_type = self.config_loader.get_model_type(perception)
        
        if model_type == 'random_forest':
            return self.train_random_forest(X_train, y_train, X_val, y_val)
        elif model_type == 'xgboost':
            return self.train_xgboost(X_train, y_train, X_val, y_val)
        elif model_type == 'realmlp_td':
            return self.train_realmlp_td(X_train, y_train, X_val, y_val)
        elif model_type == 'realmlp_hpo':
            return self.train_realmlp_hpo(X_train, y_train, X_val, y_val)
        elif model_type == 'svm':
            return self.train_svm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_random_forest(self, X_train, y_train, X_val, y_val) -> RandomForestClassifier:
        """
        Train a Random Forest classifier with parameters from configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained Random Forest model
        """
        # Get parameters from configuration
        params = self.config_loader.get_model_parameters('random_forest')
        
        model = RandomForestClassifier(**params)
        
        self.logger.info("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='binary')
        acc_val = accuracy_score(y_val, y_pred_val)
        
        self.logger.info(f"Validation F1: {f1_val:.3f}, Accuracy: {acc_val:.3f}")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train an XGBoost classifier with parameters from configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        # Get parameters from configuration
        params = self.config_loader.get_model_parameters('xgboost')
        
        model = xgb.XGBClassifier(**params)
        
        self.logger.info("Training XGBoost model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='binary')
        acc_val = accuracy_score(y_val, y_pred_val)
        
        self.logger.info(f"Validation F1: {f1_val:.3f}, Accuracy: {acc_val:.3f}")
        
        return model

    def train_svm(self, X_train, y_train, X_val, y_val) -> SVC:
        """
        Train an SVM classifier with parameters from configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained SVM model
        """
        params = self.config_loader.get_model_parameters('svm')
        model = SVC(**params)
        
        self.logger.info("Training SVM model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='binary')
        acc_val = accuracy_score(y_val, y_pred_val)
        
        self.logger.info(f"Validation F1: {f1_val:.3f}, Accuracy: {acc_val:.3f}")
        
        return model
    
    def train_realmlp_td(self, X_train, y_train, X_val, y_val):
        """
        Train a RealMLP TD classifier with parameters from configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained RealMLP TD model
        """
        if not REALMLP_AVAILABLE:
            raise ImportError("RealMLP is not available")
        
        # Get parameters from configuration
        params = self.config_loader.get_model_parameters('realmlp_td')
        
        model = RealMLP_TD_Classifier(**params)
        
        self.logger.info("Training RealMLP TD model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='binary')
        acc_val = accuracy_score(y_val, y_pred_val)
        
        self.logger.info(f"Validation F1: {f1_val:.3f}, Accuracy: {acc_val:.3f}")
        
        return model
    
    def train_realmlp_hpo(self, X_train, y_train, X_val, y_val):
        """
        Train a RealMLP HPO classifier with parameters from configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained RealMLP HPO model
        """
        if not REALMLP_AVAILABLE:
            raise ImportError("RealMLP is not available")
        
        # Get parameters from configuration
        params = self.config_loader.get_model_parameters('realmlp_hpo')
        
        model = RealMLP_HPO_Classifier(**params)
        
        self.logger.info("Training RealMLP HPO model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='binary')
        acc_val = accuracy_score(y_val, y_pred_val)
        
        self.logger.info(f"Validation F1: {f1_val:.3f}, Accuracy: {acc_val:.3f}")
        
        return model
    
    def retrain_all_models(self):
        """Retrain and save all models with per-perception configurations."""
        self.logger.info("="*60)
        self.logger.info("RETRAINING MODELS FOR FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("Using configuration-based per-perception settings:")
        for perception in self.perceptions:
            model_type = self.config_loader.get_model_type(perception)
            delta = self.config_loader.get_delta(perception)
            self.logger.info(f"  {perception}: {model_type} (δ={delta})")
        self.logger.info(f"Perceptions: {self.perceptions}")
        self.logger.info("="*60)
        
        saved_models = {}
        
        for perception in self.perceptions:
            self.logger.info(f"\nProcessing perception: {perception.upper()}")
            
            try:
                # Load data
                X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.load_data(perception)
                
                # Train model using configuration
                model = self.train_model(X_train, y_train, X_val, y_val, perception)
                
                # Prepare metadata
                model_type = self.config_loader.get_model_type(perception)
                delta = self.config_loader.get_delta(perception)
                metadata = {
                    'feature_names': feature_names,
                    'n_features': len(feature_names),
                    'n_train_samples': len(X_train),
                    'n_val_samples': len(X_val),
                    'n_test_samples': len(X_test),
                    'train_class_distribution': np.bincount(y_train).tolist(),
                    'val_class_distribution': np.bincount(y_val).tolist(),
                    'delta': delta,
                    'perception': perception,
                    'model_type': model_type
                }
                
                # Save model
                model_path = self.model_saver.save_model(
                    model=model,
                    perception=perception,
                    delta=delta,
                    model_type=model_type,
                    metadata=metadata
                )
                
                saved_models[perception] = {
                    'model_path': model_path,
                    'model': model,
                    'feature_names': feature_names,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                self.logger.info(f"✓ Model saved for {perception}: {model_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {perception}: {e}")
                continue
        
        self.logger.info(f"\n✓ Successfully saved {len(saved_models)} models")
        return saved_models
    
    def verify_models(self, saved_models: dict):
        """Verify that saved models can be loaded and work correctly."""
        self.logger.info("\n" + "="*60)
        self.logger.info("VERIFYING SAVED MODELS")
        self.logger.info("="*60)
        
        for perception, model_info in saved_models.items():
            try:
                # Load model
                loaded_data = self.model_saver.load_model(model_info['model_path'])
                
                # Verify model properties
                assert loaded_data['perception'] == perception
                assert loaded_data['delta'] == self.delta_config[perception]
                assert loaded_data['model_type'] == 'random_forest'
                
                # Test prediction
                model = loaded_data['model']
                X_test = model_info['X_test']
                y_pred = model.predict(X_test[:10])  # Test on first 10 samples
                
                self.logger.info(f"✓ {perception}: Model loads and predicts correctly")
                
            except Exception as e:
                self.logger.error(f"✗ {perception}: Verification failed - {e}")


def main():
    """Main execution function."""
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    # Retrain and save all models
    saved_models = retrainer.retrain_all_models()
    
    # Verify saved models
    retrainer.verify_models(saved_models)
    
    print("\n" + "="*60)
    print("MODEL RETRAINING COMPLETED")
    print("="*60)
    print(f"Saved models directory: {retrainer.model_saver.models_dir}")
    print(f"Models saved: {len(saved_models)}")
    for perception in saved_models:
        print(f"  - {perception}: {saved_models[perception]['model_path']}")


if __name__ == "__main__":
    main()
