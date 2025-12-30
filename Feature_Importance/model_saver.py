#!/usr/bin/env python3
"""
Model Saver Utility for Delta Sensitivity Analysis
Purpose: Save trained models for later feature importance analysis
Arguments: 
    - model: Trained model object
    - perception: Perception name (beautiful, lively, boring, safe)
    - delta: Delta value used for training
    - model_type: Type of model (random_forest, svm, etc.)
    - output_dir: Directory to save the model
Returns: Path to saved model file
"""

import os
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import logging

class ModelSaver:
    """Utility class for saving and loading trained models."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the model saver.
        
        Args:
            base_dir: Base directory for saving models (default: current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        # Prefer new unified output structure if OUTPUT_SUBDIR is provided
        output_root = self.base_dir / "output"
        output_subdir = os.environ.get("OUTPUT_SUBDIR", "").strip()
        
        if output_subdir:
            # Use environment variable if provided
            self.models_dir = output_root / output_subdir / "saved_models"
        else:
            # Use structured configuration (will be set by retrain_and_save_models.py)
            self.models_dir = self.base_dir / "outputs" / "feature_importance_analysis" / "saved_models"
        
        # Don't create directory here - let the calling code decide when to create it
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_model(self, model, perception: str, delta: float, model_type: str, 
                   metadata: dict = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            perception: Perception name (beautiful, lively, boring, safe)
            delta: Delta value used for training
            model_type: Type of model (random_forest, svm, etc.)
            metadata: Additional metadata to save with the model
            
        Returns:
            Path to saved model file
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{perception}_{model_type}_delta_{delta:.1f}_{timestamp}.pkl"
        model_path = self.models_dir / filename
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': model,
            'perception': perception,
            'delta': delta,
            'model_type': model_type,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save model using joblib (better for scikit-learn models)
        try:
            joblib.dump(model_data, model_path)
            self.logger.info(f"Model saved: {model_path}")
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Dictionary containing model and metadata
        """
        try:
            model_data = joblib.load(model_path)
            self.logger.info(f"Model loaded: {model_path}")
            return model_data
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def list_saved_models(self, perception: str = None, delta: float = None, 
                         model_type: str = None) -> list:
        """
        List saved models with optional filtering.
        
        Args:
            perception: Filter by perception name
            delta: Filter by delta value
            model_type: Filter by model type
            
        Returns:
            List of model file paths matching the criteria
        """
        model_files = list(self.models_dir.glob("*.pkl"))
        filtered_files = []
        
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                
                # Apply filters
                if perception and model_data['perception'] != perception:
                    continue
                if delta and abs(model_data['delta'] - delta) > 0.01:
                    continue
                if model_type and model_data['model_type'] != model_type:
                    continue
                    
                filtered_files.append(str(model_file))
            except Exception as e:
                self.logger.warning(f"Could not load model {model_file}: {e}")
                continue
        
        return filtered_files
    
    def get_latest_model(self, perception: str, delta: float, model_type: str = "random_forest") -> str:
        """
        Get the latest saved model for specific parameters.
        
        Args:
            perception: Perception name
            delta: Delta value
            model_type: Model type
            
        Returns:
            Path to the latest model file
        """
        models = self.list_saved_models(perception, delta, model_type)
        if not models:
            raise FileNotFoundError(f"No saved model found for {perception}, delta={delta}, {model_type}")
        
        # Return the most recent model (by filename timestamp)
        return max(models, key=lambda x: Path(x).stem.split('_')[-1])


if __name__ == "__main__":
    # Example usage
    saver = ModelSaver()
    
    # List all saved models
    models = saver.list_saved_models()
    print(f"Found {len(models)} saved models:")
    for model in models:
        print(f"  - {model}")
