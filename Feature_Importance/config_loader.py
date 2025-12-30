#!/usr/bin/env python3
"""
Configuration Loader for Feature Importance Pipeline
Purpose: Load best model and delta configurations from best_config.txt
Arguments: config_file_path (optional)
Returns: Configuration dictionary with per-perception model and delta settings
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

class ConfigLoader:
    """Load and manage configuration for feature importance analysis."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to configuration file (default: experiment_config.yaml)
        """
        if config_file is None:
            config_file = Path(__file__).parent / "experiment_config.yaml"
        
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Complete configuration dictionary
        """
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        self.logger.info(f"Loading configuration from: {self.config_file}")
        
        # Read YAML file
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['experiment', 'data', 'perceptions', 'models', 'analysis', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        self.logger.info(f"Loaded configuration for {len(config['perceptions'])} perceptions:")
        for perception, settings in config['perceptions'].items():
            if settings.get('enabled', True):
                self.logger.info(f"  {perception}: {settings['model']} (δ={settings['delta']})")
        
        return config
    
    def get_model_type(self, perception: str) -> str:
        """
        Get model type for a specific perception.
        
        Args:
            perception: Perception name
            
        Returns:
            Model type ('random_forest' or 'xgboost')
        """
        if perception not in self.config['perceptions']:
            raise ValueError(f"Perception '{perception}' not found in configuration")
        
        return self.config['perceptions'][perception]['model']
    
    def get_delta(self, perception: str) -> float:
        """
        Get delta value for a specific perception.
        
        Args:
            perception: Perception name
            
        Returns:
            Delta value
        """
        if perception not in self.config['perceptions']:
            raise ValueError(f"Perception '{perception}' not found in configuration")
        
        return self.config['perceptions'][perception]['delta']
    
    def get_perceptions(self) -> list:
        """
        Get list of enabled perceptions in configuration.
        
        Returns:
            List of enabled perception names
        """
        return [perception for perception, settings in self.config['perceptions'].items() 
                if settings.get('enabled', True)]
    
    def get_model_delta_pairs(self) -> Dict[str, Tuple[str, float]]:
        """
        Get all enabled model-delta pairs.
        
        Returns:
            Dictionary with perception as key and (model_type, delta) as value
        """
        return {perception: (settings['model'], settings['delta']) 
                for perception, settings in self.config['perceptions'].items()
                if settings.get('enabled', True)}
    
    def get_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model type.
        
        Args:
            model_type: Model type ('random_forest' or 'xgboost')
            
        Returns:
            Dictionary of model parameters
        """
        if model_type not in self.config['models']:
            raise ValueError(f"Model type '{model_type}' not found in configuration")
        
        return self.config['models'][model_type]['parameters']
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config['data']
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis configuration.
        
        Returns:
            Analysis configuration dictionary
        """
        return self.config['analysis']
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration.
        
        Returns:
            Output configuration dictionary
        """
        return self.config['output']
    
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid
        """
        valid_models = {'random_forest', 'xgboost', 'svm', 'realmlp_td', 'realmlp_hpo'}
        valid_perceptions = {'beautiful', 'lively', 'boring', 'safe'}
        
        # Validate perceptions
        for perception, settings in self.config['perceptions'].items():
            if perception not in valid_perceptions:
                self.logger.warning(f"Unknown perception: {perception}")
                return False
            
            if settings['model'] not in valid_models:
                self.logger.warning(f"Unknown model type: {settings['model']}")
                return False
            
            if not isinstance(settings['delta'], (int, float)) or settings['delta'] <= 0:
                self.logger.warning(f"Invalid delta value: {settings['delta']}")
                return False
        
        # Validate models are enabled
        for model_type, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                self.logger.info(f"Model {model_type} is disabled")
        
        # Validate data configuration
        data_config = self.config['data']
        if data_config.get('feature_count', 0) <= 0:
            self.logger.warning("Invalid feature count")
            return False
        
        self.logger.info("Configuration validation passed")
        return True


def main():
    """Test the configuration loader."""
    try:
        loader = ConfigLoader()
        
        print("Configuration loaded successfully:")
        print(f"Experiment: {loader.config['experiment']['name']}")
        print(f"Description: {loader.config['experiment']['description']}")
        
        print(f"\nPerceptions:")
        for perception, settings in loader.config['perceptions'].items():
            if settings.get('enabled', True):
                print(f"  {perception}: {settings['model']} (δ={settings['delta']})")
        
        print(f"\nEnabled perceptions: {loader.get_perceptions()}")
        print(f"Model-delta pairs: {loader.get_model_delta_pairs()}")
        
        print(f"\nData config: {loader.get_data_config()}")
        print(f"Analysis config: {loader.get_analysis_config()}")
        
        print(f"\nValidation: {'PASSED' if loader.validate_config() else 'FAILED'}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


if __name__ == "__main__":
    main()
