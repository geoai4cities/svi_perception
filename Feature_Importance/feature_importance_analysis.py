#!/usr/bin/env python3
"""
Feature Importance Analysis for Delta Sensitivity Models
Purpose: Calculate and visualize feature importance for Random Forest models at delta=1.8
Arguments: None (loads saved models automatically)
Returns: Generates publication-ready visualizations and analysis results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from scipy import stats
import warnings
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Add core_scripts to path
sys.path.append(str(Path(__file__).parent.parent / "core_scripts"))

# Import our modules
from model_saver import ModelSaver
from multiclass_delta_sensitivity import MultiClassDeltaSensitivity
from config_loader import ConfigLoader

class FeatureImportanceAnalyzer:
    """Analyze feature importance for trained models."""
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize the feature importance analyzer.
        
        Args:
            base_dir: Base directory for the project
            config_file: Path to configuration file (default: experiment_config.yaml)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Load configuration first
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        
        # Configure model saver with structured output (same as ModelRetrainer)
        output_config = self.config_loader.get_output_config()
        base_dir_config = output_config.get('base_dir', 'outputs/feature_importance_analysis')
        models_dir = output_config.get('models_dir', 'saved_models')
        structured_models_dir = self.base_dir / base_dir_config / models_dir
        self.model_saver = ModelSaver(str(self.base_dir))
        # Override the models directory with structured path
        self.model_saver.models_dir = structured_models_dir
        self.model_saver.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configuration from file
        self.perceptions = self.config_loader.get_perceptions()
        analysis_config = self.config_loader.get_analysis_config()
        self.n_repeats = analysis_config['n_repeats']
        self.alpha = analysis_config['significance_level']
        
        # Output directory using structured configuration
        output_config = self.config_loader.get_output_config()
        base_dir_config = output_config.get('base_dir', 'outputs/feature_importance_analysis')
        results_dir = output_config.get('results_dir', 'results')
        self.output_dir = self.base_dir / base_dir_config / results_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup matplotlib for publication quality
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
    
    def load_test_data(self, perception: str) -> tuple:
        """
        Load test data for a specific perception and convert to binary labels.
        
        Args:
            perception: Perception name
            
        Returns:
            Tuple of (X_test, y_test_binary, feature_names)
        """
        # Load data using configuration
        data_config = self.config_loader.get_data_config()
        input_data_dir = self.base_dir / data_config['input_dir']
        data_file = input_data_dir / f"{perception}_input.xlsx"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_excel(data_file)
        
        # Determine feature count (env overrides config)
        try:
            feature_count = int(os.environ.get('FEATURE_COUNT', data_config.get('feature_count', 36)))
        except Exception:
            feature_count = data_config.get('feature_count', 36)
        
        # Split data (city-based if provided)
        # TEST_CITY_NAME precedence: env > config
        test_city = os.environ.get('TEST_CITY_NAME', '').strip() or str(data_config.get('test_city_name', '')).strip()
        if test_city and 'city_name' in df.columns:
            self.logger.info(f"Using city-based split for test set: city='{test_city}'")
            local_test = df[df['city_name'] == test_city].copy()
            pp_pool = df[df['city_name'] != test_city].copy()
        else:
            test_size = data_config['test_size']
            local_test = df.iloc[-test_size:].copy()
            pp_pool = df.iloc[:-test_size].copy()
        
        # Prepare test data
        X_test = local_test.iloc[:, :feature_count].values
        y_test_continuous = local_test['rating_score'].values
        feature_names = local_test.columns[:feature_count].tolist()
        
        # Convert continuous scores to binary labels using per-perception delta
        # This matches the training process
        delta = self.config_loader.get_delta(perception)
        pp_mean = pp_pool['rating_score'].mean()
        pp_std = pp_pool['rating_score'].std()
        
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
        X_test_filtered = X_test[mask_valid]
        y_test_filtered = y_test_binary[mask_valid]
        
        self.logger.info(f"Test data: {len(y_test_continuous)} total, {len(y_test_filtered)} valid for binary classification (delta={delta})")
        self.logger.info(f"Binary distribution: {np.bincount(y_test_filtered)}")
        
        return X_test_filtered, y_test_filtered, feature_names
    
    def calculate_permutation_importance(self, model, X_test, y_test, feature_names: list) -> pd.DataFrame:
        """
        Calculate permutation importance with statistical significance testing.
        
        Args:
            model: Trained Random Forest model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            DataFrame with importance scores and p-values
        """
        self.logger.info(f"Calculating permutation importance with {self.n_repeats} repeats...")
        
        # Calculate permutation importance
        result = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            n_repeats=self.n_repeats,
            scoring='f1',  # Use F1-score as the performance metric
            random_state=42,
            n_jobs=-1
        )
        
        # Create DataFrame with results
        importances_df = pd.DataFrame(index=pd.Index(feature_names))
        importances_df['mean_importance'] = result.importances_mean
        importances_df['std_importance'] = result.importances_std
        
        # Calculate p-values for statistical significance
        p_values = []
        for i in range(len(feature_names)):
            # Perform a one-sample t-test: are the importance scores significantly different from 0?
            t_stat, p_val = stats.ttest_1samp(result.importances[i, :], 0, alternative='greater')
            p_values.append(p_val)
        
        importances_df['p_value'] = p_values
        
        # Add confidence intervals
        importances_df['ci_lower'] = importances_df['mean_importance'] - 1.96 * importances_df['std_importance']
        importances_df['ci_upper'] = importances_df['mean_importance'] + 1.96 * importances_df['std_importance']
        
        return importances_df
    
    def filter_significant_features(self, importances_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Filter features by statistical significance and select top N.
        
        Args:
            importances_df: DataFrame with importance scores
            top_n: Number of top features to select
            
        Returns:
            Filtered DataFrame with top significant features
        """
        # Filter by statistical significance
        significant_features = importances_df[importances_df['p_value'] < self.alpha].copy()
        
        if len(significant_features) == 0:
            self.logger.warning("No statistically significant features found!")
            # Fall back to top features by importance regardless of significance
            significant_features = importances_df.copy()
        
        # Sort by mean importance and select top N
        top_features = significant_features.sort_values(
            by='mean_importance', 
            ascending=False
        ).head(top_n)
        
        self.logger.info(f"Selected {len(top_features)} top features (p < {self.alpha})")
        
        return top_features
    
    def create_feature_importance_plot(self, all_importances: dict, output_path: Optional[str] = None):
        """
        Create publication-ready 2x2 plot of feature importance.
        
        Args:
            all_importances: Dictionary with perception names as keys and importance DataFrames as values
            output_path: Path to save the plot (optional)
            
        Returns:
            Path to saved plot file
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Define perception order
        perception_order = ['beautiful', 'lively', 'boring', 'safe']
        perception_titles = ['Beautiful', 'Lively', 'Boring', 'Safe']
        # Resolve city name for subtitle (env overrides config)
        try:
            data_cfg = self.config_loader.get_data_config()
            resolved_city = os.environ.get('TEST_CITY_NAME', '').strip() or str(data_cfg.get('test_city_name', '')).strip()
        except Exception:
            resolved_city = ''
        
        for i, (perception, title) in enumerate(zip(perception_order, perception_titles)):
            ax = axes[i]
            
            if perception not in all_importances:
                ax.text(0.5, 0.5, f'No data for {title}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Top 10 Important Features for "{title}"')
                continue
            
            # Get data for this perception
            data_to_plot = all_importances[perception].sort_values(
                by='mean_importance', 
                ascending=True
            )
            
            if len(data_to_plot) == 0:
                ax.text(0.5, 0.5, f'No significant features for {title}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Top 10 Important Features for "{title}"')
                continue
            
            # Create horizontal bar plot
            y_pos = np.arange(len(data_to_plot))
            bars = ax.barh(y_pos, data_to_plot['mean_importance'], 
                          xerr=data_to_plot['std_importance'], 
                          capsize=3, alpha=0.8)
            
            # Color bars by significance
            colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'blue' 
                     for p in data_to_plot['p_value']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data_to_plot.index, fontsize=8)
            ax.set_xlabel('Permutation Importance (Δ F1 on shuffle)', fontsize=10)
            # Fetch model and delta from config for informative titles
            try:
                model_type = self.config_loader.get_model_type(perception)
                delta_value = self.config_loader.get_delta(perception)
                city_part = f" · city={resolved_city}" if resolved_city else ""
                subtitle = f'{title} · {model_type}@δ={delta_value}{city_part}'
            except Exception:
                subtitle = f'{title}'
            ax.set_title(f'Top 10 Important Features — {subtitle}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Add significance indicators
            for j, (idx, row) in enumerate(data_to_plot.iterrows()):
                if row['p_value'] < 0.001:
                    ax.text(row['mean_importance'] + 0.001, j, '***', 
                           va='center', fontsize=8, color='red')
                elif row['p_value'] < 0.01:
                    ax.text(row['mean_importance'] + 0.001, j, '**', 
                           va='center', fontsize=8, color='orange')
                elif row['p_value'] < 0.05:
                    ax.text(row['mean_importance'] + 0.001, j, '*', 
                           va='center', fontsize=8, color='blue')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot in multiple formats
        if output_path is None:
            output_path = str(self.output_dir / "figure_9_feature_importance")
        
        # Save as PNG, PDF, and SVG
        for ext in ['png', 'pdf', 'svg']:
            plot_path = f"{output_path}.{ext}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot: {plot_path}")
        
        plt.show()
        return str(output_path)
    
    def analyze_all_perceptions(self):
        """Analyze feature importance for all perceptions."""
        self.logger.info("="*80)
        self.logger.info("FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("Using configuration-based per-perception settings:")
        for perception in self.perceptions:
            model_type = self.config_loader.get_model_type(perception)
            delta = self.config_loader.get_delta(perception)
            self.logger.info(f"  {perception}: {model_type} (δ={delta})")
        self.logger.info(f"Perceptions: {self.perceptions}")
        self.logger.info("="*80)
        
        all_importances = {}
        all_results = []
        
        for perception in self.perceptions:
            self.logger.info(f"\nAnalyzing perception: {perception.upper()}")
            
            try:
                # Load saved model using configuration
                model_type = self.config_loader.get_model_type(perception)
                delta = self.config_loader.get_delta(perception)
                model_path = self.model_saver.get_latest_model(perception, delta, model_type)
                model_data = self.model_saver.load_model(model_path)
                model = model_data['model']
                
                # Load test data
                X_test, y_test, feature_names = self.load_test_data(perception)
                
                # Calculate permutation importance
                importances_df = self.calculate_permutation_importance(
                    model, X_test, y_test, feature_names
                )
                
                # Filter significant features
                top_features = self.filter_significant_features(importances_df, top_n=10)
                
                # Store results
                all_importances[perception] = top_features
                
                # Store detailed results
                detailed_results = importances_df.copy()
                detailed_results['perception'] = perception
                detailed_results['delta'] = delta
                all_results.append(detailed_results)
                
                self.logger.info(f"✓ Analysis completed for {perception}")
                self.logger.info(f"  - Total features: {len(importances_df)}")
                self.logger.info(f"  - Significant features: {len(importances_df[importances_df['p_value'] < self.alpha])}")
                self.logger.info(f"  - Top feature: {top_features.index[0]} (importance: {top_features.iloc[0]['mean_importance']:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {perception}: {e}")
                continue
        
        # Create visualization
        self.logger.info("\nCreating feature importance visualization...")
        plot_path = self.create_feature_importance_plot(all_importances)
        
        # Save detailed results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            results_file = self.output_dir / "feature_importance_detailed.csv"
            combined_results.to_csv(results_file, index=False)
            self.logger.info(f"Detailed results saved: {results_file}")
        
        # Save summary results
        summary_results = []
        for perception, top_features in all_importances.items():
            for i, (feature, row) in enumerate(top_features.iterrows()):
                summary_results.append({
                    'perception': perception,
                    'rank': i + 1,
                    'feature': feature,
                    'mean_importance': row['mean_importance'],
                    'std_importance': row['std_importance'],
                    'p_value': row['p_value'],
                    'significant': row['p_value'] < self.alpha
                })
        
        summary_df = pd.DataFrame(summary_results)
        summary_file = self.output_dir / "feature_importance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Summary results saved: {summary_file}")
        
        # Print summary
        self.print_summary(all_importances)
        
        return all_importances, summary_df
    
    def print_summary(self, all_importances: dict):
        """Print a summary of the analysis results."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
        print("="*80)
        
        for perception, top_features in all_importances.items():
            print(f"\n{perception.upper()}:")
            print(f"  Top 5 most important features:")
            for i, (feature, row) in enumerate(top_features.head(5).iterrows()):
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"    {i+1}. {feature}: {row['mean_importance']:.4f} ± {row['std_importance']:.4f} {significance}")


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Run analysis
    all_importances, summary_df = analyzer.analyze_all_perceptions()
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
    print("="*80)
    print(f"Results directory: {analyzer.output_dir}")
    print(f"Perceptions analyzed: {len(all_importances)}")
    print(f"Total features analyzed: {len(summary_df)}")


if __name__ == "__main__":
    main()
