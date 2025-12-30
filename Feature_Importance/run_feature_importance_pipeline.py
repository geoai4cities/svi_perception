#!/usr/bin/env python3
"""
Complete Feature Importance Pipeline
Purpose: Run the complete pipeline: retrain models → save models → analyze feature importance → generate visualizations
Arguments: --config (path to YAML configuration file)
Returns: Complete feature importance analysis with publication-ready outputs
Features: 36 total features + rating_score column
Configuration: YAML-based configuration for maximum flexibility and generalizability
Models: Per-perception optimal models (Random Forest, XGBoost, and RealMLP) with optimized parameters
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from retrain_and_save_models import ModelRetrainer
from feature_importance_analysis import FeatureImportanceAnalyzer

def setup_logging(config_file=None):
    """Setup logging for the pipeline with structured output."""
    # Create logs directory if using structured config
    if config_file:
        try:
            from config_loader import ConfigLoader
            config_loader = ConfigLoader(config_file)
            output_config = config_loader.get_output_config()
            base_dir = output_config.get('base_dir', 'outputs/feature_importance_analysis')
            logs_dir = output_config.get('logs_dir', 'logs')
            logs_path = Path(base_dir) / logs_dir
            logs_path.mkdir(parents=True, exist_ok=True)
            log_file = logs_path / 'feature_importance_pipeline.log'
        except Exception:
            log_file = 'feature_importance_pipeline.log'
    else:
        log_file = 'feature_importance_pipeline.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main(config_file: Optional[str] = None):
    """
    Run the complete feature importance pipeline.
    
    Args:
        config_file: Path to configuration file (default: experiment_config.yaml)
    """
    logger = setup_logging(config_file)
    
    logger.info("="*80)
    logger.info("FEATURE IMPORTANCE PIPELINE - CONFIGURATION-BASED")
    logger.info("="*80)
    
    try:
        # Step 1: Retrain and save models
        logger.info("\n" + "="*60)
        logger.info("STEP 1: RETRAINING AND SAVING MODELS")
        logger.info("="*60)
        
        retrainer = ModelRetrainer(config_file=config_file)
        saved_models = retrainer.retrain_all_models()
        
        if not saved_models:
            logger.error("No models were saved. Exiting.")
            return
        
        logger.info(f"✓ Successfully saved {len(saved_models)} models")
        
        # Step 2: Feature importance analysis
        logger.info("\n" + "="*60)
        logger.info("STEP 2: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        analyzer = FeatureImportanceAnalyzer(config_file=config_file)
        all_importances, summary_df = analyzer.analyze_all_perceptions()
        
        if not all_importances:
            logger.error("No feature importance analysis completed. Exiting.")
            return
        
        logger.info(f"✓ Successfully analyzed {len(all_importances)} perceptions")
        
        # Step 3: Generate summary report
        logger.info("\n" + "="*60)
        logger.info("STEP 3: GENERATING SUMMARY REPORT")
        logger.info("="*60)
        
        # Load config for report generation
        from config_loader import ConfigLoader
        config_loader = ConfigLoader(config_file)
        experiment_name = config_loader.config['experiment']['name']
        
        generate_summary_report(all_importances, summary_df, analyzer.output_dir, experiment_name)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Results saved in: {analyzer.output_dir}")
        logger.info(f"Models saved in: {retrainer.model_saver.models_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def generate_summary_report(all_importances, summary_df, output_dir, experiment_name='perception_feature_importance'):
    """Generate a summary report of the analysis."""
    logger = logging.getLogger(__name__)
    
    report_path = output_dir / f"feature_importance_report_{experiment_name}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Feature Importance Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"This report presents the feature importance analysis for the {experiment_name} experiment ")
        f.write(f"trained on perception prediction data with per-perception delta values.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("- **Models**: Per-perception optimal models (Random Forest, XGBoost, and RealMLP)\n")
        f.write("- **Features**: 36 total features + rating_score column\n")
        f.write("- **Delta Values**: Per-perception optimal configuration\n")
        f.write("  - Beautiful: Random Forest (δ=1.8)\n")
        f.write("  - Lively: Random Forest (δ=1.2)\n")
        f.write("  - Boring: RealMLP TD (δ=1.8)\n")
        f.write("  - Safe: XGBoost (δ=1.4)\n")
        f.write("- **Importance Method**: Permutation Importance\n")
        f.write("- **Repeats**: 30\n")
        f.write("- **Significance Level**: α = 0.05\n")
        f.write("- **Performance Metric**: F1-Score\n")
        f.write("- **Model Parameters**: Optimized per model type from configuration\n\n")
        
        f.write("## Results Summary\n\n")
        f.write(f"| Perception | Total Features | Significant Features | Top Feature |\n")
        f.write(f"|------------|----------------|---------------------|-------------|\n")
        
        for perception, top_features in all_importances.items():
            if len(top_features) > 0:
                total_features = len(summary_df[summary_df['perception'] == perception])
                significant_features = len(summary_df[
                    (summary_df['perception'] == perception) & 
                    (summary_df['significant'] == True)
                ])
                top_feature = top_features.index[0]
                f.write(f"| {perception.capitalize()} | {total_features} | {significant_features} | {top_feature} |\n")
        
        f.write("\n## Top 10 Features by Perception\n\n")
        
        for perception, top_features in all_importances.items():
            f.write(f"### {perception.capitalize()}\n\n")
            f.write("| Rank | Feature | Importance | Std | P-value | Significance |\n")
            f.write("|------|---------|------------|-----|---------|-------------|\n")
            
            for i, (feature, row) in enumerate(top_features.iterrows()):
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                f.write(f"| {i+1} | {feature} | {row['mean_importance']:.4f} | {row['std_importance']:.4f} | {row['p_value']:.4f} | {significance} |\n")
            
            f.write("\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `figure_9_feature_importance.png/pdf/svg`: Publication-ready visualization\n")
        f.write("- `feature_importance_detailed.csv`: Complete analysis results\n")
        f.write("- `feature_importance_summary.csv`: Summary of top features\n")
        f.write("- `feature_importance_report.md`: This report\n\n")
        
        f.write("## Notes\n\n")
        f.write("- Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05\n")
        f.write("- Importance scores represent the decrease in F1-score when feature is permuted\n")
        f.write("- Higher importance scores indicate more important features\n")
    
    logger.info(f"Summary report saved: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature importance analysis pipeline')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: experiment_config.yaml)')
    
    args = parser.parse_args()
    main(config_file=args.config)
