#!/usr/bin/env python
"""
Publication-Ready Visualizer for Delta Sensitivity Analysis
Generates high-quality figures for academic publications
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')


class PublicationVisualizer:
    """
    Creates publication-ready visualizations for delta sensitivity analysis.
    Generates black & white friendly, high-DPI figures in multiple formats.
    """
    
    def __init__(self, output_dir: str, style: str = 'publication', dpi: int = 300):
        """
        Initialize the publication visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: 'publication' (B&W friendly) or 'colored'
            dpi: Plot resolution for publication quality
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # Setup publication-style formatting
        self._setup_publication_style()
        
        # Create output directories
        self._create_output_directories()
        
    def _setup_publication_style(self):
        """Configure matplotlib for publication-quality plots."""
        # Set publication-friendly defaults
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
            'axes.linewidth': 1.2,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        # Use colorful visualizations for better clarity
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        self.linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        self.hatches = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']
            
    def _create_output_directories(self):
        """Create organized output directory structure."""
        subdirs = [
            'performance_analysis',
            'model_comparison', 
            'class_balance',
            'efficiency',
            'correlation',
            'statistical_analysis',
            'dual_approach'
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
    def create_all_publication_figures(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate all publication-ready figures.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            Dictionary of figure metadata
        """
        self.logger.info("Creating publication-ready visualizations...")
        
        figures_created = {}
        
        # 1. Performance vs Delta Analysis
        figures_created.update(self._create_performance_analysis(results_df))
        
        # 2. Model Comparison Analysis
        figures_created.update(self._create_model_comparison(results_df))
        
        # 3. Class Balance Analysis
        figures_created.update(self._create_class_balance_analysis(results_df))
        
        # 4. Efficiency Analysis
        figures_created.update(self._create_efficiency_analysis(results_df))
        
        # 5. Dual Approach Comparison
        figures_created.update(self._create_dual_approach_analysis(results_df))
        
        # 6. Statistical Analysis
        figures_created.update(self._create_statistical_analysis(results_df))
        
        # 7. Train/Val/Test Performance Comparison (NEW)
        figures_created.update(self._create_train_val_test_comparison(results_df))
        
        self.logger.info(f"Created {len(figures_created)} publication figures")
        return figures_created
        
    def _create_performance_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create performance vs delta analysis figures."""
        figures = {}
        
        # Binary classification metrics
        binary_metrics = {
            'f1_binary': 'F1 Score',
            'accuracy_binary': 'Accuracy', 
            'roc_auc_binary': 'ROC-AUC',
            'pr_auc_binary': 'PR-AUC'
        }
        
        # Multi-class metrics
        multiclass_metrics = {
            'accuracy_multiclass': 'Multi-class Accuracy'
        }
        
        # Create binary performance plots
        for metric_col, metric_name in binary_metrics.items():
            if metric_col in results_df.columns:
                fig = self._plot_performance_vs_delta(
                    results_df, metric_col, metric_name, 'Binary Classification'
                )
                filename = f'performance_analysis/fig_{metric_col}_vs_delta'
                self._save_figure(fig, filename)
                figures[f'{metric_col}_vs_delta'] = filename
                plt.close(fig)
                
        # Create multi-class performance plots
        for metric_col, metric_name in multiclass_metrics.items():
            if metric_col in results_df.columns:
                fig = self._plot_performance_vs_delta(
                    results_df, metric_col, metric_name, 'Multi-class Classification'
                )
                filename = f'performance_analysis/fig_{metric_col}_vs_delta'
                self._save_figure(fig, filename)
                figures[f'{metric_col}_vs_delta'] = filename
                plt.close(fig)
                
        return figures
        
    def _plot_performance_vs_delta(self, results_df: pd.DataFrame, metric_col: str, 
                                  metric_name: str, title_suffix: str) -> plt.Figure:
        """Plot performance metric vs delta values."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        perceptions = results_df['perception'].unique()
        
        for i, perception in enumerate(perceptions):
            if i >= 4:  # Only plot first 4 perceptions
                break
                
            ax = axes[i]
            perception_data = results_df[results_df['perception'] == perception]
            
            models = perception_data['model'].unique()
            
            for j, model in enumerate(models):
                model_data = perception_data[perception_data['model'] == model].sort_values('delta')
                
                if not model_data.empty and metric_col in model_data.columns:
                    # Handle NaN values
                    valid_data = model_data.dropna(subset=[metric_col])
                    
                    if not valid_data.empty:
                        ax.plot(valid_data['delta'].values, valid_data[metric_col].values,
                               marker=self.markers[j % len(self.markers)],
                               linestyle=self.linestyles[j % len(self.linestyles)],
                               color=self.colors[j % len(self.colors)],
                               label=model.replace('_', ' ').title(),
                               markersize=6, linewidth=2)
            
            ax.set_title(f'{perception.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Delta (δ)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Set reasonable y-axis limits
            if metric_col in ['f1_binary', 'accuracy_binary', 'roc_auc_binary', 'pr_auc_binary']:
                ax.set_ylim(0, 1.05)
            elif 'accuracy_multiclass' in metric_col:
                ax.set_ylim(0, 1.05)
                
        # Hide unused subplots
        for i in range(len(perceptions), 4):
            axes[i].set_visible(False)
            
        plt.suptitle(f'{metric_name} vs Delta Values - {title_suffix}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _create_model_comparison(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create model comparison visualizations."""
        figures = {}
        
        # Model performance heatmap
        fig = self._plot_model_performance_heatmap(results_df)
        filename = 'model_comparison/fig_model_performance_heatmap'
        self._save_figure(fig, filename)
        figures['model_heatmap'] = filename
        plt.close(fig)
        
        # Best model per perception
        fig = self._plot_best_models_per_perception(results_df)
        filename = 'model_comparison/fig_best_models_per_perception'
        self._save_figure(fig, filename)
        figures['best_models'] = filename
        plt.close(fig)
        
        return figures
        
    def _plot_model_performance_heatmap(self, results_df: pd.DataFrame) -> plt.Figure:
        """Create model performance heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Binary classification heatmap
        if 'f1_binary' in results_df.columns:
            binary_pivot = results_df.groupby(['perception', 'model'])['f1_binary'].mean().unstack()
            
            im1 = ax1.imshow(binary_pivot.values, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(binary_pivot.columns)))
            ax1.set_xticklabels([col.replace('_', ' ').title() for col in binary_pivot.columns], 
                               rotation=45, ha='right')
            ax1.set_yticks(range(len(binary_pivot.index)))
            ax1.set_yticklabels([idx.capitalize() for idx in binary_pivot.index])
            ax1.set_title('Binary Classification F1-Score', fontweight='bold')
            
            # Add text annotations
            for i in range(len(binary_pivot.index)):
                for j in range(len(binary_pivot.columns)):
                    value = binary_pivot.iloc[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < 0.7 else 'black'
                        ax1.text(j, i, f'{value:.3f}', ha='center', va='center', 
                                color=text_color, fontweight='bold')
                        
            # Colorbar for binary
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('F1-Score', rotation=270, labelpad=15)
        
        # Multi-class accuracy heatmap
        if 'accuracy_multiclass' in results_df.columns:
            multi_pivot = results_df.groupby(['perception', 'model'])['accuracy_multiclass'].mean().unstack()
            
            im2 = ax2.imshow(multi_pivot.values, cmap='plasma', aspect='auto', vmin=0, vmax=1)
            ax2.set_xticks(range(len(multi_pivot.columns)))
            ax2.set_xticklabels([col.replace('_', ' ').title() for col in multi_pivot.columns], 
                               rotation=45, ha='right')
            ax2.set_yticks(range(len(multi_pivot.index)))
            ax2.set_yticklabels([idx.capitalize() for idx in multi_pivot.index])
            ax2.set_title('Multi-class Accuracy', fontweight='bold')
            
            # Add text annotations
            for i in range(len(multi_pivot.index)):
                for j in range(len(multi_pivot.columns)):
                    value = multi_pivot.iloc[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < 0.7 else 'black'
                        ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                                color=text_color, fontweight='bold')
                        
            # Colorbar for multi-class
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Accuracy', rotation=270, labelpad=15)
        
        plt.suptitle('Model Performance Comparison Across Perceptions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _plot_best_models_per_perception(self, results_df: pd.DataFrame) -> plt.Figure:
        """Plot best performing models for each perception."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Find best model per perception for binary F1
        if 'f1_binary' in results_df.columns:
            best_models = results_df.groupby('perception').apply(
                lambda x: x.loc[x['f1_binary'].idxmax()]
            ).reset_index(drop=True)
            
            perceptions = best_models['perception'].tolist()
            f1_scores = best_models['f1_binary'].tolist()
            models = best_models['model'].tolist()
            deltas = best_models['delta'].tolist()
            
            # Create bar plot
            bars = ax.bar(range(len(perceptions)), f1_scores, 
                         color=self.colors[:len(perceptions)],
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels and model information
            for i, (bar, model, delta, f1) in enumerate(zip(bars, models, deltas, f1_scores)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{f1:.3f}\n{model.replace("_", " ").title()}\nδ={delta}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
                       
            ax.set_xlabel('Perception Type', fontsize=14)
            ax.set_ylabel('Best F1-Score (Binary)', fontsize=14)
            ax.set_title('Best Performing Model per Perception Type', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(range(len(perceptions)))
            ax.set_xticklabels([p.capitalize() for p in perceptions])
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        return fig
        
    def _create_class_balance_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create class balance analysis."""
        figures = {}
        
        # Sample count analysis
        if all(col in results_df.columns for col in ['n_pos_binary', 'n_neg_binary', 'n_mid_binary']):
            fig = self._plot_class_balance_vs_delta(results_df)
            filename = 'class_balance/fig_class_balance_vs_delta'
            self._save_figure(fig, filename)
            figures['class_balance'] = filename
            plt.close(fig)
            
        return figures
        
    def _plot_class_balance_vs_delta(self, results_df: pd.DataFrame) -> plt.Figure:
        """Plot class balance changes with delta."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        perceptions = results_df['perception'].unique()
        
        for i, perception in enumerate(perceptions):
            if i >= 4:
                break
                
            ax = axes[i]
            perception_data = results_df[results_df['perception'] == perception]
            
            # Group by delta and get mean values
            delta_stats = perception_data.groupby('delta').agg({
                'n_pos_binary': 'mean',
                'n_neg_binary': 'mean', 
                'n_mid_binary': 'mean'
            }).reset_index()
            
            # Stacked bar chart
            width = 0.05
            ax.bar(delta_stats['delta'], delta_stats['n_neg_binary'], 
                  width, label='Negative', color=self.colors[0])
            ax.bar(delta_stats['delta'], delta_stats['n_pos_binary'], 
                  width, bottom=delta_stats['n_neg_binary'], 
                  label='Positive', color=self.colors[1])
            ax.bar(delta_stats['delta'], delta_stats['n_mid_binary'], 
                  width, bottom=delta_stats['n_neg_binary'] + delta_stats['n_pos_binary'],
                  label='Mid-range (Discarded)', color=self.colors[2], alpha=0.7)
                  
            ax.set_title(f'{perception.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Delta (δ)', fontsize=12)
            ax.set_ylabel('Sample Count', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(len(perceptions), 4):
            axes[i].set_visible(False)
            
        plt.suptitle('Class Distribution vs Delta Values', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _create_efficiency_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create efficiency analysis."""
        figures = {}
        
        if 'train_time' in results_df.columns:
            fig = self._plot_training_efficiency(results_df)
            filename = 'efficiency/fig_training_efficiency'
            self._save_figure(fig, filename)
            figures['training_efficiency'] = filename
            plt.close(fig)
            
        return figures
        
    def _plot_training_efficiency(self, results_df: pd.DataFrame) -> plt.Figure:
        """Plot training time efficiency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training time by model
        model_times = results_df.groupby('model')['train_time'].mean().sort_values()
        
        bars = ax1.bar(range(len(model_times)), model_times.values,
                      color=self.colors[:len(model_times)],
                      edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Average Training Time (seconds)', fontsize=12)
        ax1.set_title('Training Time by Model', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(model_times)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in model_times.index], 
                           rotation=45, ha='right')
        
        # Add value labels
        for bar, time_val in zip(bars, model_times.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Performance vs training time scatter
        if 'f1_binary' in results_df.columns:
            models = results_df['model'].unique()
            for i, model in enumerate(models):
                model_data = results_df[results_df['model'] == model]
                ax2.scatter(model_data['train_time'], model_data['f1_binary'],
                           marker=self.markers[i % len(self.markers)],
                           color=self.colors[i % len(self.colors)],
                           label=model.replace('_', ' ').title(),
                           s=60, alpha=0.7, edgecolors='black')
            
            ax2.set_xlabel('Training Time (seconds)', fontsize=12)
            ax2.set_ylabel('F1-Score (Binary)', fontsize=12)
            ax2.set_title('Performance vs Training Time', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def _create_dual_approach_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create dual approach (binary vs multiclass) analysis."""
        figures = {}
        
        if all(col in results_df.columns for col in ['f1_binary', 'accuracy_multiclass']):
            fig = self._plot_binary_vs_multiclass_comparison(results_df)
            filename = 'dual_approach/fig_binary_vs_multiclass_comparison'
            self._save_figure(fig, filename)
            figures['dual_comparison'] = filename
            plt.close(fig)
            
        return figures
        
    def _plot_binary_vs_multiclass_comparison(self, results_df: pd.DataFrame) -> plt.Figure:
        """Compare binary and multiclass performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        perceptions = results_df['perception'].unique()
        
        for i, perception in enumerate(perceptions):
            if i >= 4:
                break
                
            ax = axes[i]
            perception_data = results_df[results_df['perception'] == perception]
            
            # Scatter plot: binary F1 vs multiclass accuracy
            models = perception_data['model'].unique()
            for j, model in enumerate(models):
                model_data = perception_data[perception_data['model'] == model]
                
                # Remove NaN values
                valid_data = model_data.dropna(subset=['f1_binary', 'accuracy_multiclass'])
                
                if not valid_data.empty:
                    ax.scatter(valid_data['f1_binary'], valid_data['accuracy_multiclass'],
                              marker=self.markers[j % len(self.markers)],
                              color=self.colors[j % len(self.colors)],
                              label=model.replace('_', ' ').title(),
                              s=80, alpha=0.7, edgecolors='black')
            
            # Add diagonal line (perfect correlation)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Binary F1-Score', fontsize=12)
            ax.set_ylabel('Multi-class Accuracy', fontsize=12)
            ax.set_title(f'{perception.capitalize()}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
        # Hide unused subplots
        for i in range(len(perceptions), 4):
            axes[i].set_visible(False)
            
        plt.suptitle('Binary vs Multi-class Performance Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _create_statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create statistical analysis figures."""
        figures = {}
        
        # Summary statistics table
        fig = self._plot_summary_statistics(results_df)
        filename = 'statistical_analysis/fig_summary_statistics'
        self._save_figure(fig, filename)
        figures['summary_stats'] = filename
        plt.close(fig)
        
        return figures
        
    def _plot_summary_statistics(self, results_df: pd.DataFrame) -> plt.Figure:
        """Create summary statistics visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate summary statistics
        numeric_cols = ['f1_binary', 'accuracy_binary', 'roc_auc_binary', 'accuracy_multiclass']
        available_cols = [col for col in numeric_cols if col in results_df.columns]
        
        if available_cols:
            summary_stats = results_df.groupby(['perception', 'model'])[available_cols].agg([
                'mean', 'std', 'min', 'max'
            ]).round(3)
            
            # Create a simplified table visualization
            summary_text = "Summary Statistics\n" + "="*50 + "\n\n"
            
            for perception in results_df['perception'].unique():
                summary_text += f"{perception.upper()}:\n"
                perception_data = results_df[results_df['perception'] == perception]
                
                for col in available_cols:
                    if col in perception_data.columns:
                        mean_val = perception_data[col].mean()
                        std_val = perception_data[col].std()
                        summary_text += f"  {col}: {mean_val:.3f} ± {std_val:.3f}\n"
                        
                summary_text += "\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Experiment Summary Statistics', fontsize=16, fontweight='bold')
        
        return fig
        
    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure in multiple formats."""
        base_path = self.output_dir / filename
        
        # Ensure directory exists
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            
            if fmt == 'png':
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            elif fmt == 'pdf':
                fig.savefig(save_path, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            elif fmt == 'svg':
                fig.savefig(save_path, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                           
        self.logger.info(f"Saved figure: {filename}")
    
    def _create_train_val_test_comparison(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create train/val/test performance comparison figure.
        Shows overfitting/underfitting patterns across models.
        """
        output_dir = self.output_dir / 'train_val_test'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if train/val metrics exist
        if 'f1_train' not in results_df.columns or 'f1_val' not in results_df.columns:
            self.logger.warning("Train/val metrics not found in results. Skipping train/val/test comparison.")
            return {}
        
        # Create figure with subplots for each perception
        perceptions = results_df['perception'].unique()
        n_perceptions = len(perceptions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, perception in enumerate(perceptions):
            ax = axes[idx]
            perception_data = results_df[results_df['perception'] == perception]
            
            # Prepare data for plotting
            models = perception_data['model'].unique()
            deltas = sorted(perception_data['delta'].unique())
            
            # Calculate mean F1 scores across deltas for each split
            train_scores = []
            val_scores = []
            test_scores = []
            
            for model in models:
                model_data = perception_data[perception_data['model'] == model]
                train_scores.append(model_data['f1_train'].mean())
                val_scores.append(model_data['f1_val'].mean())
                test_scores.append(model_data['f1_binary'].mean())
            
            # Create grouped bar plot
            x = np.arange(len(models))
            width = 0.25
            
            bars1 = ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
            bars2 = ax.bar(x, val_scores, width, label='Validation', alpha=0.8)
            bars3 = ax.bar(x + width, test_scores, width, label='Test', alpha=0.8)
            
            # Customize subplot
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.set_title(f'{perception.capitalize()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Train vs Validation vs Test Performance Comparison', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save in multiple formats
        filename = 'fig_train_val_test_comparison'
        self._save_figure(fig, filename)
        plt.close()
        
        return {'train_val_test_comparison': f'train_val_test/{filename}'}


def main():
    """Test the visualizer with sample data."""
    # Create sample test data
    np.random.seed(42)
    
    sample_data = []
    perceptions = ['beautiful', 'lively', 'boring', 'safe']
    models = ['random_forest', 'svm', 'xgboost', 'mlp']
    deltas = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    
    for perception in perceptions:
        for model in models:
            for delta in deltas:
                sample_data.append({
                    'perception': perception,
                    'model': model,
                    'delta': delta,
                    'f1_binary': np.random.uniform(0.6, 0.9),
                    'accuracy_binary': np.random.uniform(0.6, 0.9),
                    'roc_auc_binary': np.random.uniform(0.7, 0.95),
                    'pr_auc_binary': np.random.uniform(0.7, 0.95),
                    'accuracy_multiclass': np.random.uniform(0.3, 0.7),
                    'n_pos_binary': np.random.randint(40, 80),
                    'n_neg_binary': np.random.randint(40, 80),
                    'n_mid_binary': np.random.randint(100, 200),
                    'train_time': np.random.uniform(0.5, 3.0)
                })
    
    results_df = pd.DataFrame(sample_data)
    
    # Test visualizer
    visualizer = PublicationVisualizer('/tmp/test_publication_viz')
    figures = visualizer.create_all_publication_figures(results_df)
    
    print(f"Created {len(figures)} publication figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()