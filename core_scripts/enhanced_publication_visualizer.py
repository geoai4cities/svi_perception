#!/usr/bin/env python
"""
Enhanced Publication-Ready Visualizer with Mean Performance Analysis
Generates high-quality figures including overall mean performance across perceptions
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

# Import the base visualizer
from publication_visualizer import PublicationVisualizer


class EnhancedPublicationVisualizer(PublicationVisualizer):
    """
    Enhanced visualizer with mean performance analysis across all perceptions.
    """
    
    def __init__(self, output_dir: str, style: str = 'publication', dpi: int = 300):
        """Initialize enhanced visualizer."""
        super().__init__(output_dir, style, dpi)
        
        # Add mean analysis directory
        (self.output_dir / 'mean_analysis').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'overall_comparison').mkdir(parents=True, exist_ok=True)

        # Enforce consistent model order for uniform colors across figures
        self.model_order = ['random_forest', 'svm', 'xgboost', 'mlp']

    def _standardize_model_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose:
            Enforce a consistent categorical ordering for the 'model' column so that
            color assignments remain uniform across all plots.

        Arguments:
            - df (pd.DataFrame): Results dataframe with a 'model' column.

        Returns:
            - pd.DataFrame: Copy of df with ordered categorical 'model' and sorted rows.
        """
        if 'model' in df.columns:
            order = [m for m in self.model_order if m in df['model'].unique().tolist()]
            # Append any other models at the end deterministically
            others = [m for m in sorted(df['model'].unique().tolist()) if m not in order]
            cat_order = order + others
            df = df.copy()
            df['model'] = pd.Categorical(df['model'], categories=cat_order, ordered=True)
            df = df.sort_values(['model', 'delta']).reset_index(drop=True)
        return df

    def create_all_publication_figures(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Override to standardize model order/colors before generating base figures.

        Purpose:
            Ensure that all 18 figures use a consistent color mapping for models by
            imposing a fixed ordering on the results dataframe.

        Arguments:
            - results_df (pd.DataFrame): Experiment results.

        Returns:
            - Dict[str, Any]: Figure metadata from the base implementation.
        """
        std_df = self._standardize_model_order(results_df)
        return super().create_all_publication_figures(std_df)
        
    def create_mean_performance_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create mean performance visualizations across all perceptions.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            Dictionary of figure paths
        """
        figures = {}
        
        # 1. Mean F1 score across perceptions for each model at each delta
        fig = self._plot_mean_f1_by_delta(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'mean_analysis', 'mean_f1_by_delta')
            figures['mean_f1_by_delta'] = path
            plt.close(fig)
        
        # 2. Best overall model at each delta (mean across perceptions)
        fig = self._plot_best_overall_model_by_delta(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'mean_analysis', 'best_overall_model_by_delta')
            figures['best_overall_model_by_delta'] = path
            plt.close(fig)
        
        # 3. Mean performance heatmap (model x delta)
        fig = self._plot_mean_performance_heatmap(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'mean_analysis', 'mean_performance_heatmap')
            figures['mean_performance_heatmap'] = path
            plt.close(fig)
        
        # 4. Overall model ranking across all deltas and perceptions
        fig = self._plot_overall_model_ranking(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'overall_comparison', 'overall_model_ranking')
            figures['overall_model_ranking'] = path
            plt.close(fig)
        
        # 5. Mean ROC-AUC and PR-AUC comparison
        fig = self._plot_mean_auc_comparison(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'mean_analysis', 'mean_auc_comparison')
            figures['mean_auc_comparison'] = path
            plt.close(fig)
        
        # 6. Delta sensitivity for mean performance
        fig = self._plot_delta_sensitivity_mean(results_df)
        if fig:
            path = self._save_enhanced_figure(fig, 'mean_analysis', 'delta_sensitivity_mean')
            figures['delta_sensitivity_mean'] = path
            plt.close(fig)
        
        return figures
    
    def _plot_mean_f1_by_delta(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Plot mean F1 score across perceptions for each model at each delta."""
        try:
            # Calculate mean F1 across perceptions for each model-delta combination
            mean_df = df.groupby(['model', 'delta'])['f1_binary'].mean().reset_index()
            mean_df.columns = ['model', 'delta', 'mean_f1']
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            models = mean_df['model'].unique()
            for i, model in enumerate(models):
                model_data = mean_df[mean_df['model'] == model].sort_values('delta')
                # Ensure 1D numpy arrays to avoid multi-dimensional indexing issues
                x_vals = np.asarray(model_data['delta'].to_numpy()).reshape(-1)
                y_vals = np.asarray(model_data['mean_f1'].to_numpy()).reshape(-1)
                ax.plot(x_vals, y_vals,
                       marker=self.markers[i % len(self.markers)],
                       linestyle=self.linestyles[i % len(self.linestyles)],
                       color=self.colors[i % len(self.colors)],
                       label=model.upper().replace('_', ' '),
                       markersize=10, linewidth=2.5)
            
            # Find and mark the best overall configuration
            best_idx = mean_df['mean_f1'].idxmax()
            best_row = mean_df.loc[best_idx]
            ax.scatter(np.asarray([best_row['delta']]).reshape(-1)[0], 
                      np.asarray([best_row['mean_f1']]).reshape(-1)[0], 
                      s=300, color='red', marker='*', zorder=5,
                      label=f'Best: {best_row["model"].upper()} @ δ={best_row["delta"]:.1f}')
            
            ax.set_xlabel('Delta (δ)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean F1 Score (across all perceptions)', fontsize=14, fontweight='bold')
            ax.set_title('Overall Model Performance: Mean F1 Score vs Delta', 
                        fontsize=16, fontweight='bold')
            ax.legend(loc='best', frameon=True, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value annotations for best points per model
            for model in models:
                model_data = mean_df[mean_df['model'] == model]
                best_model_idx = model_data['mean_f1'].idxmax()
                best_model_row = model_data.loc[best_model_idx]
                ax.annotate(f'{best_model_row["mean_f1"]:.3f}',
                           xy=(float(best_model_row['delta']), float(best_model_row['mean_f1'])),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating mean F1 plot: {e}")
            return None
    
    def _plot_best_overall_model_by_delta(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Plot the best performing model at each delta value across all perceptions."""
        try:
            # Calculate mean metrics across perceptions
            mean_metrics = df.groupby(['model', 'delta'])[
                ['f1_binary', 'roc_auc_binary', 'pr_auc_binary', 'accuracy_binary']
            ].mean().reset_index()
            
            # Find best model at each delta
            best_models = []
            for delta in sorted(mean_metrics['delta'].unique()):
                delta_data = mean_metrics[mean_metrics['delta'] == delta]
                best_idx = delta_data['f1_binary'].idxmax()
                best_row = delta_data.loc[best_idx].to_dict()
                best_row['delta'] = delta
                best_models.append(best_row)
            
            best_df = pd.DataFrame(best_models)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            metrics = ['f1_binary', 'roc_auc_binary', 'pr_auc_binary', 'accuracy_binary']
            metric_names = ['F1 Score', 'ROC-AUC', 'PR-AUC', 'Accuracy']
            
            for idx, (ax, metric, name) in enumerate(zip(axes.flat, metrics, metric_names)):
                # Plot bars colored by model
                models = best_df['model'].unique()
                colors_map = {model: self.colors[i % len(self.colors)] 
                             for i, model in enumerate(models)}
                
                bars = ax.bar(best_df['delta'].astype(str), best_df[metric],
                             color=[colors_map[m] for m in best_df['model']])
                
                # Add model labels on bars
                for bar, model in zip(bars, best_df['model']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           model.upper()[:3], ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
                
                ax.set_xlabel('Delta (δ)', fontsize=12)
                ax.set_ylabel(name, fontsize=12)
                ax.set_title(f'Best Model {name} at Each Delta', fontsize=13, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            handles = [mpatches.Patch(color=colors_map[model], 
                                      label=model.upper().replace('_', ' '))
                      for model in models]
            fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, -0.05),
                      ncol=len(models), fontsize=11)
            
            plt.suptitle('Best Overall Models at Each Delta (Mean Across Perceptions)',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating best model plot: {e}")
            return None
    
    def _plot_mean_performance_heatmap(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Create heatmap of mean performance (model x delta)."""
        try:
            # Calculate mean F1 across perceptions
            pivot_data = df.pivot_table(
                values='f1_binary',
                index='model',
                columns='delta',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap with annotations
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Mean F1 Score'},
                       linewidths=0.5, linecolor='gray',
                       vmin=0, vmax=1, ax=ax)
            
            # Highlight best value in each column (delta)
            for col_idx, col in enumerate(pivot_data.columns):
                best_row_idx = pivot_data[col].idxmax()
                row_idx = list(pivot_data.index).index(best_row_idx)
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                                         fill=False, edgecolor='blue', lw=3))
            
            ax.set_xlabel('Delta (δ)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Model', fontsize=14, fontweight='bold')
            ax.set_title('Mean Performance Heatmap: Model × Delta\n(Mean F1 Score Across All Perceptions)',
                        fontsize=16, fontweight='bold')
            
            # Format y-axis labels
            ax.set_yticklabels([m.upper().replace('_', ' ') for m in pivot_data.index],
                              rotation=0)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance heatmap: {e}")
            return None
    
    def _plot_overall_model_ranking(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Create overall model ranking across all deltas and perceptions."""
        try:
            # Calculate comprehensive scores for each model
            model_scores = []
            
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                
                scores = {
                    'model': model,
                    'mean_f1': model_data['f1_binary'].mean(),
                    'std_f1': model_data['f1_binary'].std(),
                    'mean_roc_auc': model_data['roc_auc_binary'].mean(),
                    'mean_pr_auc': model_data['pr_auc_binary'].mean(),
                    'mean_accuracy': model_data['accuracy_binary'].mean(),
                    'best_f1': model_data['f1_binary'].max(),
                    'worst_f1': model_data['f1_binary'].min()
                }
                model_scores.append(scores)
            
            scores_df = pd.DataFrame(model_scores)
            scores_df = scores_df.sort_values('mean_f1', ascending=False)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Overall ranking bar chart
            ax = axes[0, 0]
            bars = ax.barh(range(len(scores_df)), scores_df['mean_f1'],
                          color=[self.colors[i % len(self.colors)] 
                                for i in range(len(scores_df))])
            ax.set_yticks(range(len(scores_df)))
            ax.set_yticklabels([m.upper().replace('_', ' ') for m in scores_df['model']])
            ax.set_xlabel('Mean F1 Score', fontsize=12)
            ax.set_title('Overall Model Ranking', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, scores_df['mean_f1'])):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10)
            
            # 2. F1 Score distribution (mean ± std)
            ax = axes[0, 1]
            x_pos = range(len(scores_df))
            ax.bar(x_pos, scores_df['mean_f1'], yerr=scores_df['std_f1'],
                  color=[self.colors[i % len(self.colors)] for i in range(len(scores_df))],
                  capsize=5, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([m.upper()[:3] for m in scores_df['model']], rotation=45)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title('F1 Score: Mean ± Std Dev', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # 3. Multiple metrics comparison
            ax = axes[1, 0]
            metrics = ['mean_f1', 'mean_roc_auc', 'mean_pr_auc', 'mean_accuracy']
            metric_labels = ['F1', 'ROC-AUC', 'PR-AUC', 'Accuracy']
            x = np.arange(len(scores_df))
            width = 0.2
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax.bar(x + i*width, scores_df[metric], width, 
                      label=label, alpha=0.8)
            
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([m.upper()[:3] for m in scores_df['model']], rotation=45)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Multiple Metrics Comparison', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. Best vs Worst F1 range
            ax = axes[1, 1]
            ax.scatter(scores_df['worst_f1'], scores_df['best_f1'],
                      s=200, alpha=0.6,
                      c=[self.colors[i % len(self.colors)] for i in range(len(scores_df))])
            
            # Add model labels
            for i, row in scores_df.iterrows():
                ax.annotate(row['model'].upper()[:3],
                           (row['worst_f1'], row['best_f1']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
            
            # Add diagonal line
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', alpha=0.3)
            
            ax.set_xlabel('Worst F1 Score', fontsize=12)
            ax.set_ylabel('Best F1 Score', fontsize=12)
            ax.set_title('F1 Score Range: Best vs Worst', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            plt.suptitle('Overall Model Performance Analysis\n(Across All Deltas and Perceptions)',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model ranking plot: {e}")
            return None
    
    def _plot_mean_auc_comparison(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Plot mean ROC-AUC and PR-AUC comparison across models and deltas."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # ROC-AUC
            ax = axes[0]
            mean_roc = df.groupby(['model', 'delta'])['roc_auc_binary'].mean().reset_index()
            
            for i, model in enumerate(mean_roc['model'].unique()):
                model_data = mean_roc[mean_roc['model'] == model].sort_values('delta')
                x_vals = np.asarray(model_data['delta'].to_numpy()).reshape(-1)
                y_vals = np.asarray(model_data['roc_auc_binary'].to_numpy()).reshape(-1)
                ax.plot(x_vals, y_vals,
                       marker=self.markers[i % len(self.markers)],
                       linestyle=self.linestyles[i % len(self.linestyles)],
                       color=self.colors[i % len(self.colors)],
                       label=model.upper().replace('_', ' '),
                       markersize=8, linewidth=2)
            
            ax.set_xlabel('Delta (δ)', fontsize=12)
            ax.set_ylabel('Mean ROC-AUC', fontsize=12)
            ax.set_title('Mean ROC-AUC vs Delta', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # PR-AUC
            ax = axes[1]
            mean_pr = df.groupby(['model', 'delta'])['pr_auc_binary'].mean().reset_index()
            
            for i, model in enumerate(mean_pr['model'].unique()):
                model_data = mean_pr[mean_pr['model'] == model].sort_values('delta')
                x_vals = np.asarray(model_data['delta'].to_numpy()).reshape(-1)
                y_vals = np.asarray(model_data['pr_auc_binary'].to_numpy()).reshape(-1)
                ax.plot(x_vals, y_vals,
                       marker=self.markers[i % len(self.markers)],
                       linestyle=self.linestyles[i % len(self.linestyles)],
                       color=self.colors[i % len(self.colors)],
                       label=model.upper().replace('_', ' '),
                       markersize=8, linewidth=2)
            
            ax.set_xlabel('Delta (δ)', fontsize=12)
            ax.set_ylabel('Mean PR-AUC', fontsize=12)
            ax.set_title('Mean PR-AUC vs Delta', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            plt.suptitle('Mean AUC Comparison Across All Perceptions',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating AUC comparison plot: {e}")
            return None
    
    def _plot_delta_sensitivity_mean(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Plot delta sensitivity analysis for mean performance."""
        try:
            # Calculate coefficient of variation for each model across deltas
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. Stability analysis (std dev of F1 across deltas)
            ax = axes[0]
            stability_data = []
            
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                for perception in df['perception'].unique():
                    perc_data = model_data[model_data['perception'] == perception]
                    if not perc_data.empty:
                        stability_data.append({
                            'model': model,
                            'perception': perception,
                            'std_f1': perc_data['f1_binary'].std(),
                            'cv_f1': perc_data['f1_binary'].std() / perc_data['f1_binary'].mean()
                            if perc_data['f1_binary'].mean() > 0 else 0
                        })
            
            stability_df = pd.DataFrame(stability_data)
            
            # Plot coefficient of variation
            model_cv = stability_df.groupby('model')['cv_f1'].mean().sort_values()
            bars = ax.barh(range(len(model_cv)), model_cv.values,
                          color=[self.colors[i % len(self.colors)] 
                                for i in range(len(model_cv))])
            ax.set_yticks(range(len(model_cv)))
            ax.set_yticklabels([m.upper().replace('_', ' ') for m in model_cv.index])
            ax.set_xlabel('Coefficient of Variation (lower is more stable)', fontsize=12)
            ax.set_title('Model Stability Across Delta Values', fontsize=13, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, model_cv.values):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10)
            
            # 2. Optimal delta identification
            ax = axes[1]
            delta_performance = df.groupby('delta')['f1_binary'].agg(['mean', 'std']).reset_index()
            
            ax.bar(delta_performance['delta'].astype(str), delta_performance['mean'],
                  yerr=delta_performance['std'], capsize=5,
                  color=self.colors[0], alpha=0.7)
            
            # Highlight best delta
            best_idx = delta_performance['mean'].idxmax()
            best_delta = delta_performance.loc[best_idx, 'delta']
            best_mean = delta_performance.loc[best_idx, 'mean']
            ax.scatter(str(best_delta), best_mean, s=200, color='red', 
                      marker='*', zorder=5, label=f'Best: δ={best_delta}')
            
            ax.set_xlabel('Delta (δ)', fontsize=12)
            ax.set_ylabel('Mean F1 Score', fontsize=12)
            ax.set_title('Optimal Delta Selection (Mean ± Std)', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Delta Sensitivity Analysis for Mean Performance',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating delta sensitivity plot: {e}")
            return None
    
    def _save_enhanced_figure(self, fig: plt.Figure, subdir: str, name: str) -> Dict[str, str]:
        """Save figure in multiple formats."""
        paths = {}
        base_path = self.output_dir / subdir / f'fig_{name}'
        
        for fmt in ['png', 'pdf', 'svg']:
            path = f'{base_path}.{fmt}'
            fig.savefig(path, format=fmt, dpi=self.dpi, bbox_inches='tight')
            paths[fmt] = path
            
        return paths
    
    def create_enhanced_publication_figures(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate all publication figures including mean analysis.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            Dictionary of all generated figure paths
        """
        all_figures = {}
        
        # Generate original publication figures
        original_figures = self.create_all_publication_figures(results_df)
        all_figures.update(original_figures)
        
        # Generate mean performance analysis figures
        mean_figures = self.create_mean_performance_analysis(results_df)
        all_figures.update(mean_figures)
        
        self.logger.info(f"Generated {len(all_figures)} enhanced publication figures")
        
        return all_figures