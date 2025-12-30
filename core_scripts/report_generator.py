#!/usr/bin/env python
"""
Comprehensive Report Generator for Delta Sensitivity Analysis
Generates detailed HTML and Markdown reports with statistical analysis
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ReportGenerator:
    """
    Generates comprehensive analysis reports for delta sensitivity experiments.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize report generator.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create reports directory
        self.reports_dir = self.experiment_dir / "04_analysis" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_report(self, results_df: pd.DataFrame, 
                                     config: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate comprehensive analysis report in multiple formats.
        
        Args:
            results_df: DataFrame with experiment results
            config: Experiment configuration dictionary
            
        Returns:
            Dictionary with paths to generated reports
        """
        report_paths = {}
        
        # Generate markdown report
        md_path = self._generate_markdown_report(results_df, config)
        report_paths['markdown'] = str(md_path)
        
        # Generate HTML report
        html_path = self._generate_html_report(results_df, config)
        report_paths['html'] = str(html_path)
        
        # Generate JSON summary
        json_path = self._generate_json_summary(results_df, config)
        report_paths['json'] = str(json_path)
        
        # Generate CSV detailed analysis
        csv_path = self._generate_csv_analysis(results_df)
        report_paths['csv'] = str(csv_path)
        
        self.logger.info(f"Generated {len(report_paths)} report formats")
        
        return report_paths
    
    def _generate_markdown_report(self, df: pd.DataFrame, config: Dict = None) -> Path:
        """Generate detailed markdown report."""
        report_path = self.reports_dir / "comprehensive_analysis_report.md"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# Delta Sensitivity Analysis - Comprehensive Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = self._generate_executive_summary(df)
            f.write(summary + "\n\n")
            
            # Overall Performance Analysis
            f.write("## 1. Overall Performance Analysis\n\n")
            f.write("### 1.1 Best Models Across All Configurations\n\n")
            best_overall = self._analyze_best_overall(df)
            f.write(best_overall + "\n\n")
            
            f.write("### 1.2 Mean Performance Across Perceptions\n\n")
            mean_perf = self._analyze_mean_performance(df)
            f.write(mean_perf + "\n\n")
            
            # Per-Perception Analysis
            f.write("## 2. Per-Perception Analysis\n\n")
            for perception in sorted(df['perception'].unique()):
                f.write(f"### 2.{list(df['perception'].unique()).index(perception)+1} {perception.title()}\n\n")
                perc_analysis = self._analyze_perception(df, perception)
                f.write(perc_analysis + "\n\n")
            
            # Delta Sensitivity Analysis
            f.write("## 3. Delta Sensitivity Analysis\n\n")
            f.write("### 3.1 Optimal Delta Values\n\n")
            delta_analysis = self._analyze_delta_sensitivity(df)
            f.write(delta_analysis + "\n\n")
            
            f.write("### 3.2 Model Stability Across Deltas\n\n")
            stability = self._analyze_model_stability(df)
            f.write(stability + "\n\n")
            
            # Model Comparison
            f.write("## 4. Model Comparison\n\n")
            f.write("### 4.1 Head-to-Head Comparison\n\n")
            comparison = self._compare_models(df)
            f.write(comparison + "\n\n")
            
            f.write("### 4.2 Statistical Significance Testing\n\n")
            significance = self._test_statistical_significance(df)
            f.write(significance + "\n\n")
            
            # Computational Efficiency
            f.write("## 5. Computational Efficiency\n\n")
            efficiency = self._analyze_efficiency(df)
            f.write(efficiency + "\n\n")
            
            # Recommendations
            f.write("## 6. Recommendations\n\n")
            recommendations = self._generate_recommendations(df)
            f.write(recommendations + "\n\n")
            
            # Appendix
            f.write("## Appendix\n\n")
            f.write("### A. Experiment Configuration\n\n")
            if config:
                f.write("```json\n")
                f.write(json.dumps(config, indent=2))
                f.write("\n```\n\n")
            
            f.write("### B. Detailed Metrics Table\n\n")
            metrics_table = self._generate_metrics_table(df)
            f.write(metrics_table + "\n")
        
        self.logger.info(f"Markdown report saved to: {report_path}")
        return report_path
    
    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary section."""
        summary = []
        
        # Overall best configuration
        best_idx = df['f1_binary'].idxmax()
        best_row = df.loc[best_idx]
        
        summary.append(f"**Best Overall Configuration:**")
        summary.append(f"- Model: **{best_row['model'].upper()}**")
        summary.append(f"- Perception: **{best_row['perception'].title()}**")
        summary.append(f"- Delta: **{best_row['delta']}**")
        summary.append(f"- F1 Score: **{best_row['f1_binary']:.4f}**")
        summary.append("")
        
        # Mean performance summary
        mean_f1 = df.groupby('model')['f1_binary'].mean()
        best_mean_model = mean_f1.idxmax()
        
        summary.append(f"**Best Average Model:** {best_mean_model.upper()} (Mean F1: {mean_f1[best_mean_model]:.4f})")
        summary.append("")
        
        # Key findings
        summary.append("**Key Findings:**")
        
        # Find optimal delta across all models
        delta_means = df.groupby('delta')['f1_binary'].mean()
        best_delta = delta_means.idxmax()
        summary.append(f"1. Optimal delta value across all models: **δ = {best_delta}**")
        
        # Model ranking
        model_ranking = mean_f1.sort_values(ascending=False)
        summary.append(f"2. Model ranking by mean F1: {', '.join([m.upper() for m in model_ranking.index])}")
        
        # Perception difficulty
        perc_means = df.groupby('perception')['f1_binary'].mean().sort_values(ascending=False)
        summary.append(f"3. Perception difficulty (easiest to hardest): {', '.join([p.title() for p in perc_means.index])}")
        
        # Performance range
        f1_range = df['f1_binary'].max() - df['f1_binary'].min()
        summary.append(f"4. F1 score range: {df['f1_binary'].min():.4f} - {df['f1_binary'].max():.4f} (Δ = {f1_range:.4f})")
        
        return "\n".join(summary)
    
    def _analyze_best_overall(self, df: pd.DataFrame) -> str:
        """Analyze best overall configurations."""
        analysis = []
        
        # Top 10 configurations
        top_10 = df.nlargest(10, 'f1_binary')[['perception', 'model', 'delta', 'f1_binary', 
                                               'roc_auc_binary', 'pr_auc_binary']]
        
        analysis.append("| Rank | Perception | Model | Delta | F1 Score | ROC-AUC | PR-AUC |")
        analysis.append("|------|------------|-------|-------|----------|---------|---------|")
        
        for i, row in enumerate(top_10.itertuples(), 1):
            analysis.append(f"| {i} | {row.perception.title()} | {row.model.upper()} | "
                          f"{row.delta} | {row.f1_binary:.4f} | {row.roc_auc_binary:.4f} | "
                          f"{row.pr_auc_binary:.4f} |")
        
        return "\n".join(analysis)
    
    def _analyze_mean_performance(self, df: pd.DataFrame) -> str:
        """Analyze mean performance across perceptions."""
        analysis = []
        
        # Calculate mean metrics for each model-delta combination
        mean_metrics = df.groupby(['model', 'delta'])[
            ['f1_binary', 'roc_auc_binary', 'pr_auc_binary', 'accuracy_binary']
        ].mean().round(4)
        
        # Find best configuration for mean performance
        best_idx = mean_metrics['f1_binary'].idxmax()
        best_config = mean_metrics.loc[best_idx]
        
        analysis.append(f"**Best Mean Configuration:** {best_idx[0].upper()} at δ={best_idx[1]}")
        analysis.append(f"- Mean F1: {best_config['f1_binary']:.4f}")
        analysis.append(f"- Mean ROC-AUC: {best_config['roc_auc_binary']:.4f}")
        analysis.append(f"- Mean PR-AUC: {best_config['pr_auc_binary']:.4f}")
        analysis.append(f"- Mean Accuracy: {best_config['accuracy_binary']:.4f}")
        analysis.append("")
        
        # Model comparison table
        model_means = df.groupby('model')[
            ['f1_binary', 'roc_auc_binary', 'pr_auc_binary', 'accuracy_binary']
        ].agg(['mean', 'std']).round(4)
        
        analysis.append("**Model Performance Summary (Mean ± Std):**")
        analysis.append("")
        analysis.append("| Model | F1 Score | ROC-AUC | PR-AUC | Accuracy |")
        analysis.append("|-------|----------|---------|---------|----------|")
        
        for model in model_means.index:
            f1_mean = model_means.loc[model, ('f1_binary', 'mean')]
            f1_std = model_means.loc[model, ('f1_binary', 'std')]
            roc_mean = model_means.loc[model, ('roc_auc_binary', 'mean')]
            roc_std = model_means.loc[model, ('roc_auc_binary', 'std')]
            pr_mean = model_means.loc[model, ('pr_auc_binary', 'mean')]
            pr_std = model_means.loc[model, ('pr_auc_binary', 'std')]
            acc_mean = model_means.loc[model, ('accuracy_binary', 'mean')]
            acc_std = model_means.loc[model, ('accuracy_binary', 'std')]
            
            analysis.append(f"| {model.upper()} | {f1_mean:.3f}±{f1_std:.3f} | "
                          f"{roc_mean:.3f}±{roc_std:.3f} | {pr_mean:.3f}±{pr_std:.3f} | "
                          f"{acc_mean:.3f}±{acc_std:.3f} |")
        
        return "\n".join(analysis)
    
    def _analyze_perception(self, df: pd.DataFrame, perception: str) -> str:
        """Analyze results for a specific perception."""
        analysis = []
        perc_df = df[df['perception'] == perception]
        
        # Best configuration for this perception
        best_idx = perc_df['f1_binary'].idxmax()
        best_row = perc_df.loc[best_idx]
        
        analysis.append(f"**Best Configuration:**")
        analysis.append(f"- Model: {best_row['model'].upper()}")
        analysis.append(f"- Delta: {best_row['delta']}")
        analysis.append(f"- F1 Score: {best_row['f1_binary']:.4f}")
        analysis.append(f"- ROC-AUC: {best_row['roc_auc_binary']:.4f}")
        analysis.append("")
        
        # Performance by delta
        delta_perf = perc_df.groupby('delta')['f1_binary'].agg(['mean', 'std']).round(4)
        optimal_delta = delta_perf['mean'].idxmax()
        
        analysis.append(f"**Optimal Delta:** {optimal_delta} (Mean F1: {delta_perf.loc[optimal_delta, 'mean']:.4f})")
        analysis.append("")
        
        # Model ranking for this perception
        model_perf = perc_df.groupby('model')['f1_binary'].mean().sort_values(ascending=False)
        
        analysis.append("**Model Ranking:**")
        for i, (model, score) in enumerate(model_perf.items(), 1):
            analysis.append(f"{i}. {model.upper()}: {score:.4f}")
        
        return "\n".join(analysis)
    
    def _analyze_delta_sensitivity(self, df: pd.DataFrame) -> str:
        """Analyze delta sensitivity."""
        analysis = []
        
        # Overall delta performance
        delta_stats = df.groupby('delta')[['f1_binary', 'roc_auc_binary']].agg(['mean', 'std'])
        
        analysis.append("| Delta | F1 Mean | F1 Std | ROC-AUC Mean | ROC-AUC Std |")
        analysis.append("|-------|---------|--------|--------------|-------------|")
        
        for delta in delta_stats.index:
            f1_mean = delta_stats.loc[delta, ('f1_binary', 'mean')]
            f1_std = delta_stats.loc[delta, ('f1_binary', 'std')]
            roc_mean = delta_stats.loc[delta, ('roc_auc_binary', 'mean')]
            roc_std = delta_stats.loc[delta, ('roc_auc_binary', 'std')]
            
            analysis.append(f"| {delta} | {f1_mean:.4f} | {f1_std:.4f} | "
                          f"{roc_mean:.4f} | {roc_std:.4f} |")
        
        analysis.append("")
        
        # Optimal delta per perception
        analysis.append("**Optimal Delta per Perception:**")
        analysis.append("")
        
        for perception in sorted(df['perception'].unique()):
            perc_df = df[df['perception'] == perception]
            delta_means = perc_df.groupby('delta')['f1_binary'].mean()
            best_delta = delta_means.idxmax()
            best_score = delta_means[best_delta]
            
            analysis.append(f"- {perception.title()}: δ = {best_delta} (F1: {best_score:.4f})")
        
        return "\n".join(analysis)
    
    def _analyze_model_stability(self, df: pd.DataFrame) -> str:
        """Analyze model stability across deltas."""
        analysis = []
        
        stability_metrics = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Calculate coefficient of variation
            cv_f1 = model_df['f1_binary'].std() / model_df['f1_binary'].mean()
            cv_roc = model_df['roc_auc_binary'].std() / model_df['roc_auc_binary'].mean()
            
            # Calculate range
            f1_range = model_df['f1_binary'].max() - model_df['f1_binary'].min()
            
            stability_metrics.append({
                'model': model.upper(),
                'cv_f1': cv_f1,
                'cv_roc': cv_roc,
                'f1_range': f1_range,
                'min_f1': model_df['f1_binary'].min(),
                'max_f1': model_df['f1_binary'].max()
            })
        
        stability_df = pd.DataFrame(stability_metrics).sort_values('cv_f1')
        
        analysis.append("**Model Stability Ranking (by Coefficient of Variation):**")
        analysis.append("")
        analysis.append("| Rank | Model | CV(F1) | CV(ROC) | F1 Range | Min F1 | Max F1 |")
        analysis.append("|------|-------|--------|---------|----------|--------|--------|")
        
        for i, row in enumerate(stability_df.itertuples(), 1):
            analysis.append(f"| {i} | {row.model} | {row.cv_f1:.4f} | {row.cv_roc:.4f} | "
                          f"{row.f1_range:.4f} | {row.min_f1:.4f} | {row.max_f1:.4f} |")
        
        analysis.append("")
        analysis.append("*Note: Lower CV indicates more stable performance across delta values*")
        
        return "\n".join(analysis)
    
    def _compare_models(self, df: pd.DataFrame) -> str:
        """Generate head-to-head model comparison."""
        analysis = []
        
        models = sorted(df['model'].unique())
        n_models = len(models)
        
        # Create win matrix
        win_matrix = pd.DataFrame(0, index=models, columns=models)
        
        for perception in df['perception'].unique():
            for delta in df['delta'].unique():
                config_df = df[(df['perception'] == perception) & (df['delta'] == delta)]
                if len(config_df) > 1:
                    sorted_models = config_df.sort_values('f1_binary', ascending=False)['model'].values
                    for i in range(len(sorted_models)):
                        for j in range(i+1, len(sorted_models)):
                            win_matrix.loc[sorted_models[i], sorted_models[j]] += 1
        
        analysis.append("**Head-to-Head Win Matrix:**")
        analysis.append("*(Number of configurations where row model beats column model)*")
        analysis.append("")
        
        # Format as table
        header = "| Model |" + " | ".join([m.upper()[:3] for m in models]) + " | Total Wins |"
        separator = "|-------|" + "------|" * (n_models + 1)
        
        analysis.append(header)
        analysis.append(separator)
        
        for model in models:
            row = f"| {model.upper()[:3]} |"
            total_wins = 0
            for other_model in models:
                if model == other_model:
                    row += " - |"
                else:
                    wins = win_matrix.loc[model, other_model]
                    total_wins += wins
                    row += f" {wins} |"
            row += f" {total_wins} |"
            analysis.append(row)
        
        return "\n".join(analysis)
    
    def _test_statistical_significance(self, df: pd.DataFrame) -> str:
        """Test statistical significance between models."""
        analysis = []
        
        models = sorted(df['model'].unique())
        
        analysis.append("**Pairwise T-Test Results (p-values for F1 scores):**")
        analysis.append("")
        
        # Perform pairwise t-tests
        p_values = []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:
                    scores1 = df[df['model'] == model1]['f1_binary'].values
                    scores2 = df[df['model'] == model2]['f1_binary'].values
                    
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    mean_diff = scores1.mean() - scores2.mean()
                    
                    p_values.append({
                        'model1': model1.upper(),
                        'model2': model2.upper(),
                        'mean_diff': mean_diff,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        if p_values:
            # Sort by p-value
            p_values_df = pd.DataFrame(p_values).sort_values('p_value')
            
            analysis.append("| Model 1 | Model 2 | Mean Diff | P-Value | Significant |")
            analysis.append("|---------|---------|-----------|---------|-------------|")
            
            for row in p_values_df.itertuples():
                sig_marker = "✓" if row.significant else "✗"
                analysis.append(f"| {row.model1} | {row.model2} | {row.mean_diff:+.4f} | "
                              f"{row.p_value:.4f} | {sig_marker} |")
            
            analysis.append("")
            analysis.append("*Significance level: α = 0.05*")
        else:
            analysis.append("*No pairwise comparisons available (insufficient model pairs)*")
            analysis.append("")
        
        return "\n".join(analysis)
    
    def _analyze_efficiency(self, df: pd.DataFrame) -> str:
        """Analyze computational efficiency."""
        analysis = []
        
        if 'train_time' in df.columns:
            # Training time analysis
            time_stats = df.groupby('model')['train_time'].agg(['mean', 'std', 'min', 'max'])
            time_stats = time_stats.sort_values('mean')
            
            analysis.append("**Training Time Analysis (seconds):**")
            analysis.append("")
            analysis.append("| Model | Mean | Std | Min | Max |")
            analysis.append("|-------|------|-----|-----|-----|")
            
            for model in time_stats.index:
                analysis.append(f"| {model.upper()} | {time_stats.loc[model, 'mean']:.2f} | "
                              f"{time_stats.loc[model, 'std']:.2f} | "
                              f"{time_stats.loc[model, 'min']:.2f} | "
                              f"{time_stats.loc[model, 'max']:.2f} |")
            
            analysis.append("")
            
            # Efficiency score (F1 / training time)
            df['efficiency'] = df['f1_binary'] / (df['train_time'] + 1e-6)
            eff_means = df.groupby('model')['efficiency'].mean().sort_values(ascending=False)
            
            analysis.append("**Efficiency Ranking (F1/Time):**")
            for i, (model, eff) in enumerate(eff_means.items(), 1):
                analysis.append(f"{i}. {model.upper()}: {eff:.4f}")
        else:
            analysis.append("*Training time data not available*")
        
        return "\n".join(analysis)
    
    def _generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Overall best model
        mean_f1 = df.groupby('model')['f1_binary'].mean()
        best_model = mean_f1.idxmax()
        
        recommendations.append(f"### For General Use:")
        recommendations.append(f"**Recommended Model:** {best_model.upper()}")
        recommendations.append(f"- Highest average F1 score: {mean_f1[best_model]:.4f}")
        recommendations.append("")
        
        # Optimal delta
        delta_means = df.groupby('delta')['f1_binary'].mean()
        best_delta = delta_means.idxmax()
        
        recommendations.append(f"**Recommended Delta:** {best_delta}")
        recommendations.append(f"- Best average performance across all models")
        recommendations.append("")
        
        # Per-perception recommendations
        recommendations.append("### Perception-Specific Recommendations:")
        recommendations.append("")
        
        for perception in sorted(df['perception'].unique()):
            perc_df = df[df['perception'] == perception]
            best_idx = perc_df['f1_binary'].idxmax()
            best_config = perc_df.loc[best_idx]
            
            recommendations.append(f"**{perception.title()}:**")
            recommendations.append(f"- Model: {best_config['model'].upper()}")
            recommendations.append(f"- Delta: {best_config['delta']}")
            recommendations.append(f"- Expected F1: {best_config['f1_binary']:.4f}")
            recommendations.append("")
        
        # Stability vs Performance trade-off
        recommendations.append("### Stability vs Performance Trade-off:")
        
        # Calculate stability scores
        stability_scores = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            cv = model_df['f1_binary'].std() / model_df['f1_binary'].mean()
            mean_f1_score = model_df['f1_binary'].mean()
            stability_scores.append({
                'model': model,
                'cv': cv,
                'mean_f1': mean_f1_score
            })
        
        stability_df = pd.DataFrame(stability_scores)
        
        # Find best balance (low CV, high F1)
        stability_df['balance_score'] = stability_df['mean_f1'] / (1 + stability_df['cv'])
        best_balance = stability_df.loc[stability_df['balance_score'].idxmax()]
        
        recommendations.append(f"**Best Balance:** {best_balance['model'].upper()}")
        recommendations.append(f"- Good performance (F1: {best_balance['mean_f1']:.4f}) "
                             f"with stability (CV: {best_balance['cv']:.4f})")
        
        return "\n".join(recommendations)
    
    def _generate_metrics_table(self, df: pd.DataFrame) -> str:
        """Generate detailed metrics table."""
        table = []
        
        # Select key columns
        key_cols = ['perception', 'model', 'delta', 'f1_binary', 'roc_auc_binary', 
                   'pr_auc_binary', 'accuracy_binary']
        
        if 'train_time' in df.columns:
            key_cols.append('train_time')
        
        # Sort by F1 score
        sorted_df = df[key_cols].sort_values('f1_binary', ascending=False).head(20)
        
        table.append("**Top 20 Configurations:**")
        table.append("")
        
        # Create markdown table
        headers = ['Perception', 'Model', 'Delta', 'F1', 'ROC-AUC', 'PR-AUC', 'Accuracy']
        if 'train_time' in key_cols:
            headers.append('Time(s)')
        
        table.append("| " + " | ".join(headers) + " |")
        table.append("|" + "---|" * len(headers))
        
        for row in sorted_df.itertuples():
            row_str = f"| {row.perception[:4]} | {row.model[:3].upper()} | {row.delta} | "
            row_str += f"{row.f1_binary:.3f} | {row.roc_auc_binary:.3f} | "
            row_str += f"{row.pr_auc_binary:.3f} | {row.accuracy_binary:.3f} |"
            
            if 'train_time' in key_cols:
                row_str += f" {row.train_time:.1f} |"
            
            table.append(row_str)
        
        return "\n".join(table)
    
    def _generate_html_report(self, df: pd.DataFrame, config: Dict = None) -> Path:
        """Generate HTML report with styling."""
        report_path = self.reports_dir / "comprehensive_analysis_report.html"
        
        # Read markdown report and convert to HTML
        md_report_path = self.reports_dir / "comprehensive_analysis_report.md"
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Delta Sensitivity Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .summary-box {
            background-color: #ecf0f1;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .recommendation {
            background-color: #e8f6f3;
            border-left: 5px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }
        .warning {
            background-color: #fef5e7;
            border-left: 5px solid #f39c12;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
"""
        
        # Convert markdown content to HTML (simplified conversion)
        if md_report_path.exists():
            with open(md_report_path, 'r') as f:
                md_content = f.read()
                
            # Simple markdown to HTML conversion
            html_body = md_content
            html_body = html_body.replace('# ', '<h1>').replace('\n\n', '</h1>\n\n')
            html_body = html_body.replace('## ', '<h2>').replace('\n\n', '</h2>\n\n')
            html_body = html_body.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')
            html_body = html_body.replace('**', '<strong>').replace('**', '</strong>')
            html_body = html_body.replace('*', '<em>').replace('*', '</em>')
            
            # Convert tables
            lines = html_body.split('\n')
            in_table = False
            converted_lines = []
            
            for line in lines:
                if line.startswith('|') and not in_table:
                    in_table = True
                    converted_lines.append('<table>')
                    # Header row
                    headers = [h.strip() for h in line.split('|')[1:-1]]
                    converted_lines.append('<tr>')
                    for header in headers:
                        converted_lines.append(f'<th>{header}</th>')
                    converted_lines.append('</tr>')
                elif line.startswith('|---') and in_table:
                    continue  # Skip separator line
                elif line.startswith('|') and in_table:
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    converted_lines.append('<tr>')
                    for cell in cells:
                        converted_lines.append(f'<td>{cell}</td>')
                    converted_lines.append('</tr>')
                elif in_table and not line.startswith('|'):
                    in_table = False
                    converted_lines.append('</table>')
                    converted_lines.append(line)
                else:
                    converted_lines.append(line)
            
            if in_table:
                converted_lines.append('</table>')
            
            html_body = '\n'.join(converted_lines)
            html_content += html_body
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to: {report_path}")
        return report_path
    
    def _generate_json_summary(self, df: pd.DataFrame, config: Dict = None) -> Path:
        """Generate JSON summary of key findings."""
        summary_path = self.reports_dir / "analysis_summary.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(df),
            'perceptions': list(df['perception'].unique()),
            'models': list(df['model'].unique()),
            'delta_values': sorted(df['delta'].unique().tolist()),
            'best_overall': {},
            'best_per_perception': {},
            'best_per_model': {},
            'mean_performance': {},
            'optimal_deltas': {},
            'statistical_tests': {}
        }
        
        # Best overall
        best_idx = df['f1_binary'].idxmax()
        best_row = df.loc[best_idx]
        summary['best_overall'] = {
            'model': best_row['model'],
            'perception': best_row['perception'],
            'delta': float(best_row['delta']),
            'f1_score': float(best_row['f1_binary']),
            'roc_auc': float(best_row['roc_auc_binary']),
            'pr_auc': float(best_row['pr_auc_binary'])
        }
        
        # Best per perception
        for perception in df['perception'].unique():
            perc_df = df[df['perception'] == perception]
            best_idx = perc_df['f1_binary'].idxmax()
            best_row = perc_df.loc[best_idx]
            summary['best_per_perception'][perception] = {
                'model': best_row['model'],
                'delta': float(best_row['delta']),
                'f1_score': float(best_row['f1_binary'])
            }
        
        # Best per model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_idx = model_df['f1_binary'].idxmax()
            best_row = model_df.loc[best_idx]
            summary['best_per_model'][model] = {
                'perception': best_row['perception'],
                'delta': float(best_row['delta']),
                'f1_score': float(best_row['f1_binary'])
            }
        
        # Mean performance
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            summary['mean_performance'][model] = {
                'f1_mean': float(model_df['f1_binary'].mean()),
                'f1_std': float(model_df['f1_binary'].std()),
                'roc_auc_mean': float(model_df['roc_auc_binary'].mean()),
                'pr_auc_mean': float(model_df['pr_auc_binary'].mean())
            }
        
        # Optimal deltas
        for perception in df['perception'].unique():
            perc_df = df[df['perception'] == perception]
            delta_means = perc_df.groupby('delta')['f1_binary'].mean()
            best_delta = delta_means.idxmax()
            summary['optimal_deltas'][perception] = float(best_delta)
        
        # Add configuration if provided
        if config:
            summary['experiment_config'] = config
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"JSON summary saved to: {summary_path}")
        return summary_path
    
    def _generate_csv_analysis(self, df: pd.DataFrame) -> Path:
        """Generate CSV with detailed analysis."""
        csv_path = self.reports_dir / "detailed_analysis.csv"
        
        # Add calculated columns
        analysis_df = df.copy()
        
        # Add rank columns
        analysis_df['f1_rank'] = analysis_df['f1_binary'].rank(ascending=False, method='min')
        analysis_df['roc_rank'] = analysis_df['roc_auc_binary'].rank(ascending=False, method='min')
        
        # Add normalized scores
        analysis_df['f1_normalized'] = (analysis_df['f1_binary'] - analysis_df['f1_binary'].min()) / \
                                       (analysis_df['f1_binary'].max() - analysis_df['f1_binary'].min())
        
        # Add composite score
        analysis_df['composite_score'] = (analysis_df['f1_binary'] + 
                                         analysis_df['roc_auc_binary'] + 
                                         analysis_df['pr_auc_binary']) / 3
        
        # Sort by composite score
        analysis_df = analysis_df.sort_values('composite_score', ascending=False)
        
        # Save to CSV
        analysis_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"CSV analysis saved to: {csv_path}")
        return csv_path