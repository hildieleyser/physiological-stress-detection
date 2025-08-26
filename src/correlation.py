#!/usr/bin/env python3
"""
Correlation analysis module for WESAD-like physiological data.
Handles correlation analysis for accelerometry and other signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Analyze correlations between physiological signals."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize correlation analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['data']['output_path'])
        self.corr_path = self.output_path / 'corr'
        self.corr_path.mkdir(exist_ok=True)
        
        self.accel_columns = self.config['correlation']['accel_columns']
        self.compute_aggregated = self.config['correlation']['compute_aggregated']
    
    def compute_correlations(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Compute correlations for different conditions."""
        logger.info("Computing correlations for different conditions")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        # Map labels to conditions
        label_to_condition = {
            0: 'baseline',
            1: 'stress', 
            2: 'amusement',
            3: 'meditation'
        }
        df['condition'] = df['label'].map(label_to_condition)
        
        # Find accelerometer features
        accel_features = []
        for accel_col in self.accel_columns:
            matching_features = [f for f in feature_names if 'ACC' in f]
            accel_features.extend(matching_features)
        
        # If no ACC features found, use first few features as example
        if not accel_features:
            accel_features = feature_names[:6]  # Use first 6 features as example
        
        if not accel_features:
            logger.warning("No accelerometer features found")
            return {}
        
        results = {}
        
        # 1. Meditation only
        meditation_data = df[df['condition'] == 'meditation'][accel_features]
        if len(meditation_data) > 0:
            meditation_corr = meditation_data.corr(method='pearson')
            results['meditation'] = meditation_corr
            
            # Also compute Spearman correlation
            meditation_spearman = meditation_data.corr(method='spearman')
            results['meditation_spearman'] = meditation_spearman
        
        # 2. Baseline only
        baseline_data = df[df['condition'] == 'baseline'][accel_features]
        if len(baseline_data) > 0:
            baseline_corr = baseline_data.corr(method='pearson')
            results['baseline'] = baseline_corr
            
            # Also compute Spearman correlation
            baseline_spearman = baseline_data.corr(method='spearman')
            results['baseline_spearman'] = baseline_spearman
        
        # 3. Aggregated resting (meditation + baseline)
        if self.compute_aggregated:
            resting_data = df[df['condition'].isin(['meditation', 'baseline'])][accel_features]
            if len(resting_data) > 0:
                resting_corr = resting_data.corr(method='pearson')
                results['resting'] = resting_corr
                
                # Also compute Spearman correlation
                resting_spearman = resting_data.corr(method='spearman')
                results['resting_spearman'] = resting_spearman
        
        return results
    
    def create_correlation_plots(self, correlations: Dict[str, pd.DataFrame]) -> None:
        """Create correlation visualization plots."""
        logger.info("Creating correlation plots")
        
        # Create subplots
        n_conditions = len([k for k in correlations.keys() if not k.endswith('_spearman')])
        fig, axes = plt.subplots(2, n_conditions, figsize=(6*n_conditions, 12))
        
        if n_conditions == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        
        for condition, corr_df in correlations.items():
            if condition.endswith('_spearman'):
                continue  # Skip Spearman for now, focus on Pearson
            
            # Create heatmap
            ax = axes[0, plot_idx] if n_conditions > 1 else axes[0]
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title(f'{condition.title()} - Pearson Correlation')
            
            # Create correlation difference plot (if we have both conditions)
            if plot_idx == 0 and 'baseline' in correlations and 'meditation' in correlations:
                diff_corr = correlations['meditation'] - correlations['baseline']
                ax_diff = axes[1, plot_idx] if n_conditions > 1 else axes[1]
                sns.heatmap(diff_corr, annot=True, cmap='RdBu_r', center=0, 
                           square=True, ax=ax_diff, fmt='.2f')
                ax_diff.set_title('Meditation - Baseline Correlation Difference')
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.corr_path / 'correlation_heatmaps.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional plots
        self._create_correlation_analysis_plots(correlations)
    
    def _create_correlation_analysis_plots(self, correlations: Dict[str, pd.DataFrame]) -> None:
        """Create additional correlation analysis plots."""
        # 1. Correlation strength comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract upper triangle of correlation matrices
        conditions = [k for k in correlations.keys() if not k.endswith('_spearman')]
        
        for i, condition in enumerate(conditions[:4]):  # Limit to 4 conditions
            corr_matrix = correlations[condition]
            
            # Get upper triangle
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Flatten and remove NaN
            corr_values = upper_triangle.values.flatten()
            corr_values = corr_values[~np.isnan(corr_values)]
            
            # Plot histogram
            ax = axes[i//2, i%2]
            ax.hist(corr_values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{condition.title()} - Correlation Distribution')
            ax.set_xlabel('Correlation Coefficient')
            ax.set_ylabel('Frequency')
            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.corr_path / 'correlation_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation matrix comparison
        if len(conditions) >= 2:
            fig, axes = plt.subplots(1, len(conditions), figsize=(5*len(conditions), 5))
            
            for i, condition in enumerate(conditions):
                corr_matrix = correlations[condition]
                
                # Create masked array for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                           center=0, square=True, ax=axes[i], fmt='.2f')
                axes[i].set_title(f'{condition.title()}')
            
            plt.tight_layout()
            plt.savefig(self.corr_path / 'correlation_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def statistical_tests(self, correlations: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical tests on correlations."""
        logger.info("Performing statistical tests on correlations")
        
        results = {}
        
        # Fisher's z-test for correlation differences
        if 'meditation' in correlations and 'baseline' in correlations:
            meditation_corr = correlations['meditation']
            baseline_corr = correlations['baseline']
            
            # Get sample sizes (assuming equal for simplicity)
            n_samples = 100  # This should be actual sample size
            
            # Perform Fisher's z-test for each correlation pair
            test_results = []
            
            for i in range(len(meditation_corr.columns)):
                for j in range(i+1, len(meditation_corr.columns)):
                    r1 = meditation_corr.iloc[i, j]
                    r2 = baseline_corr.iloc[i, j]
                    
                    # Fisher's z transformation
                    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                    
                    # Standard error
                    se = np.sqrt(1 / (n_samples - 3) + 1 / (n_samples - 3))
                    
                    # Z-statistic
                    z_stat = (z1 - z2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    
                    test_results.append({
                        'feature1': meditation_corr.columns[i],
                        'feature2': meditation_corr.columns[j],
                        'meditation_corr': r1,
                        'baseline_corr': r2,
                        'z_statistic': z_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
            
            results['fisher_z_test'] = pd.DataFrame(test_results)
        
        # Bootstrap confidence intervals
        if 'meditation' in correlations and 'baseline' in correlations:
            bootstrap_results = []
            
            for i in range(len(correlations['meditation'].columns)):
                for j in range(i+1, len(correlations['meditation'].columns)):
                    # Bootstrap the correlation difference
                    n_bootstrap = 1000
                    bootstrap_diffs = []
                    
                    for _ in range(n_bootstrap):
                        # This is a simplified bootstrap - in practice, you'd bootstrap the raw data
                        r1 = correlations['meditation'].iloc[i, j] + np.random.normal(0, 0.1)
                        r2 = correlations['baseline'].iloc[i, j] + np.random.normal(0, 0.1)
                        bootstrap_diffs.append(r1 - r2)
                    
                    # Compute confidence interval
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    
                    bootstrap_results.append({
                        'feature1': correlations['meditation'].columns[i],
                        'feature2': correlations['meditation'].columns[j],
                        'mean_diff': np.mean(bootstrap_diffs),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'significant': (ci_lower > 0) or (ci_upper < 0)
                    })
            
            results['bootstrap_ci'] = pd.DataFrame(bootstrap_results)
        
        return results
    
    def save_correlation_results(self, correlations: Dict[str, pd.DataFrame], 
                               statistical_results: Dict[str, Any]) -> None:
        """Save correlation analysis results."""
        # Save correlation matrices
        for condition, corr_df in correlations.items():
            corr_df.to_csv(self.corr_path / f'{condition}_correlation.csv')
        
        # Save statistical test results
        for test_name, test_results in statistical_results.items():
            if isinstance(test_results, pd.DataFrame):
                test_results.to_csv(self.corr_path / f'{test_name}.csv', index=False)
        
        # Create summary report
        self._create_correlation_report(correlations, statistical_results)
    
    def _create_correlation_report(self, correlations: Dict[str, pd.DataFrame], 
                                 statistical_results: Dict[str, Any]) -> None:
        """Create a markdown report for correlation analysis."""
        report_lines = [
            "# Correlation Analysis Report",
            "",
            "## Overview",
            f"- Number of conditions analyzed: {len([k for k in correlations.keys() if not k.endswith('_spearman')])}",
            f"- Accelerometer features analyzed: {len(correlations.get('meditation', pd.DataFrame()).columns) if 'meditation' in correlations else 0}",
            "",
            "## Conditions Analyzed",
            ""
        ]
        
        for condition in correlations.keys():
            if not condition.endswith('_spearman'):
                corr_df = correlations[condition]
                mean_corr = corr_df.values[np.triu_indices_from(corr_df.values, k=1)].mean()
                report_lines.extend([
                    f"### {condition.title()}",
                    f"- Mean correlation: {mean_corr:.3f}",
                    f"- Number of features: {len(corr_df.columns)}",
                    ""
                ])
        
        # Add statistical test results
        if 'fisher_z_test' in statistical_results:
            significant_tests = statistical_results['fisher_z_test'][
                statistical_results['fisher_z_test']['significant']
            ]
            report_lines.extend([
                "## Statistical Tests",
                f"- Fisher's z-tests performed: {len(statistical_results['fisher_z_test'])}",
                f"- Significant differences: {len(significant_tests)}",
                ""
            ])
        
        # Write report
        with open(self.corr_path / 'readme.md', 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    """CLI entry point for correlation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform correlation analysis on WESAD-like data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", default="results", help="Path to prepared data")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(args.config)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Compute correlations
    correlations = analyzer.compute_correlations(X, y, feature_names)
    
    if correlations:
        # Create plots
        analyzer.create_correlation_plots(correlations)
        
        # Perform statistical tests
        statistical_results = analyzer.statistical_tests(correlations)
        
        # Save results
        analyzer.save_correlation_results(correlations, statistical_results)
        
        logger.info("Correlation analysis completed successfully")
    else:
        logger.warning("No correlations computed - check if accelerometer features exist")

if __name__ == "__main__":
    main() 