#!/usr/bin/env python3
"""
SHAP utilities for feature importance analysis.
Handles SHAP analysis for different model types and creates visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tqdm import tqdm
import joblib
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """Analyze feature importance using SHAP."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize SHAP analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['data']['output_path'])
        self.shap_path = self.output_path / 'shap'
        self.shap_path.mkdir(exist_ok=True)
    
    def create_explainer(self, model, X_sample: np.ndarray, model_type: str) -> Any:
        """Create appropriate SHAP explainer for model type."""
        if model_type == 'logistic_regression':
            return shap.LinearExplainer(model, X_sample)
        elif model_type in ['random_forest', 'xgboost']:
            return shap.TreeExplainer(model)
        else:
            # For other models, try to use KernelExplainer
            return shap.KernelExplainer(model.predict_proba, X_sample)
    
    def compute_shap_values(self, model, X: np.ndarray, model_type: str, 
                           max_samples: int = None) -> np.ndarray:
        """Compute SHAP values for a model."""
        if max_samples is None:
            max_samples = self.config['shap']['max_samples']
        
        # Sample data if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        logger.info(f"Computing SHAP values for {model_type} with {len(X_sample)} samples")
        
        try:
            explainer = self.create_explainer(model, X_sample, model_type)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values = np.array(shap_values)
            else:
                # Binary case
                shap_values = np.array([shap_values])
            
            return shap_values, X_sample
            
        except Exception as e:
            logger.warning(f"Could not compute SHAP for {model_type}: {e}")
            return None, None
    
    def analyze_global_importance(self, model, X: np.ndarray, y: np.ndarray, 
                                 model_type: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze global feature importance."""
        logger.info(f"Analyzing global SHAP importance for {model_type}")
        
        shap_values, X_sample = self.compute_shap_values(model, X, model_type)
        
        if shap_values is None:
            return None
        
        # Compute global importance (mean absolute SHAP values)
        if len(shap_values.shape) == 3:
            # Multi-class case
            global_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:
            # Binary case
            global_importance = np.mean(np.abs(shap_values[0]), axis=0)
        
        # Create feature importance DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(global_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        
        # Save importance rankings
        importance_df.to_csv(self.shap_path / f'{model_type}_global_importance.csv', index=False)
        
        return {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'importance_df': importance_df,
            'global_importance': global_importance
        }
    
    def create_shap_plots(self, shap_values: np.ndarray, X_sample: np.ndarray, 
                          model_type: str, feature_names: List[str] = None,
                          top_k: int = None) -> None:
        """Create SHAP visualization plots."""
        if top_k is None:
            top_k = self.config['shap']['top_k_features']
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
        
        # Set feature names for SHAP
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Summary plot
        if len(shap_values.shape) == 3:
            # Multi-class case - use first class for summary
            shap.summary_plot(shap_values[0], X_sample_df, 
                            show=False, max_display=top_k, ax=axes[0, 0])
        else:
            # Binary case
            shap.summary_plot(shap_values[0], X_sample_df, 
                            show=False, max_display=top_k, ax=axes[0, 0])
        axes[0, 0].set_title(f'SHAP Summary Plot - {model_type}')
        
        # 2. Bar plot of global importance
        importance_df = pd.read_csv(self.shap_path / f'{model_type}_global_importance.csv')
        top_features = importance_df.head(top_k)
        
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_title(f'Global Feature Importance - {model_type}')
        axes[0, 1].set_xlabel('Mean |SHAP|')
        
        # 3. Beeswarm plot
        if len(shap_values.shape) == 3:
            shap.plots.beeswarm(shap_values[0], show=False, max_display=top_k, ax=axes[1, 0])
        else:
            shap.plots.beeswarm(shap_values[0], show=False, max_display=top_k, ax=axes[1, 0])
        axes[1, 0].set_title(f'SHAP Beeswarm Plot - {model_type}')
        
        # 4. Dependence plots for top features
        top_feature = top_features.iloc[0]['feature']
        if len(shap_values.shape) == 3:
            shap.dependence_plot(top_feature, shap_values[0], X_sample_df, 
                               show=False, ax=axes[1, 1])
        else:
            shap.dependence_plot(top_feature, shap_values[0], X_sample_df, 
                               show=False, ax=axes[1, 1])
        axes[1, 1].set_title(f'Dependence Plot - {top_feature}')
        
        plt.tight_layout()
        plt.savefig(self.shap_path / f'{model_type}_shap_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_fold_importance(self, cv_results: Dict[str, List], X: np.ndarray, 
                               feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze feature importance across CV folds."""
        logger.info("Analyzing SHAP importance across CV folds")
        
        fold_importance = {}
        
        for model_name, fold_results in cv_results.items():
            logger.info(f"Analyzing {model_name} across folds")
            
            fold_importances = []
            
            for fold_idx, fold_result in enumerate(fold_results):
                model = fold_result['model']
                
                # Determine model type
                if isinstance(model, LogisticRegression):
                    model_type = 'logistic_regression'
                elif isinstance(model, RandomForestClassifier):
                    model_type = 'random_forest'
                elif isinstance(model, xgb.XGBClassifier):
                    model_type = 'xgboost'
                else:
                    model_type = 'other'
                
                # Compute SHAP values
                shap_values, X_sample = self.compute_shap_values(model, X, model_type)
                
                if shap_values is not None:
                    # Compute importance for this fold
                    if len(shap_values.shape) == 3:
                        importance = np.mean(np.abs(shap_values), axis=(0, 1))
                    else:
                        importance = np.mean(np.abs(shap_values[0]), axis=0)
                    
                    fold_importances.append(importance)
            
            if fold_importances:
                # Compute stability across folds
                fold_importances = np.array(fold_importances)
                mean_importance = np.mean(fold_importances, axis=0)
                std_importance = np.std(fold_importances, axis=0)
                
                # Create stability DataFrame
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(mean_importance))]
                
                stability_df = pd.DataFrame({
                    'feature': feature_names,
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'cv_stability': 1 - (std_importance / (mean_importance + 1e-8))
                }).sort_values('mean_importance', ascending=False)
                
                stability_df.to_csv(self.shap_path / f'{model_name}_fold_stability.csv', index=False)
                
                fold_importance[model_name] = {
                    'fold_importances': fold_importances,
                    'stability_df': stability_df,
                    'mean_importance': mean_importance,
                    'std_importance': std_importance
                }
        
        return fold_importance
    
    def univariate_screen(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None) -> pd.DataFrame:
        """Perform univariate feature screening."""
        logger.info("Performing univariate feature screening")
        
        from sklearn.linear_model import LogisticRegression
        from scipy import stats
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        results = []
        
        for i, feature_name in enumerate(tqdm(feature_names, desc="Univariate screening")):
            X_feature = X[:, i].reshape(-1, 1)
            
            # Fit logistic regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_feature, y)
            
            # Get p-value
            from sklearn.feature_selection import f_regression
            f_stat, p_value = f_regression(X_feature, y)
            
            # Compute effect size (R²)
            from sklearn.metrics import r2_score
            y_pred = lr.predict(X_feature)
            r2 = r2_score(y, y_pred)
            
            # Compute correlation
            correlation = np.corrcoef(X[:, i], y)[0, 1]
            
            results.append({
                'feature': feature_name,
                'p_value': p_value[0],
                'f_statistic': f_stat[0],
                'r2_score': r2,
                'correlation': correlation,
                'coefficient': lr.coef_[0, 0] if len(lr.coef_) > 0 else 0
            })
        
        # Create DataFrame and sort by p-value
        screen_df = pd.DataFrame(results)
        screen_df = screen_df.sort_values('p_value')
        
        # Add significance flags
        screen_df['significant_0.05'] = screen_df['p_value'] < 0.05
        screen_df['significant_0.01'] = screen_df['p_value'] < 0.01
        screen_df['significant_0.001'] = screen_df['p_value'] < 0.001
        
        # Save results
        screen_df.to_csv(self.output_path / 'univariate_screen.csv', index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # P-value distribution
        axes[0, 0].hist(screen_df['p_value'], bins=20, alpha=0.7)
        axes[0, 0].axvline(0.05, color='red', linestyle='--', label='p=0.05')
        axes[0, 0].set_title('P-value Distribution')
        axes[0, 0].set_xlabel('P-value')
        axes[0, 0].legend()
        
        # Correlation vs p-value
        axes[0, 1].scatter(screen_df['correlation'], -np.log10(screen_df['p_value']))
        axes[0, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0, 1].set_title('Correlation vs -log10(p-value)')
        axes[0, 1].set_xlabel('Correlation')
        axes[0, 1].set_ylabel('-log10(p-value)')
        axes[0, 1].legend()
        
        # Top features by importance
        top_features = screen_df.head(10)
        axes[1, 0].barh(range(len(top_features)), -np.log10(top_features['p_value']))
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_title('Top Features by Significance')
        axes[1, 0].set_xlabel('-log10(p-value)')
        
        # R² vs p-value
        axes[1, 1].scatter(screen_df['r2_score'], -np.log10(screen_df['p_value']))
        axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1, 1].set_title('R² vs -log10(p-value)')
        axes[1, 1].set_xlabel('R²')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'univariate_screen.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return screen_df

def main():
    """CLI entry point for SHAP analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform SHAP analysis on WESAD-like data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", default="results", help="Path to prepared data")
    parser.add_argument("--cv-results", help="Path to CV results pickle file")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer(args.config)
    
    # Perform univariate screening
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    screen_df = analyzer.univariate_screen(X, y, feature_names)
    
    # Load CV results if available
    if args.cv_results:
        with open(args.cv_results, 'rb') as f:
            import pickle
            cv_results = pickle.load(f)
        
        # Analyze fold importance
        fold_importance = analyzer.analyze_fold_importance(cv_results, X, feature_names)
    
    logger.info("SHAP analysis completed successfully")

if __name__ == "__main__":
    main() 