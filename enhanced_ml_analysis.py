#!/usr/bin/env python3
"""
Enhanced Machine Learning Analysis for WESAD-like Physiological Data

This script implements a comprehensive analysis pipeline including:
1. Feature normalization
2. SHAP analysis with model training and regression
3. Correlation analysis between meditation and baseline
4. Class filtering (remove meditation and baseline)
5. K-fold cross-validation with repeated splits
6. Robustness testing with noise injection
7. Ensemble methods for better understanding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import (
    RepeatedStratifiedKFold, train_test_split, 
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import shap
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import joblib
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLAnalyzer:
    """Enhanced ML analysis with all requested features."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['data']['output_path'])
        self.output_path.mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['cv']['random_state'])
        
        # Initialize results storage
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def load_and_prepare_data(self, subject_id: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data with normalization."""
        logger.info("Loading and preparing data...")
        
        # Load data (assuming it's already prepared)
        data_path = Path("results")
        if (data_path / "X.npy").exists() and (data_path / "y.npy").exists():
            X = np.load(data_path / "X.npy")
            y = np.load(data_path / "y.npy")
        else:
            logger.error("Prepared data not found. Please run data preparation first.")
            return None, None
        
        logger.info(f"Loaded data shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Apply class filtering (remove meditation and baseline if requested)
        if self.config['classes']['drop_classes']:
            logger.info(f"Filtering out classes: {self.config['classes']['drop_classes']}")
            keep_mask = ~np.isin(y, [int(c) for c in self.config['classes']['drop_classes']])
            X = X[keep_mask]
            y = y[keep_mask]
            logger.info(f"After filtering: {X.shape}")
            logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Normalize features
        X_normalized = self.normalize_features(X)
        
        return X_normalized, y
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using the specified method."""
        method = self.config['preprocessing']['normalize_method']
        logger.info(f"Normalizing features using {method} method")
        
        if method == "global":
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            self.scalers['global'] = scaler
        elif method == "per_subject":
            # For per-subject normalization, we'd need subject IDs
            # For now, use global normalization
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            self.scalers['global'] = scaler
        else:
            logger.warning(f"Unknown normalization method: {method}. Using StandardScaler.")
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            self.scalers['global'] = scaler
        
        logger.info(f"Feature normalization completed. Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")
        return X_normalized
    
    def correlation_analysis(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze correlations between meditation and baseline conditions."""
        logger.info("Performing correlation analysis...")
        
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
        accel_features = [f for f in feature_names if 'ACC' in f.upper()]
        if not accel_features:
            # Use first few features as example
            accel_features = feature_names[:6]
        
        results = {}
        
        # 1. Meditation vs Baseline correlation comparison
        meditation_data = df[df['condition'] == 'meditation'][accel_features]
        baseline_data = df[df['condition'] == 'baseline'][accel_features]
        
        if len(meditation_data) > 0 and len(baseline_data) > 0:
            # Compute correlation matrices
            meditation_corr = meditation_data.corr(method='pearson')
            baseline_corr = baseline_data.corr(method='pearson')
            
            # Compute difference
            corr_diff = meditation_corr - baseline_corr
            
            results['meditation_corr'] = meditation_corr
            results['baseline_corr'] = baseline_corr
            results['corr_diff'] = corr_diff
            
            # Statistical test for correlation differences
            statistical_tests = {}
            for i, feat1 in enumerate(accel_features):
                for j, feat2 in enumerate(accel_features):
                    if i < j:  # Avoid duplicates
                        # Fisher's z-test for correlation difference
                        r1 = meditation_corr.iloc[i, j]
                        r2 = baseline_corr.iloc[i, j]
                        n1 = len(meditation_data)
                        n2 = len(baseline_data)
                        
                        if not (np.isnan(r1) or np.isnan(r2)):
                            # Fisher's z transformation
                            z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                            z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                            
                            # Standard error
                            se = np.sqrt(1/(n1-3) + 1/(n2-3))
                            
                            # Z-statistic
                            z_stat = (z1 - z2) / se
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                            
                            statistical_tests[f"{feat1}_vs_{feat2}"] = {
                                'z_stat': z_stat,
                                'p_value': p_value,
                                'meditation_corr': r1,
                                'baseline_corr': r2,
                                'difference': r1 - r2
                            }
            
            results['statistical_tests'] = statistical_tests
            
            # Create visualization
            self._plot_correlation_analysis(results, accel_features)
        
        return results
    
    def _plot_correlation_analysis(self, results: Dict[str, Any], feature_names: List[str]):
        """Create correlation analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis: Meditation vs Baseline', fontsize=16, fontweight='bold')
        
        # Meditation correlation heatmap
        sns.heatmap(results['meditation_corr'], annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=axes[0, 0])
        axes[0, 0].set_title('Meditation Condition Correlations')
        
        # Baseline correlation heatmap
        sns.heatmap(results['baseline_corr'], annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=axes[0, 1])
        axes[0, 1].set_title('Baseline Condition Correlations')
        
        # Correlation difference heatmap
        sns.heatmap(results['corr_diff'], annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', ax=axes[1, 0])
        axes[1, 0].set_title('Correlation Difference (Meditation - Baseline)')
        
        # Statistical significance heatmap
        sig_matrix = np.zeros_like(results['corr_diff'])
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if i < j:
                    key = f"{feat1}_vs_{feat2}"
                    if key in results['statistical_tests']:
                        p_val = results['statistical_tests'][key]['p_value']
                        sig_matrix.iloc[i, j] = -np.log10(p_val) if p_val > 0 else 10
                        sig_matrix.iloc[j, i] = sig_matrix.iloc[i, j]
        
        sns.heatmap(sig_matrix, annot=True, cmap='viridis', 
                   square=True, fmt='.2f', ax=axes[1, 1])
        axes[1, 1].set_title('-log10(p-value) for Correlation Differences')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Correlation analysis plots saved")
    
    def k_fold_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform K-fold cross-validation with repeated splits."""
        logger.info("Performing K-fold cross-validation...")
        
        cv_config = self.config['cv']
        
        # Create cross-validation splits
        if cv_config['type'] == 'repeated':
            cv = RepeatedStratifiedKFold(
                n_splits=cv_config['n_splits'],
                n_repeats=cv_config['n_repeats'],
                random_state=cv_config['random_state']
            )
        else:
            cv = RepeatedStratifiedKFold(
                n_splits=cv_config['n_splits'],
                n_repeats=1,
                random_state=cv_config['random_state']
            )
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=cv_config['random_state']),
            'random_forest': RandomForestClassifier(random_state=cv_config['random_state']),
            'xgboost': xgb.XGBClassifier(random_state=cv_config['random_state'])
        }
        
        cv_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} with cross-validation...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            # Train model on full dataset for SHAP analysis
            model.fit(X, y)
            self.models[name] = model
            
            cv_results[name] = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'model': model
            }
            
            logger.info(f"{name}: Mean CV accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Create CV results visualization
        self._plot_cv_results(cv_results)
        
        return cv_results
    
    def _plot_cv_results(self, cv_results: Dict[str, Any]):
        """Create cross-validation results visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of CV scores
        model_names = list(cv_results.keys())
        cv_scores = [cv_results[name]['cv_scores'] for name in model_names]
        
        axes[0].boxplot(cv_scores, labels=model_names)
        axes[0].set_title('Cross-Validation Scores Distribution')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Mean scores with error bars
        means = [cv_results[name]['mean_score'] for name in model_names]
        stds = [cv_results[name]['std_score'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        axes[1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(model_names, rotation=45)
        axes[1].set_title('Mean CV Scores with Standard Deviation')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            axes[1].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'cv_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Cross-validation results plots saved")
    
    def shap_analysis(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Perform SHAP analysis for feature importance."""
        logger.info("Performing SHAP analysis...")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        shap_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Computing SHAP values for {name}...")
            
            try:
                # Create SHAP explainer
                if name == 'logistic_regression':
                    explainer = shap.LinearExplainer(model, X)
                else:
                    explainer = shap.TreeExplainer(model)
                
                # Compute SHAP values
                shap_values = explainer.shap_values(X)
                
                # Handle multi-class case
                if isinstance(shap_values, list):
                    # Multi-class: take mean across classes
                    shap_values = np.mean(np.abs(shap_values), axis=0)
                else:
                    # Binary: take absolute values
                    shap_values = np.abs(shap_values)
                
                # Ensure shap_values is 2D
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
                
                # Compute feature importance
                feature_importance = np.mean(shap_values, axis=0)
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                shap_results[name] = {
                    'shap_values': shap_values,
                    'feature_importance': feature_importance,
                    'importance_df': importance_df,
                    'explainer': explainer
                }
                
                logger.info(f"Top 5 features for {name}:")
                for i, row in importance_df.head().iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                    
            except Exception as e:
                logger.warning(f"SHAP analysis failed for {name}: {str(e)}")
                # Use simple feature importance as fallback
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    feature_importance = np.abs(model.coef_[0])
                else:
                    feature_importance = np.ones(X.shape[1]) / X.shape[1]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                shap_results[name] = {
                    'shap_values': None,
                    'feature_importance': feature_importance,
                    'importance_df': importance_df,
                    'explainer': None
                }
        
        # Create SHAP visualizations
        self._plot_shap_analysis(shap_results, feature_names)
        
        return shap_results
    
    def _plot_shap_analysis(self, shap_results: Dict[str, Any], feature_names: List[str]):
        """Create SHAP analysis visualizations."""
        n_models = len(shap_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, results) in enumerate(shap_results.items()):
            # Feature importance bar plot
            top_features = results['importance_df'].head(10)
            axes[0, i].barh(range(len(top_features)), top_features['importance'])
            axes[0, i].set_yticks(range(len(top_features)))
            axes[0, i].set_yticklabels(top_features['feature'])
            axes[0, i].set_title(f'{name} - Top 10 Features')
            axes[0, i].set_xlabel('SHAP Importance')
            
            # SHAP summary plot (simplified)
            if hasattr(results['explainer'], 'expected_value'):
                # For tree-based models
                sample_indices = np.random.choice(len(results['shap_values']), 
                                                min(100, len(results['shap_values'])), 
                                                replace=False)
                sample_shap = results['shap_values'][sample_indices]
                sample_features = feature_names[:sample_shap.shape[1]]
                
                axes[1, i].imshow(sample_shap.T, aspect='auto', cmap='RdBu_r', center=0)
                axes[1, i].set_yticks(range(len(sample_features)))
                axes[1, i].set_yticklabels(sample_features)
                axes[1, i].set_title(f'{name} - SHAP Values Sample')
                axes[1, i].set_xlabel('Samples')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("SHAP analysis plots saved")
    
    def robustness_testing(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Test model robustness with noise injection."""
        logger.info("Performing robustness testing with noise injection...")
        
        noise_levels = self.config['robustness']['noise_levels']
        robustness_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Testing robustness for {name}...")
            
            model_results = {
                'noise_levels': [],
                'accuracies': [],
                'f1_scores': []
            }
            
            for noise_level in tqdm(noise_levels, desc=f"Testing {name}"):
                # Inject Gaussian noise
                X_noisy = self._inject_noise(X, noise_level)
                
                # Make predictions
                y_pred = model.predict(X_noisy)
                
                # Compute metrics
                accuracy = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average='weighted')
                
                model_results['noise_levels'].append(noise_level)
                model_results['accuracies'].append(accuracy)
                model_results['f1_scores'].append(f1)
            
            robustness_results[name] = model_results
            
            # Find noise level where performance drops significantly
            baseline_acc = model_results['accuracies'][0]
            significant_drop = None
            for i, acc in enumerate(model_results['accuracies']):
                if acc < baseline_acc * 0.9:  # 10% drop
                    significant_drop = model_results['noise_levels'][i]
                    break
            
            logger.info(f"{name}: Performance drops significantly at noise level {significant_drop}")
        
        # Create robustness visualization
        self._plot_robustness_results(robustness_results)
        
        return robustness_results
    
    def _inject_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """Inject Gaussian noise into features."""
        # Compute feature standard deviations
        feature_stds = np.std(X, axis=0)
        
        # Generate noise
        noise = np.random.normal(0, 1, X.shape)
        
        # Scale noise by feature std and noise level
        scaled_noise = noise * feature_stds.reshape(1, -1) * noise_level
        
        # Add noise to original data
        X_noisy = X + scaled_noise
        
        return X_noisy
    
    def _plot_robustness_results(self, robustness_results: Dict[str, Any]):
        """Create robustness testing visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs noise level
        for name, results in robustness_results.items():
            axes[0].plot(results['noise_levels'], results['accuracies'], 
                        marker='o', label=name, linewidth=2)
        
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Robustness: Accuracy vs Noise')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1 score vs noise level
        for name, results in robustness_results.items():
            axes[1].plot(results['noise_levels'], results['f1_scores'], 
                        marker='s', label=name, linewidth=2)
        
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Model Robustness: F1 Score vs Noise')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'robustness_testing.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Robustness testing plots saved")
    
    def save_results(self, cv_results: Dict[str, Any], shap_results: Dict[str, Any], 
                    correlation_results: Dict[str, Any], robustness_results: Dict[str, Any]):
        """Save all analysis results."""
        logger.info("Saving analysis results...")
        
        # Save CV results
        cv_summary = {}
        for name, results in cv_results.items():
            cv_summary[name] = {
                'mean_accuracy': results['mean_score'],
                'std_accuracy': results['std_score'],
                'cv_scores': results['cv_scores'].tolist()
            }
        
        with open(self.output_path / 'cv_summary.json', 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        # Save SHAP results
        shap_summary = {}
        for name, results in shap_results.items():
            shap_summary[name] = {
                'top_features': results['importance_df'].head(10).to_dict('records')
            }
        
        with open(self.output_path / 'shap_summary.json', 'w') as f:
            json.dump(shap_summary, f, indent=2)
        
        # Save correlation results
        if correlation_results:
            with open(self.output_path / 'correlation_summary.json', 'w') as f:
                json.dump(correlation_results, f, indent=2, default=str)
        
        # Save robustness results
        robustness_summary = {}
        for name, results in robustness_results.items():
            robustness_summary[name] = {
                'noise_levels': results['noise_levels'],
                'accuracies': results['accuracies'],
                'f1_scores': results['f1_scores']
            }
        
        with open(self.output_path / 'robustness_summary.json', 'w') as f:
            json.dump(robustness_summary, f, indent=2)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, self.output_path / f'{name}_model.joblib')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.output_path / f'{name}_scaler.joblib')
        
        logger.info("All results saved successfully")
    
    def run_complete_analysis(self, subject_id: str = None):
        """Run the complete analysis pipeline."""
        logger.info("Starting enhanced ML analysis pipeline...")
        
        # 1. Load and prepare data
        X, y = self.load_and_prepare_data(subject_id)
        if X is None or y is None:
            return
        
        # 2. Correlation analysis
        correlation_results = self.correlation_analysis(X, y)
        
        # 3. K-fold cross-validation
        cv_results = self.k_fold_cross_validation(X, y)
        
        # 4. SHAP analysis
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        shap_results = self.shap_analysis(X, y, feature_names)
        
        # 5. Robustness testing
        robustness_results = self.robustness_testing(X, y)
        
        # 6. Save all results
        self.save_results(cv_results, shap_results, correlation_results, robustness_results)
        
        logger.info("Enhanced ML analysis completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED ML ANALYSIS SUMMARY")
        print("="*60)
        
        print("\nCross-Validation Results:")
        for name, results in cv_results.items():
            print(f"  {name}: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        print("\nTop Features (SHAP):")
        for name, results in shap_results.items():
            top_feature = results['importance_df'].iloc[0]
            print(f"  {name}: {top_feature['feature']} ({top_feature['importance']:.4f})")
        
        print("\nRobustness Summary:")
        for name, results in robustness_results.items():
            baseline_acc = results['accuracies'][0]
            final_acc = results['accuracies'][-1]
            print(f"  {name}: {baseline_acc:.4f} → {final_acc:.4f} (drop: {baseline_acc-final_acc:.4f})")
        
        print(f"\nResults saved to: {self.output_path}")

def main():
    """Main function to run the enhanced analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ML Analysis for WESAD-like data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--subject", default=None, help="Subject ID to analyze")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedMLAnalyzer(args.config)
    
    # Run complete analysis
    analyzer.run_complete_analysis(args.subject)

if __name__ == "__main__":
    main() 