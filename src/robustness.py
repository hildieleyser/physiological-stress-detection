#!/usr/bin/env python3
"""
Robustness testing module for WESAD-like physiological data.
Handles noise injection and model robustness evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustnessTester:
    """Test model robustness to noise injection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize robustness tester with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['data']['output_path'])
        self.robustness_path = self.output_path / 'robustness'
        self.robustness_path.mkdir(exist_ok=True)
        
        self.noise_levels = self.config['robustness']['noise_levels']
    
    def inject_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
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
    
    def evaluate_model_robustness(self, model, X: np.ndarray, y: np.ndarray, 
                                 model_name: str) -> Dict[str, List]:
        """Evaluate model robustness across different noise levels."""
        logger.info(f"Evaluating robustness for {model_name}")
        
        results = {
            'noise_level': [],
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'roc_auc': []
        }
        
        for noise_level in tqdm(self.noise_levels, desc=f"Testing {model_name}"):
            # Inject noise
            X_noisy = self.inject_noise(X, noise_level)
            
            # Make predictions
            y_pred = model.predict(X_noisy)
            y_proba = model.predict_proba(X_noisy) if hasattr(model, 'predict_proba') else None
            
            # Compute metrics
            accuracy = accuracy_score(y, y_pred)
            f1_macro = f1_score(y, y_pred, average='macro')
            f1_weighted = f1_score(y, y_pred, average='weighted')
            
            # ROC AUC (if probabilities available)
            roc_auc = None
            if y_proba is not None:
                if len(np.unique(y)) == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y, y_proba[:, 1])
                else:
                    # Multi-class classification
                    roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
            
            # Store results
            results['noise_level'].append(noise_level)
            results['accuracy'].append(accuracy)
            results['f1_macro'].append(f1_macro)
            results['f1_weighted'].append(f1_weighted)
            results['roc_auc'].append(roc_auc)
        
        return results
    
    def test_ensemble_robustness(self, ensemble_models: Dict[str, List], 
                                X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Test robustness of ensemble models."""
        logger.info("Testing ensemble model robustness")
        
        ensemble_results = {}
        
        for model_name, ensemble_info in ensemble_models.items():
            logger.info(f"Testing ensemble: {model_name}")
            
            ensemble_results[model_name] = {}
            
            # Test each individual model in ensemble
            individual_results = []
            for i, model in enumerate(ensemble_info['models']):
                results = self.evaluate_model_robustness(model, X, y, f"{model_name}_model_{i}")
                individual_results.append(results)
            
            # Test ensemble average
            ensemble_results[model_name]['individual'] = individual_results
            
            # Compute ensemble average predictions for each noise level
            ensemble_avg_results = {
                'noise_level': [],
                'accuracy': [],
                'f1_macro': [],
                'f1_weighted': [],
                'roc_auc': []
            }
            
            for noise_level in self.noise_levels:
                X_noisy = self.inject_noise(X, noise_level)
                
                # Get predictions from all models
                all_predictions = []
                all_probabilities = []
                
                for model in ensemble_info['models']:
                    pred = model.predict(X_noisy)
                    proba = model.predict_proba(X_noisy) if hasattr(model, 'predict_proba') else None
                    
                    all_predictions.append(pred)
                    if proba is not None:
                        all_probabilities.append(proba)
                
                # Average predictions
                if all_probabilities:
                    avg_proba = np.mean(all_probabilities, axis=0)
                    ensemble_pred = np.argmax(avg_proba, axis=1)
                    
                    # Compute metrics
                    accuracy = accuracy_score(y, ensemble_pred)
                    f1_macro = f1_score(y, ensemble_pred, average='macro')
                    f1_weighted = f1_score(y, ensemble_pred, average='weighted')
                    
                    # ROC AUC
                    roc_auc = None
                    if len(np.unique(y)) == 2:
                        roc_auc = roc_auc_score(y, avg_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y, avg_proba, multi_class='ovr', average='macro')
                else:
                    # Use voting for predictions
                    ensemble_pred = np.array(all_predictions).mean(axis=0).round().astype(int)
                    accuracy = accuracy_score(y, ensemble_pred)
                    f1_macro = f1_score(y, ensemble_pred, average='macro')
                    f1_weighted = f1_score(y, ensemble_pred, average='weighted')
                    roc_auc = None
                
                ensemble_avg_results['noise_level'].append(noise_level)
                ensemble_avg_results['accuracy'].append(accuracy)
                ensemble_avg_results['f1_macro'].append(f1_macro)
                ensemble_avg_results['f1_weighted'].append(f1_weighted)
                ensemble_avg_results['roc_auc'].append(roc_auc)
            
            ensemble_results[model_name]['ensemble_avg'] = ensemble_avg_results
        
        return ensemble_results
    
    def create_robustness_plots(self, results: Dict[str, Dict], 
                               ensemble_results: Dict[str, Dict] = None) -> None:
        """Create robustness visualization plots."""
        logger.info("Creating robustness plots")
        
        # Create main robustness plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Accuracy vs Noise Level
        for model_name, model_results in results.items():
            axes[0, 0].plot(model_results['noise_level'], model_results['accuracy'], 
                           marker='o', label=model_name, linewidth=2)
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                axes[0, 0].plot(avg_results['noise_level'], avg_results['accuracy'], 
                               marker='s', label=f"{model_name}_ensemble", linewidth=2, linestyle='--')
        
        axes[0, 0].set_xlabel('Noise Level (σ)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Robustness: Accuracy vs Noise')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: F1 Macro vs Noise Level
        for model_name, model_results in results.items():
            axes[0, 1].plot(model_results['noise_level'], model_results['f1_macro'], 
                           marker='o', label=model_name, linewidth=2)
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                axes[0, 1].plot(avg_results['noise_level'], avg_results['f1_macro'], 
                               marker='s', label=f"{model_name}_ensemble", linewidth=2, linestyle='--')
        
        axes[0, 1].set_xlabel('Noise Level (σ)')
        axes[0, 1].set_ylabel('F1 Macro')
        axes[0, 1].set_title('Model Robustness: F1 Macro vs Noise')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ROC AUC vs Noise Level (if available)
        for model_name, model_results in results.items():
            if any(x is not None for x in model_results['roc_auc']):
                valid_indices = [i for i, x in enumerate(model_results['roc_auc']) if x is not None]
                if valid_indices:
                    noise_levels = [model_results['noise_level'][i] for i in valid_indices]
                    roc_aucs = [model_results['roc_auc'][i] for i in valid_indices]
                    axes[1, 0].plot(noise_levels, roc_aucs, marker='o', label=model_name, linewidth=2)
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                if any(x is not None for x in avg_results['roc_auc']):
                    valid_indices = [i for i, x in enumerate(avg_results['roc_auc']) if x is not None]
                    if valid_indices:
                        noise_levels = [avg_results['noise_level'][i] for i in valid_indices]
                        roc_aucs = [avg_results['roc_auc'][i] for i in valid_indices]
                        axes[1, 0].plot(noise_levels, roc_aucs, marker='s', 
                                       label=f"{model_name}_ensemble", linewidth=2, linestyle='--')
        
        axes[1, 0].set_xlabel('Noise Level (σ)')
        axes[1, 0].set_ylabel('ROC AUC')
        axes[1, 0].set_title('Model Robustness: ROC AUC vs Noise')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance degradation summary
        degradation_data = []
        for model_name, model_results in results.items():
            baseline_acc = model_results['accuracy'][0]  # No noise
            for i, noise_level in enumerate(model_results['noise_level']):
                degradation = baseline_acc - model_results['accuracy'][i]
                degradation_data.append({
                    'model': model_name,
                    'noise_level': noise_level,
                    'degradation': degradation,
                    'baseline': baseline_acc
                })
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                baseline_acc = avg_results['accuracy'][0]
                for i, noise_level in enumerate(avg_results['noise_level']):
                    degradation = baseline_acc - avg_results['accuracy'][i]
                    degradation_data.append({
                        'model': f"{model_name}_ensemble",
                        'noise_level': noise_level,
                        'degradation': degradation,
                        'baseline': baseline_acc
                    })
        
        degradation_df = pd.DataFrame(degradation_data)
        sns.boxplot(data=degradation_df, x='noise_level', y='degradation', hue='model', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Noise Level (σ)')
        axes[1, 1].set_ylabel('Performance Degradation')
        axes[1, 1].set_title('Performance Degradation by Noise Level')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.robustness_path / 'robustness_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed degradation plot
        self._create_degradation_plot(results, ensemble_results)
    
    def _create_degradation_plot(self, results: Dict[str, Dict], 
                                ensemble_results: Dict[str, Dict] = None) -> None:
        """Create detailed performance degradation plot."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot degradation curves
        for model_name, model_results in results.items():
            baseline_acc = model_results['accuracy'][0]
            degradation = [baseline_acc - acc for acc in model_results['accuracy']]
            axes[0].plot(model_results['noise_level'], degradation, 
                        marker='o', label=model_name, linewidth=2)
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                baseline_acc = avg_results['accuracy'][0]
                degradation = [baseline_acc - acc for acc in avg_results['accuracy']]
                axes[0].plot(avg_results['noise_level'], degradation, 
                           marker='s', label=f"{model_name}_ensemble", linewidth=2, linestyle='--')
        
        axes[0].set_xlabel('Noise Level (σ)')
        axes[0].set_ylabel('Accuracy Degradation')
        axes[0].set_title('Performance Degradation Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot robustness threshold
        threshold_data = []
        for model_name, model_results in results.items():
            baseline_acc = model_results['accuracy'][0]
            for i, noise_level in enumerate(model_results['noise_level']):
                current_acc = model_results['accuracy'][i]
                if current_acc < 0.5:  # Random performance threshold
                    threshold_data.append({
                        'model': model_name,
                        'noise_threshold': noise_level,
                        'baseline_accuracy': baseline_acc,
                        'threshold_accuracy': current_acc
                    })
                    break
        
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                avg_results = ensemble_info['ensemble_avg']
                baseline_acc = avg_results['accuracy'][0]
                for i, noise_level in enumerate(avg_results['noise_level']):
                    current_acc = avg_results['accuracy'][i]
                    if current_acc < 0.5:
                        threshold_data.append({
                            'model': f"{model_name}_ensemble",
                            'noise_threshold': noise_level,
                            'baseline_accuracy': baseline_acc,
                            'threshold_accuracy': current_acc
                        })
                        break
        
        if threshold_data:
            threshold_df = pd.DataFrame(threshold_data)
            sns.barplot(data=threshold_df, x='model', y='noise_threshold', ax=axes[1])
            axes[1].set_title('Noise Threshold for Random Performance')
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Noise Level (σ)')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.robustness_path / 'degradation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_robustness_results(self, results: Dict[str, Dict], 
                               ensemble_results: Dict[str, Dict] = None) -> None:
        """Save robustness testing results."""
        # Save individual model results
        for model_name, model_results in results.items():
            results_df = pd.DataFrame(model_results)
            results_df.to_csv(self.robustness_path / f'{model_name}_robustness.csv', index=False)
        
        # Save ensemble results
        if ensemble_results:
            for model_name, ensemble_info in ensemble_results.items():
                # Save ensemble average results
                avg_results = ensemble_info['ensemble_avg']
                avg_df = pd.DataFrame(avg_results)
                avg_df.to_csv(self.robustness_path / f'{model_name}_ensemble_robustness.csv', index=False)
                
                # Save individual model results in ensemble
                for i, individual_results in enumerate(ensemble_info['individual']):
                    ind_df = pd.DataFrame(individual_results)
                    ind_df.to_csv(self.robustness_path / f'{model_name}_ensemble_model_{i}_robustness.csv', index=False)
        
        # Create summary report
        summary = {
            'noise_levels_tested': self.noise_levels,
            'models_tested': list(results.keys()),
            'ensemble_models_tested': list(ensemble_results.keys()) if ensemble_results else []
        }
        
        with open(self.robustness_path / 'robustness_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """CLI entry point for robustness testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model robustness to noise")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", default="results", help="Path to prepared data")
    parser.add_argument("--models-path", default="results", help="Path to trained models")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    
    # Initialize tester
    tester = RobustnessTester(args.config)
    
    # Load trained models
    models_path = Path(args.models_path)
    results = {}
    ensemble_results = {}
    
    # Load individual models
    for model_file in models_path.glob("*_ensemble_*.joblib"):
        model_name = model_file.stem.replace("_ensemble_", "_")
        model = joblib.load(model_file)
        
        # Test robustness
        model_results = tester.evaluate_model_robustness(model, X, y, model_name)
        results[model_name] = model_results
    
    # Load ensemble models
    if (models_path / "ensemble.json").exists():
        with open(models_path / "ensemble.json", 'r') as f:
            ensemble_info = json.load(f)
        
        for model_name in ensemble_info['ensemble_models']:
            ensemble_models = []
            for i in range(ensemble_info['n_models_per_ensemble']):
                model_file = models_path / f"{model_name}_ensemble_{i}.joblib"
                if model_file.exists():
                    model = joblib.load(model_file)
                    ensemble_models.append(model)
            
            if ensemble_models:
                ensemble_results[model_name] = {
                    'models': ensemble_models,
                    'predictions': None,  # Will be computed during testing
                    'probabilities': None
                }
    
    # Test ensemble robustness
    if ensemble_results:
        ensemble_robustness = tester.test_ensemble_robustness(ensemble_results, X, y)
        ensemble_results.update(ensemble_robustness)
    
    # Create plots and save results
    tester.create_robustness_plots(results, ensemble_results)
    tester.save_robustness_results(results, ensemble_results)
    
    logger.info("Robustness testing completed successfully")

if __name__ == "__main__":
    main() 