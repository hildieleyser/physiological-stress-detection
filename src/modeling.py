#!/usr/bin/env python3
"""
Modeling module for WESAD-like physiological data.
Handles pipelines, cross-validation, training, and hyperparameter search.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.metrics import average_precision_score
import xgboost as xgb
import yaml
import logging
from tqdm import tqdm
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMLP(nn.Module):
    """Simple MLP for deep learning experiments."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [64, 32]):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for PyTorch models to work with sklearn."""
    
    def __init__(self, model_class, **model_kwargs):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = None
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create model
        self.model = self.model_class(X.shape[1], num_classes, **self.model_kwargs)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(50):  # Simple training
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.numpy()

class ModelTrainer:
    """Train and evaluate models with cross-validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['data']['output_path'])
        self.models = {}
        self.results = {}
        
        # Set random seeds
        np.random.seed(self.config['cv']['random_state'])
        torch.manual_seed(self.config['cv']['random_state'])
    
    def get_models(self) -> Dict[str, Any]:
        """Get model definitions."""
        models = {
            'logistic_regression': LogisticRegression(random_state=self.config['cv']['random_state']),
            'random_forest': RandomForestClassifier(random_state=self.config['cv']['random_state']),
            'xgboost': xgb.XGBClassifier(random_state=self.config['cv']['random_state'])
        }
        
        if self.config['models']['use_deep']:
            models['mlp'] = SklearnWrapper(SimpleMLP, hidden_dims=[64, 32])
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter search grids."""
        grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        return grids
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
        elif y_proba is not None:
            # Multi-class classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['pr_auc'] = average_precision_score(y_true, y_proba, average='macro')
        
        return metrics
    
    def train_cv(self, X: np.ndarray, y: np.ndarray, splits: List[Tuple]) -> Dict[str, Any]:
        """Train models with cross-validation."""
        logger.info("Starting cross-validation training...")
        
        models = self.get_models()
        results = {name: [] for name in models.keys()}
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="CV Folds")):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_results = {}
            
            for name, model in models.items():
                logger.info(f"Training {name} on fold {fold_idx + 1}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Compute metrics
                metrics = self.compute_metrics(y_val, y_pred, y_proba)
                metrics['fold'] = fold_idx
                metrics['model'] = name
                
                fold_results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'true_labels': y_val
                }
                
                results[name].append(fold_results[name])
        
        # Save per-fold results
        self._save_cv_results(results)
        
        return results
    
    def _save_cv_results(self, results: Dict[str, List]):
        """Save cross-validation results."""
        # Create metrics DataFrame
        all_metrics = []
        for model_name, fold_results in results.items():
            for fold_result in fold_results:
                all_metrics.append(fold_result['metrics'])
        
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.output_path / 'cv' / 'metrics_per_fold.csv', index=False)
        
        # Compute summary statistics
        summary = {}
        for model_name in results.keys():
            model_metrics = metrics_df[metrics_df['model'] == model_name]
            summary[model_name] = {
                'mean_accuracy': model_metrics['accuracy'].mean(),
                'std_accuracy': model_metrics['accuracy'].std(),
                'mean_f1_macro': model_metrics['f1_macro'].mean(),
                'std_f1_macro': model_metrics['f1_macro'].std(),
                'mean_roc_auc': model_metrics['roc_auc'].mean() if 'roc_auc' in model_metrics else None,
                'std_roc_auc': model_metrics['roc_auc'].std() if 'roc_auc' in model_metrics else None
            }
        
        # Save summary
        with open(self.output_path / 'cv' / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create plots
        self._create_cv_plots(results, metrics_df)
    
    def _create_cv_plots(self, results: Dict[str, List], metrics_df: pd.DataFrame):
        """Create cross-validation plots."""
        # Metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        sns.boxplot(data=metrics_df, x='model', y='accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy by Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Macro comparison
        sns.boxplot(data=metrics_df, x='model', y='f1_macro', ax=axes[0, 1])
        axes[0, 1].set_title('F1 Macro by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC AUC comparison (if available)
        if 'roc_auc' in metrics_df.columns:
            sns.boxplot(data=metrics_df, x='model', y='roc_auc', ax=axes[1, 0])
            axes[1, 0].set_title('ROC AUC by Model')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Confusion matrices
        best_model = metrics_df.groupby('model')['accuracy'].mean().idxmax()
        best_model_data = metrics_df[metrics_df['model'] == best_model]
        best_fold_idx = best_model_data['accuracy'].idxmax()
        best_fold = best_model_data.loc[best_fold_idx, 'fold']
        best_result = results[best_model][best_fold]
        
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model}')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'cv' / 'cv_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray, 
                            splits: List[Tuple]) -> Dict[str, Any]:
        """Perform hyperparameter search."""
        logger.info("Starting hyperparameter search...")
        
        models = self.get_models()
        grids = self.get_hyperparameter_grids()
        search_results = {}
        
        for model_name in models.keys():
            if model_name not in grids:
                continue
            
            logger.info(f"Performing hyperparameter search for {model_name}")
            
            # Use first split for hyperparameter search
            train_idx, val_idx = splits[0]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if self.config['hparams']['search_method'] == 'random':
                search = RandomizedSearchCV(
                    models[model_name],
                    grids[model_name],
                    n_iter=self.config['hparams']['n_iter'],
                    cv=3,
                    scoring='accuracy',
                    random_state=self.config['cv']['random_state'],
                    n_jobs=-1
                )
            else:
                search = GridSearchCV(
                    models[model_name],
                    grids[model_name],
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            search.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = search.predict(X_val)
            y_proba = search.predict_proba(X_val) if hasattr(search, 'predict_proba') else None
            metrics = self.compute_metrics(y_val, y_pred, y_proba)
            
            search_results[model_name] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'val_metrics': metrics,
                'best_model': search.best_estimator_
            }
        
        # Save hyperparameter results
        self._save_hparam_results(search_results)
        
        return search_results
    
    def _save_hparam_results(self, search_results: Dict[str, Any]):
        """Save hyperparameter search results."""
        # Save best parameters
        best_params = {name: result['best_params'] for name, result in search_results.items()}
        with open(self.output_path / 'hparams' / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save validation metrics
        val_metrics = []
        for name, result in search_results.items():
            metrics = result['val_metrics']
            metrics['model'] = name
            val_metrics.append(metrics)
        
        val_metrics_df = pd.DataFrame(val_metrics)
        val_metrics_df.to_csv(self.output_path / 'hparams' / 'validation_metrics.csv', index=False)
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      best_models: Dict[str, Any]) -> Dict[str, Any]:
        """Train ensemble of models."""
        logger.info("Training ensemble models...")
        
        n_models = self.config['models']['n_models_ensemble']
        ensemble_results = {}
        
        for model_name, best_model in best_models.items():
            logger.info(f"Training {n_models} {model_name} models for ensemble")
            
            ensemble_models = []
            ensemble_predictions = []
            
            for seed in range(n_models):
                # Set random seed
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Clone and train model
                if hasattr(best_model, 'clone'):
                    model = best_model.clone()
                else:
                    # For models without clone method, create new instance
                    model = type(best_model)(**best_model.get_params())
                
                model.fit(X, y)
                ensemble_models.append(model)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    ensemble_predictions.append(model.predict_proba(X))
                else:
                    # Convert predictions to probabilities
                    pred = model.predict(X)
                    proba = np.zeros((len(X), len(np.unique(y))))
                    for i, p in enumerate(pred):
                        proba[i, p] = 1.0
                    ensemble_predictions.append(proba)
            
            # Average predictions
            avg_predictions = np.mean(ensemble_predictions, axis=0)
            ensemble_pred = np.argmax(avg_predictions, axis=1)
            
            # Compute metrics
            metrics = self.compute_metrics(y, ensemble_pred, avg_predictions)
            
            ensemble_results[model_name] = {
                'models': ensemble_models,
                'predictions': ensemble_pred,
                'probabilities': avg_predictions,
                'metrics': metrics
            }
        
        # Save ensemble results
        self._save_ensemble_results(ensemble_results)
        
        return ensemble_results
    
    def _save_ensemble_results(self, ensemble_results: Dict[str, Any]):
        """Save ensemble results."""
        # Save ensemble metrics
        ensemble_metrics = []
        for name, result in ensemble_results.items():
            metrics = result['metrics']
            metrics['model'] = name
            ensemble_metrics.append(metrics)
        
        ensemble_metrics_df = pd.DataFrame(ensemble_metrics)
        ensemble_metrics_df.to_csv(self.output_path / 'ensemble_metrics.csv', index=False)
        
        # Save models
        for name, result in ensemble_results.items():
            for i, model in enumerate(result['models']):
                joblib.dump(model, self.output_path / f'{name}_ensemble_{i}.joblib')
        
        # Save ensemble manifest
        manifest = {
            'ensemble_models': list(ensemble_results.keys()),
            'n_models_per_ensemble': self.config['models']['n_models_ensemble']
        }
        with open(self.output_path / 'ensemble.json', 'w') as f:
            json.dump(manifest, f, indent=2)

def main():
    """CLI entry point for modeling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models on WESAD-like data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", default="results", help="Path to prepared data")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    
    with open(data_path / "splits.pkl", 'rb') as f:
        import pickle
        splits = pickle.load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    # Train with cross-validation
    cv_results = trainer.train_cv(X, y, splits)
    
    # Hyperparameter search
    hparam_results = trainer.hyperparameter_search(X, y, splits)
    
    # Train ensemble
    best_models = {name: result['best_model'] for name, result in hparam_results.items()}
    ensemble_results = trainer.train_ensemble(X, y, best_models)
    
    logger.info("Modeling completed successfully")

if __name__ == "__main__":
    main() 