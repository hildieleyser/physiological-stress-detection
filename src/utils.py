#!/usr/bin/env python3
"""
Utilities module for WESAD-like physiological data analysis.
Handles metrics, plotting, seeds, and common utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import random
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_micro': f1_score(y_true, y_pred, average='micro')
    }
    
    if y_proba is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = precision_recall_auc_score(y_true, y_proba[:, 1])
        else:
            # Multi-class classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['pr_auc'] = precision_recall_auc_score(y_true, y_proba, average='macro')
    
    return metrics

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: List[str] = None, 
                                title: str = "Confusion Matrix") -> plt.Figure:
    """Create confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    return fig

def create_roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, 
                          class_names: List[str] = None) -> plt.Figure:
    """Create ROC curve visualization."""
    n_classes = y_proba.shape[1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class classification
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
            roc_auc = roc_auc_score(y_true == i, y_proba[:, i])
            class_name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_pr_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, 
                        class_names: List[str] = None) -> plt.Figure:
    """Create Precision-Recall curve visualization."""
    n_classes = y_proba.shape[1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if n_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = precision_recall_auc_score(y_true, y_proba[:, 1])
        ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    else:
        # Multi-class classification
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
            pr_auc = precision_recall_auc_score(y_true == i, y_proba[:, i])
            class_name = class_names[i] if class_names else f'Class {i}'
            ax.plot(recall, precision, label=f'{class_name} (AUC = {pr_auc:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_metrics_comparison_plot(metrics_df: pd.DataFrame, 
                                  metric: str = 'accuracy') -> plt.Figure:
    """Create comparison plot for metrics across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=metrics_df, x='model', y=metric, ax=ax)
    ax.set_title(f'{metric.title()} Comparison Across Models')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.title())
    ax.tick_params(axis='x', rotation=45)
    
    return fig

def save_plots(figures: Dict[str, plt.Figure], output_path: Path, 
               prefix: str = "") -> None:
    """Save multiple figures to files."""
    output_path.mkdir(exist_ok=True)
    
    for name, fig in figures.items():
        filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_summary_report(metrics_df: pd.DataFrame, 
                         config: Dict[str, Any],
                         output_path: Path) -> None:
    """Create a human-readable summary report."""
    report_lines = [
        "# WESAD-like Physiological Data Analysis Summary",
        "",
        "## Configuration",
        f"- Data path: {config['data']['data_path']}",
        f"- Subject ID: {config['data']['subject_id']}",
        f"- Window length: {config['preprocessing']['window_len']}",
        f"- Stride: {config['preprocessing']['stride']}",
        f"- Normalization: {config['preprocessing']['normalize_method']}",
        "",
        "## Model Performance Summary",
        ""
    ]
    
    # Add model performance summary
    for model_name in metrics_df['model'].unique():
        model_metrics = metrics_df[metrics_df['model'] == model_name]
        report_lines.extend([
            f"### {model_name.replace('_', ' ').title()}",
            f"- Mean Accuracy: {model_metrics['accuracy'].mean():.3f} ± {model_metrics['accuracy'].std():.3f}",
            f"- Mean F1 Macro: {model_metrics['f1_macro'].mean():.3f} ± {model_metrics['f1_macro'].std():.3f}",
        ])
        
        if 'roc_auc' in model_metrics.columns:
            report_lines.append(f"- Mean ROC AUC: {model_metrics['roc_auc'].mean():.3f} ± {model_metrics['roc_auc'].std():.3f}")
        
        report_lines.append("")
    
    # Add insights
    best_model = metrics_df.groupby('model')['accuracy'].mean().idxmax()
    best_accuracy = metrics_df.groupby('model')['accuracy'].mean().max()
    
    report_lines.extend([
        "## Key Insights",
        f"- Best performing model: {best_model} (Accuracy: {best_accuracy:.3f})",
        f"- Number of models tested: {len(metrics_df['model'].unique())}",
        f"- Cross-validation folds: {len(metrics_df) // len(metrics_df['model'].unique())}",
        "",
        "## Files Generated",
        "- class_balance.csv: Class distribution before/after filtering",
        "- cv/metrics_per_fold.csv: Detailed CV metrics",
        "- cv/summary.json: CV summary statistics",
        "- shap/: SHAP analysis results",
        "- robustness/: Noise robustness analysis",
        "- hparams/: Hyperparameter search results",
        ""
    ])
    
    # Write report
    with open(output_path / 'metrics_overview.md', 'w') as f:
        f.write('\n'.join(report_lines))

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save resolved configuration to output directory."""
    with open(output_path / 'run_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_feature_names(n_features: int, signal_names: List[str] = None) -> List[str]:
    """Create feature names for the dataset."""
    if signal_names is None:
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA', 'TEMP']
    
    feature_names = []
    for signal in signal_names:
        # Add statistical features for each signal
        stats = ['mean', 'std', 'min', 'max', 'q25', 'q75', 'median']
        for stat in stats:
            feature_names.append(f"{signal}_{stat}")
    
    # If we have more features than expected, add generic names
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names)}")
    
    return feature_names[:n_features]

def validate_data(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate data format and basic properties."""
    # Check shapes
    if len(X.shape) != 2:
        logger.error(f"X should be 2D, got shape {X.shape}")
        return False
    
    if len(y.shape) != 1:
        logger.error(f"y should be 1D, got shape {y.shape}")
        return False
    
    # Check lengths match
    if len(X) != len(y):
        logger.error(f"X and y lengths don't match: {len(X)} vs {len(y)}")
        return False
    
    # Check for NaN values
    if np.isnan(X).any():
        logger.error("X contains NaN values")
        return False
    
    if np.isnan(y).any():
        logger.error("y contains NaN values")
        return False
    
    # Check data types
    if not np.issubdtype(X.dtype, np.floating):
        logger.warning(f"X is not float type: {X.dtype}")
    
    if not np.issubdtype(y.dtype, np.integer):
        logger.warning(f"y is not integer type: {y.dtype}")
    
    logger.info(f"Data validation passed: X shape {X.shape}, y shape {y.shape}")
    return True

def print_data_summary(X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str] = None) -> None:
    """Print a summary of the dataset."""
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"  Class {class_id}: {count} samples ({count/len(y)*100:.1f}%)")
    
    print(f"\nFeature statistics:")
    print(f"  Mean: {np.mean(X, axis=0).mean():.3f}")
    print(f"  Std: {np.std(X, axis=0).mean():.3f}")
    print(f"  Min: {np.min(X):.3f}")
    print(f"  Max: {np.max(X):.3f}")
    
    if feature_names:
        print(f"\nTop 10 features by variance:")
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {feature_names[idx]}: {variances[idx]:.3f}")
    
    print("=" * 50)

def main():
    """CLI entry point for utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Utility functions for WESAD-like data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", default="results", help="Path to prepared data")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    if (data_path / "X.npy").exists() and (data_path / "y.npy").exists():
        X = np.load(data_path / "X.npy")
        y = np.load(data_path / "y.npy")
        
        # Validate data
        if validate_data(X, y):
            # Create feature names
            feature_names = create_feature_names(X.shape[1])
            
            # Print summary
            print_data_summary(X, y, feature_names)
        else:
            logger.error("Data validation failed")
    else:
        logger.error("Data files not found")

if __name__ == "__main__":
    main() 