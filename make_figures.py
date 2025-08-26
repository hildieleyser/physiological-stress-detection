#!/usr/bin/env python3
"""
make_figures.py - Generate publication-ready figures from CSV data

Usage:
    python make_figures.py \
      --class-counts data/class_counts.csv \
      --metrics data/metrics_per_model.csv \
      --shap data/shap_importances.csv \
      --confusion data/confusion_matrix_best_model.csv \
      --outdir results/figures

Required CSVs:
    --class-counts: split,class,count
    --metrics: model,split,metric,value  
    --shap: feature,mean_abs_shap
    --confusion: square matrix with class names (optional)

Outputs:
    - class_distribution.png/pdf
    - model_accuracy.png/pdf  
    - shap_top_features.png/pdf
    - confusion_matrix.png/pdf (if provided)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Global style settings
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titleweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'grid.alpha': 0.3,
    'axes.facecolor': '#F8F9FA',
    'figure.facecolor': 'white'
})

# Color palette
COLORS = {
    'train': '#190B28',      # Dark purple
    'val': '#685762',        # Wenge  
    'test': '#9B9987',       # Battleship gray
    'LR': '#E55381',         # Blush
    'RF': '#EFA9AE',         # Cherry blossom
    'XGB': '#190B28'         # Dark purple
}

def validate_file_exists(filepath: Path, description: str) -> None:
    """Validate that a file exists and is readable."""
    if not filepath.exists():
        raise ValueError(f"{description} file not found: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"{description} path is not a file: {filepath}")

def validate_class_counts(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Validate class_counts CSV schema and data."""
    required_cols = ['split', 'class', 'count']
    
    # Check columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"class_counts missing required columns: {missing_cols}. Expected: {required_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"class_counts file is empty: {filepath}")
    
    # Normalize split names
    df['split'] = df['split'].str.lower()
    
    # Validate splits
    valid_splits = {'train', 'val', 'test'}
    invalid_splits = set(df['split'].unique()) - valid_splits
    if invalid_splits:
        raise ValueError(f"class_counts contains invalid splits: {invalid_splits}. Expected: {valid_splits}")
    
    # Check all splits present
    missing_splits = valid_splits - set(df['split'].unique())
    if missing_splits:
        raise ValueError(f"class_counts missing splits: {missing_splits}")
    
    # Validate counts
    if not (df['count'] >= 0).all():
        raise ValueError("class_counts contains negative counts")
    
    if not pd.api.types.is_numeric_dtype(df['count']):
        raise ValueError("class_counts 'count' column must be numeric")
    
    return df

def validate_metrics(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Validate metrics CSV schema and data."""
    required_cols = ['model', 'split', 'metric', 'value']
    
    # Check columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"metrics missing required columns: {missing_cols}. Expected: {required_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"metrics file is empty: {filepath}")
    
    # Normalize split names
    df['split'] = df['split'].str.lower()
    
    # Validate splits
    valid_splits = {'train', 'val', 'test'}
    invalid_splits = set(df['split'].unique()) - valid_splits
    if invalid_splits:
        raise ValueError(f"metrics contains invalid splits: {invalid_splits}. Expected: {valid_splits}")
    
    # Check all splits present
    missing_splits = valid_splits - set(df['split'].unique())
    if missing_splits:
        raise ValueError(f"metrics missing splits: {missing_splits}")
    
    # Filter for accuracy metric
    accuracy_df = df[df['metric'].str.lower() == 'accuracy'].copy()
    if accuracy_df.empty:
        raise ValueError("metrics file contains no 'accuracy' metric")
    
    # Validate values
    if not pd.api.types.is_numeric_dtype(accuracy_df['value']):
        raise ValueError("metrics 'value' column must be numeric")
    
    # Convert percentages to fractions if needed
    if (accuracy_df['value'] > 1).any():
        accuracy_df.loc[accuracy_df['value'] > 1, 'value'] /= 100
    
    # Validate value range
    if not ((accuracy_df['value'] >= 0) & (accuracy_df['value'] <= 1)).all():
        raise ValueError("metrics accuracy values must be in [0,1] or [0,100]")
    
    return accuracy_df

def validate_shap(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Validate SHAP CSV schema and data."""
    required_cols = ['feature', 'mean_abs_shap']
    
    # Check columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"shap missing required columns: {missing_cols}. Expected: {required_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"shap file is empty: {filepath}")
    
    # Validate mean_abs_shap
    if not pd.api.types.is_numeric_dtype(df['mean_abs_shap']):
        raise ValueError("shap 'mean_abs_shap' column must be numeric")
    
    # Remove NaN/inf values
    df_clean = df.dropna(subset=['mean_abs_shap'])
    if df_clean.empty:
        raise ValueError("shap file contains no valid mean_abs_shap values after removing NaN/inf")
    
    # Check for positive values
    if not (df_clean['mean_abs_shap'] > 0).any():
        raise ValueError("shap file contains no positive mean_abs_shap values")
    
    return df_clean

def validate_confusion(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Validate confusion matrix CSV schema and data."""
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"confusion file is empty: {filepath}")
    
    # Check if square matrix
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {df.shape}")
    
    # Check if numeric
    if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        raise ValueError("confusion matrix must contain only numeric values")
    
    # Check for negative values
    if (df < 0).any().any():
        raise ValueError("confusion matrix contains negative values")
    
    return df

def plot_class_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """Plot class distribution per split."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Pivot data for plotting
    pivot_df = df.pivot(index='class', columns='split', values='count')
    
    # Create grouped bar chart
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, pivot_df['train'], width, label='Train', 
                   color=COLORS['train'], alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, pivot_df['val'], width, label='Val', 
                   color=COLORS['val'], alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, pivot_df['test'], width, label='Test', 
                   color=COLORS['test'], alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontsize=12, fontweight='normal')
    
    # Add totals text
    train_total = df[df['split'] == 'train']['count'].sum()
    val_total = df[df['split'] == 'val']['count'].sum()
    test_total = df[df['split'] == 'test']['count'].sum()
    
    ax.text(0.02, 0.98, f'Train: {train_total} | Val: {val_total} | Test: {test_total}', 
           transform=ax.transAxes, fontsize=14, fontweight='normal',
           verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor='gray', alpha=0.8))
    
    # Customize plot
    ax.set_title('Dataset & Class Counts (Train/Val/Test)', fontsize=20, fontweight='normal', pad=20)
    ax.set_xlabel('Classes', fontsize=14, fontweight='normal')
    ax.set_ylabel('Count', fontsize=14, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45 if len(pivot_df.index) > 3 else 0)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(outdir / 'class_distribution.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'class_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class distribution saved")
    print(f"  Totals: Train={train_total}, Val={val_total}, Test={test_total}")

def plot_model_accuracy(df: pd.DataFrame, outdir: Path) -> None:
    """Plot model accuracy by split."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Pivot data for plotting
    pivot_df = df.pivot(index='model', columns='split', values='value')
    
    # Create grouped bar chart
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, pivot_df['train'] * 100, width, label='Train', 
                   color=COLORS['train'], alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, pivot_df['val'] * 100, width, label='Val', 
                   color=COLORS['val'], alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, pivot_df['test'] * 100, width, label='Test', 
                   color=COLORS['test'], alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=12, fontweight='normal')
    
    # Find best test accuracy
    best_test_idx = pivot_df['test'].idxmax()
    best_test_acc = pivot_df.loc[best_test_idx, 'test'] * 100
    
    # Add horizontal line for best test accuracy
    ax.axhline(y=best_test_acc, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.02, best_test_acc + 1, f'Best Test: {best_test_idx} ({best_test_acc:.1f}%)', 
           fontsize=12, fontweight='normal', color='red',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor='red', alpha=0.8))
    
    # Customize plot
    ax.set_title('Accuracy by Model & Split', fontsize=20, fontweight='normal', pad=20)
    ax.set_xlabel('Models', fontsize=14, fontweight='normal')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(outdir / 'model_accuracy.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'model_accuracy.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model accuracy saved")
    print(f"  Best test accuracy: {best_test_idx} ({best_test_acc:.1f}%)")

def plot_shap_features(df: pd.DataFrame, outdir: Path) -> None:
    """Plot SHAP top features."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Sort by mean_abs_shap and get top 5
    top_features = df.nlargest(5, 'mean_abs_shap')
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_features['mean_abs_shap'],
                   color='#190B28', alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['mean_abs_shap'])):
        ax.text(value + max(top_features['mean_abs_shap']) * 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}', ha='left', va='center', 
               fontsize=12, fontweight='normal')
    
    # Customize plot
    ax.set_title('Top Features (SHAP Mean |Contribution|)', fontsize=20, fontweight='normal', pad=20)
    ax.set_xlabel('Mean |SHAP value|', fontsize=14, fontweight='normal')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(outdir / 'shap_top_features.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'shap_top_features.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ SHAP features saved")
    print(f"  Top 5 features: {', '.join(top_features['feature'].tolist())}")

def plot_confusion_matrix(df: pd.DataFrame, outdir: Path) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Determine if matrix is normalized
    row_sums = df.sum(axis=1)
    is_normalized = all(abs(row_sum - 1.0) < 0.01 for row_sum in row_sums)
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.2f' if is_normalized else 'd', 
                cmap='Blues', cbar=True, ax=ax)
    
    # Customize plot
    ax.set_title('Confusion Matrix (Test) — Best Model', fontsize=20, fontweight='normal', pad=20)
    ax.set_xlabel('Predicted', fontsize=14, fontweight='normal')
    ax.set_ylabel('Actual', fontsize=14, fontweight='normal')
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(outdir / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'confusion_matrix.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved")
    if is_normalized:
        # Calculate overall accuracy from diagonal
        overall_acc = np.trace(df.values) / len(df) * 100
        print(f"  Overall accuracy: {overall_acc:.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_figures.py \\
    --class-counts data/class_counts.csv \\
    --metrics data/metrics_per_model.csv \\
    --shap data/shap_importances.csv \\
    --confusion data/confusion_matrix.csv \\
    --outdir results/figures
        """
    )
    
    parser.add_argument('--class-counts', required=True, type=Path,
                       help='CSV with columns: split,class,count')
    parser.add_argument('--metrics', required=True, type=Path,
                       help='CSV with columns: model,split,metric,value')
    parser.add_argument('--shap', required=True, type=Path,
                       help='CSV with columns: feature,mean_abs_shap')
    parser.add_argument('--confusion', type=Path,
                       help='Optional: square matrix CSV with class names')
    parser.add_argument('--outdir', default=Path('results/figures'), type=Path,
                       help='Output directory (default: results/figures)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and validate class counts
        print("Loading and validating class counts...")
        validate_file_exists(args.class_counts, "Class counts")
        class_counts_df = pd.read_csv(args.class_counts)
        class_counts_df = validate_class_counts(class_counts_df, args.class_counts)
        
        # Load and validate metrics
        print("Loading and validating metrics...")
        validate_file_exists(args.metrics, "Metrics")
        metrics_df = pd.read_csv(args.metrics)
        metrics_df = validate_metrics(metrics_df, args.metrics)
        
        # Load and validate SHAP
        print("Loading and validating SHAP data...")
        validate_file_exists(args.shap, "SHAP")
        shap_df = pd.read_csv(args.shap)
        shap_df = validate_shap(shap_df, args.shap)
        
        # Load and validate confusion matrix (if provided)
        confusion_df = None
        if args.confusion:
            print("Loading and validating confusion matrix...")
            validate_file_exists(args.confusion, "Confusion matrix")
            confusion_df = pd.read_csv(args.confusion, index_col=0)
            confusion_df = validate_confusion(confusion_df, args.confusion)
        
        # Generate figures
        print("\nGenerating figures...")
        
        plot_class_distribution(class_counts_df, args.outdir)
        plot_model_accuracy(metrics_df, args.outdir)
        plot_shap_features(shap_df, args.outdir)
        
        if confusion_df is not None:
            plot_confusion_matrix(confusion_df, args.outdir)
        else:
            print("✓ Confusion matrix skipped (not provided)")
        
        print(f"\n✓ All figures saved to: {args.outdir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 